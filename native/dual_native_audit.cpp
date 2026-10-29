// Unqualified shadow-only prototype. Never linked into the production runner.
// ABI1: metrics[81][23], states[81][54], boundary_geometry[17][ngeom][12].
// RK slots0..63; boundary slots64..80. All calculations use mjtNum == double.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static_assert(sizeof(mjtNum) == sizeof(double), "double MuJoCo required");
static_assert(mjNWARNING==7, "ABI1 warning count differs");
namespace {
constexpr double KF=3.16e-10, KM=7.94e-12, HOVER=14468.429183500699;
constexpr double PI=3.14159265358979323846, INF=std::numeric_limits<double>::infinity();
constexpr int WIDTH=23, STATE=54;
thread_local std::string error;
std::mutex callback_mutex;
struct ModelDelete { void operator()(mjModel* p) const { if(p) mj_deleteModel(p); } };
struct DataDelete { void operator()(mjData* p) const { if(p) mj_deleteData(p); } };
using Model=std::unique_ptr<mjModel,ModelDelete>;
using Data=std::unique_ptr<mjData,DataDelete>;
void require(bool yes,const char* message) { if(!yes) throw std::runtime_error(message); }
void no_callbacks() {
  require(!mjcb_control && !mjcb_passive && !mjcb_contactfilter && !mjcb_act_dyn && !mjcb_act_gain && !mjcb_act_bias && !mjcb_sensor,
          "existing native callback unsupported");
}
bool finite(const mjtNum* a,int n) { for(int i=0;i<n;++i) if(!std::isfinite(a[i])) return false; return true; }
bool zero(const mjtNum* a,int n) { for(int i=0;i<n;++i) if(a[i]!=0) return false; return true; }
double norm3(const double* a) { return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
double dot3(const double* a,const double* b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
double rotation_angle(const double* a,const double* b) {
  double relative[9]{};
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) relative[3*i+j]+=a[3*k+i]*b[3*k+j];
  double skew[3]={(relative[7]-relative[5])/2,(relative[2]-relative[6])/2,(relative[3]-relative[1])/2};
  return std::atan2(norm3(skew),std::clamp((relative[0]+relative[4]+relative[8]-1)/2,-1.,1.));
}
bool proper_rotation(const double* a) {
  if(!finite(a,9)) return false;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    double value=0;for(int k=0;k<3;++k)value+=a[3*k+i]*a[3*k+j];
    if(std::abs(value-(i==j?1.:0.))>1e-8)return false;
  }
  double determinant=a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
  return std::abs(determinant-1)<1e-8;
}
void rotation_midpoint_small(const double* a,const double* b,double* out) {
  double theta=rotation_angle(a,b);
  if(theta<1e-14) {std::copy(a,a+9,out);return;}
  double relative[9]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)relative[3*i+j]+=a[3*k+i]*b[3*k+j];
  double axis[3]={relative[7]-relative[5],relative[2]-relative[6],relative[3]-relative[1]};
  double norm=norm3(axis);for(double& x:axis)x/=norm;
  double cross[9]={0,-axis[2],axis[1],axis[2],0,-axis[0],-axis[1],axis[0],0},half[9]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    double square=0;for(int k=0;k<3;++k)square+=cross[3*i+k]*cross[3*k+j];
    half[3*i+j]=(i==j?1.:0.)+std::sin(theta/2)*cross[3*i+j]+(1-std::cos(theta/2))*square;
  }
  std::fill(out,out+9,0.);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)out[3*i+j]+=a[3*i+k]*half[3*k+j];
}
int owner_name(const char* name) {
  if(!name) return -1;
  if(std::string(name).rfind("drone-1/",0)==0) return 0;
  if(std::string(name).rfind("drone-2/",0)==0) return 1;
  return -1;
}
int id(const mjModel* m,int kind,const std::string& name) {
  int x=mj_name2id(m,kind,name.c_str()); require(x>=0,"missing required named model metadata"); return x;
}
struct Shape { int owner=-1, rotor=0, kind=0, body=0; double radius=0; };
struct Context {
  Model model; Data data,scratch;
  std::vector<Shape> shapes;
  std::vector<std::tuple<int,int,double>> pairs;
  std::vector<std::pair<int,int>> self_pairs;
  int base[2]{}, payload[2]{}, motors[8]{};
  double package_half[2]{};
  std::array<double,81*WIDTH> metrics{};
  std::array<double,81*STATE> states{};
  std::vector<double> geometry;
  std::array<double,17*48> body_geometry{};
  int rk=0; bool failed=false; std::string callback_error;
  Context(const mjModel* source,const mjData* initial) {
    require(mj_version()==3015000 && mjVERSION_HEADER==3015000,"MuJoCo3.15.0 header/library required");
    model.reset(mj_copyModel(nullptr,source)); require(bool(model),"model copy failed");
    auto m=model.get();
    require(m->nq==30 && m->nv==24 && m->nu==8 && m->na==0,"unexpected dual dimensions");
    require(m->opt.integrator==mjINT_RK4 && std::abs(m->opt.timestep-.01/16)<1e-15,"native16 RK4 required");
    require(m->opt.disableactuator==0 && !(m->opt.disableflags & (mjDSBL_ACTUATION|mjDSBL_CONTACT)),"disabled actuator/contact authority");
    require(m->nplugin==0,"plugin callbacks are outside shadow contract");
    data.reset(mj_makeData(m)); scratch.reset(mj_makeData(m));
    require(bool(data)&&bool(scratch),"native allocation failed");
    mj_copyData(data.get(),m,initial);
    int prop[2][4]{};
    for(int d=0;d<2;++d) {
      const std::string p="drone-"+std::to_string(d+1)+"/";
      base[d]=id(m,mjOBJ_BODY,p+"base_link"); payload[d]=id(m,mjOBJ_BODY,p+"payload");
      id(m,mjOBJ_BODY,p+"suspension_rod");
      for(int i=0;i<4;++i) {
        prop[d][i]=id(m,mjOBJ_BODY,p+"prop"+std::to_string(i)+"_link");
        int a=motors[4*d+i]=id(m,mjOBJ_ACTUATOR,p+"motor"+std::to_string(i));
        const double gear[6]={0,0,1,0,0,(i%2?1.:-1.)*KM/KF};
        for(int k=0;k<6;++k) require(std::abs(m->actuator_gear[6*a+k]-gear[k])<1e-14,"changed motor authority gear");
        require(m->actuator_trntype[a]==mjTRN_SITE && m->actuator_dyntype[a]==mjDYN_NONE &&
          m->actuator_gaintype[a]==mjGAIN_FIXED && m->actuator_biastype[a]==mjBIAS_NONE &&
          m->actuator_gainprm[mjNGAIN*a]==1 && m->actuator_ctrllimited[a] &&
          m->actuator_ctrlrange[2*a]==0 && std::abs(m->actuator_ctrlrange[2*a+1]-KF*std::pow(1.8*HOVER,2))<1e-14,
          "changed original motor authority");
        require(m->site_bodyid[m->actuator_trnid[2*a]]==prop[d][i],"motor site ownership differs");
        int site=m->actuator_trnid[2*a];
        require(m->actuator_trnid[2*a+1]==-1 && !m->actuator_forcelimited[a],"changed motor authority limits/reference site");
        for(int k=0;k<3;++k) require(std::abs(m->site_pos[3*site+k]-m->body_ipos[3*prop[d][i]+k])<1e-14,"motor authority not at original COM");
        for(int k=0;k<4;++k) require(std::abs(m->site_quat[4*site+k]-(k==0?1.:0.))<1e-14,"motor authority site axis changed");
      }
    }
    for(int i=0;i<m->nexclude;++i) {
      int left=m->exclude_signature[i]>>16, right=m->exclude_signature[i]&0xffff;
      require(owner_name(mj_id2name(m,mjOBJ_BODY,left))==owner_name(mj_id2name(m,mjOBJ_BODY,right)),
              "cross-assembly contact exclusion forbidden");
    }
    int packages[2]={0,0}, covered[2]={0,0};
    for(int g=0;g<m->ngeom;++g) {
      Shape s; s.kind=m->geom_type[g]; s.body=m->geom_bodyid[g];
      if(m->geom_contype[g] || m->geom_conaffinity[g]) s.owner=owner_name(mj_id2name(m,mjOBJ_GEOM,g));
      const double* z=m->geom_size+3*g;
      if(s.owner>=0) {
        require(owner_name(mj_id2name(m,mjOBJ_BODY,s.body))==s.owner,"geometry body ownership mismatch");
        require(s.kind==mjGEOM_BOX || s.kind==mjGEOM_CYLINDER || s.kind==mjGEOM_CAPSULE || s.kind==mjGEOM_SPHERE,
                "unsupported covered collision geometry");
        s.radius=s.kind==mjGEOM_BOX?norm3(z):s.kind==mjGEOM_CYLINDER?std::hypot(z[0],z[1]):s.kind==mjGEOM_CAPSULE?z[0]+z[1]:z[0];
        for(int i=0;i<4;++i) s.rotor |= s.body==prop[s.owner][i];
        covered[s.owner]++;
        if(s.body==payload[s.owner]) { packages[s.owner]++; package_half[s.owner]=z[2]; }
      }
      shapes.push_back(s);
    }
    require(covered[0]>=7 && covered[1]>=7 && packages[0]==1 && packages[1]==1,"incomplete assembly surface coverage");
    for(int a=0;a<m->ngeom;++a) for(int b=0;b<m->ngeom;++b)
      if(shapes[a].owner==0 && shapes[b].owner==1) pairs.emplace_back(a,b,shapes[a].radius+shapes[b].radius);
    for(int a=0;a<m->ngeom;++a) for(int b=a+1;b<m->ngeom;++b) {
      if(shapes[a].owner<0 || shapes[a].owner!=shapes[b].owner) continue;
      int ba=m->geom_bodyid[a],bb=m->geom_bodyid[b],wa=m->body_weldid[ba],wb=m->body_weldid[bb];
      int pa=m->body_weldid[m->body_parentid[wa]],pb=m->body_weldid[m->body_parentid[wb]];
      if(wa==wb) continue;
      bool excluded=false;
      for(int i=0;i<m->nexclude;++i) excluded |= m->exclude_signature[i]==((std::min(ba,bb)<<16)|std::max(ba,bb));
      if(excluded || (!(m->opt.disableflags & mjDSBL_FILTERPARENT) && wa && wb && (wa==pb || wb==pa))) continue;
      if((m->geom_contype[a]&m->geom_conaffinity[b]) || (m->geom_contype[b]&m->geom_conaffinity[a])) self_pairs.emplace_back(a,b);
    }
    geometry.resize(17*m->ngeom*12);
  }
  double floor(int g,const mjData* d) const {
    const double* r=d->geom_xmat+9*g+6; const double* z=model->geom_size+3*g; double support;
    switch(shapes[g].kind) {
      case mjGEOM_BOX: support=std::abs(r[0])*z[0]+std::abs(r[1])*z[1]+std::abs(r[2])*z[2]; break;
      case mjGEOM_CYLINDER: support=z[0]*std::hypot(r[0],r[1])+z[1]*std::abs(r[2]); break;
      case mjGEOM_CAPSULE: support=z[0]+z[1]*std::abs(r[2]); break;
      default: support=z[0];
    }
    return d->geom_xpos[3*g+2]-support;
  }
  void point(const mjData* input,double* out,double* state) {
    auto m=model.get(); auto d=scratch.get();
    mj_resetData(m,d);
    mju_copy(d->qpos,input->qpos,30); mju_copy(d->qvel,input->qvel,24); mju_copy(d->ctrl,input->ctrl,8); d->time=input->time;
    // Callback dispatch ignores scratch data: forward cannot recursively audit.
    mj_forward(m,d);
    std::fill(out,out+WIDTH,0.); out[0]=input->time; out[1]=INF; out[2]=out[3]=INF;
    std::vector<std::tuple<double,int,int>> ordered;
    for(auto [a,b,radius]:pairs) {
      double delta[3]; for(int k=0;k<3;++k) delta[k]=d->geom_xpos[3*a+k]-d->geom_xpos[3*b+k];
      ordered.emplace_back(norm3(delta)-radius-1e-12,a,b);
    }
    std::stable_sort(ordered.begin(),ordered.end());
    int left=-1,right=-1;
    for(auto [lower,a,b]:ordered) {
      if(lower>out[1]) break;
      double distance=mj_geomDistance(m,d,a,b,100.,nullptr); ++out[14];
      require(std::isfinite(distance),"nonfinite native shape distance");
      if(distance<out[1] || (distance==out[1] && std::pair<int,int>(a,b)<std::pair<int,int>(left,right)))
        { out[1]=distance; left=a; right=b; }
    }
    out[10]=left; out[11]=right; out[12]=d->ncon;
    for(int g=0;g<m->ngeom;++g) if(shapes[g].owner>=0) out[2+shapes[g].owner]=std::min(out[2+shapes[g].owner],floor(g,d));
    for(int a=0;a<m->ngeom;++a) if(shapes[a].rotor) for(int b=0;b<m->ngeom;++b)
      if(shapes[b].owner>=0 && shapes[a].owner!=shapes[b].owner) {
        double dx=d->geom_xpos[3*b]-d->geom_xpos[3*a], dy=d->geom_xpos[3*b+1]-d->geom_xpos[3*a+1], dz=d->geom_xpos[3*b+2]-d->geom_xpos[3*a+2], r=shapes[b].radius;
        if(std::hypot(dx,dy)-r<.60 && dz-r<=0 && dz+r>=-1) ++out[13];
      }
    for(int a=0;a<2;++a) {
      const double* rd=d->xmat+9*base[a]; const double* rp=d->xmat+9*payload[a];
      double suspension[3];
      for(int k=0;k<3;++k) suspension[k]=d->xpos[3*payload[a]+k]+rp[3*k+2]*package_half[a]-d->xpos[3*base[a]+k]+rd[3*k+2]*.020;
      double length=norm3(suspension), velocity[6];
      mj_objectVelocity(m,d,mjOBJ_BODY,payload[a],velocity,0);
      out[4]=std::max(out[4],std::abs(length-.30));
      out[5]=std::max({out[5],std::abs(std::atan2(rd[7],rd[8]))*180/PI,std::abs(std::asin(std::clamp(-rd[6],-1.,1.)))*180/PI});
      out[6]=std::max(out[6],std::acos(std::clamp(-suspension[2]/length,-1.,1.))*180/PI);
      out[7]=std::max(out[7],norm3(velocity)*180/PI);
    }
    // Preserve the original auditor's live (potentially stale) qacc finite
    // check as well as independently recomputed scratch acceleration.
    out[22]=finite(input->qacc,24);
    out[8]=finite(d->qpos,30)&&finite(d->qvel,24)&&finite(d->qacc,24)&&out[22];
    unsigned flags=out[8]?0:1;
    if(!std::isfinite(out[1])||out[1]<.2) flags|=2;
    if(!std::isfinite(out[2])||!std::isfinite(out[3])||out[2]<=.1||out[3]<=.1) flags|=4;
    if(d->ncon) flags|=8; if(out[13]) flags|=16;
    if(!std::isfinite(out[4])||out[4]>2e-5) flags|=32;
    if(!std::isfinite(out[5])||out[5]>60) flags|=64;
    for(int w=0;w<mjNWARNING;++w) { out[15+w]=std::max(input->warning[w].number,d->warning[w].number); if(out[15+w]) flags|=128; }
    bool authority=zero(input->qfrc_applied,24)&&zero(input->xfrc_applied,6*m->nbody)&&finite(input->ctrl,8);
    for(int i=0;i<8;++i) authority &= input->ctrl[i]>=0 && input->ctrl[i]<=KF*std::pow(1.8*HOVER,2)+1e-14;
    if(!authority) flags|=256;
    out[9]=flags; mju_copy(state,input->qpos,30); mju_copy(state+30,input->qvel,24);
  }
  void boundary(int i) {
    point(data.get(),metrics.data()+(64+i)*WIDTH,states.data()+(64+i)*STATE);
    for(int g=0;g<model->ngeom;++g) {
      double* to=geometry.data()+(i*model->ngeom+g)*12;
      mju_copy(to,data->geom_xpos+3*g,3); mju_copy(to+3,data->geom_xmat+9*g,9);
    }
    for(int d=0;d<2;++d) for(int b=0;b<2;++b) {
      int body=b?payload[d]:base[d]; double* to=body_geometry.data()+i*48+(2*d+b)*12;
      mju_copy(to,data->xpos+3*body,3); mju_copy(to+3,data->xmat+9*body,9);
    }
  }
  double floor_pose(int g,const double* pose) {
    const double* r=pose+9;const double* z=model->geom_size+3*g;double support;
    switch(shapes[g].kind) {
      case mjGEOM_BOX:support=std::abs(r[0])*z[0]+std::abs(r[1])*z[1]+std::abs(r[2])*z[2];break;
      case mjGEOM_CYLINDER:support=z[0]*std::hypot(r[0],r[1])+z[1]*std::abs(r[2]);break;
      case mjGEOM_CAPSULE:support=z[0]+z[1]*std::abs(r[2]);break;
      default:support=z[0];
    }
    return pose[2]-support;
  }
  void fast_interval(const double* ga,const double* gb,const double* ba,const double* bb,double* out) {
    // Root-cell sufficient conditions only. Every unresolved/near case returns
    // certified=0 for unchanged Python subdivision, never an optimistic result.
    std::fill(out,out+8,0.);out[1]=out[2]=out[7]=INF;
    int n=model->ngeom;
    require(finite(ga,n*12)&&finite(gb,n*12)&&finite(ba,48)&&finite(bb,48),"finite interval poses required");
    for(int g=0;g<n;++g) require(proper_rotation(ga+12*g+3)&&proper_rotation(gb+12*g+3),"proper interval geometry rotation required");
    for(int b=0;b<4;++b) require(proper_rotation(ba+12*b+3)&&proper_rotation(bb+12*b+3),"proper interval body rotation required");
    std::vector<double> movement(n),da(pairs.size()+self_pairs.size()),db(da.size());
    std::vector<std::pair<int,int>> all;for(auto [a,b,r]:pairs)all.emplace_back(a,b);
    all.insert(all.end(),self_pairs.begin(),self_pairs.end());
    for(int g=0;g<n;++g) {
      double delta[3];for(int k=0;k<3;++k)delta[k]=gb[12*g+k]-ga[12*g+k];
      double angle=rotation_angle(ga+12*g+3,gb+12*g+3);
      if(angle>.05)return;
      movement[g]=norm3(delta)+shapes[g].radius*angle;
    }
    for(int endpoint=0;endpoint<2;++endpoint) {
      const double* poses=endpoint?gb:ga;auto& distances=endpoint?db:da;
      for(int g=0;g<n;++g) {mju_copy(scratch->geom_xpos+3*g,poses+12*g,3);mju_copy(scratch->geom_xmat+9*g,poses+12*g+3,9);}
      for(size_t p=0;p<all.size();++p) {
        auto [a,b]=all[p];double delta[3];for(int k=0;k<3;++k)delta[k]=poses[12*a+k]-poses[12*b+k];
        double lower=norm3(delta)-shapes[a].radius-shapes[b].radius-1e-12,margin=p<pairs.size()?.2:0.;
        distances[p]=lower;
        if(lower<margin) {distances[p]=mj_geomDistance(model.get(),scratch.get(),a,b,100.,nullptr);out[6]++;}
        if(!std::isfinite(distances[p]) || distances[p]<margin)return;
        if(p<pairs.size())out[7]=std::min(out[7],distances[p]);
      }
    }
    for(size_t p=0;p<all.size();++p) {
      auto [a,b]=all[p];double lower=std::min(da[p],db[p])-(movement[a]+movement[b])/2-1e-12;
      // Extra fast-path slack only requests fallback near floating thresholds;
      // the unchanged Python certificate retains the exact original limits.
      if(lower<(p<pairs.size()?.2:0.)+1e-10)return;
      out[p<pairs.size()?1:2]=std::min(out[p<pairs.size()?1:2],lower);
    }
    for(int g=0;g<n;++g) if(shapes[g].owner>=0)
      if(std::min(floor_pose(g,ga+12*g),floor_pose(g,gb+12*g))-movement[g]/2-1e-12<=.1+1e-10)return;
    for(int a=0;a<n;++a)if(shapes[a].rotor)for(int b=0;b<n;++b)if(shapes[b].owner>=0 && shapes[a].owner!=shapes[b].owner) {
      double u[3],v[3];for(int k=0;k<3;++k){u[k]=ga[12*b+k]-ga[12*a+k];v[k]=gb[12*b+k]-gb[12*a+k];}
      double dx=v[0]-u[0],dy=v[1]-u[1],squared=dx*dx+dy*dy;
      double f=squared>0?std::clamp(-(u[0]*dx+u[1]*dy)/squared,0.,1.):0.,r=shapes[b].radius;
      if(std::hypot(u[0]+f*dx,u[1]+f*dy)-r<.60+1e-10 && std::min(u[2],v[2])-r<=1e-10 && std::max(u[2],v[2])+r>=-1-1e-10)return;
    }
    for(int drone=0;drone<2;++drone) {
      const double* a=ba+24*drone;const double* b=bb+24*drone;double mid[24];
      double angles[2];
      for(int body=0;body<2;++body) {
        angles[body]=rotation_angle(a+12*body+3,b+12*body+3)+1e-12;if(angles[body]>.05)return;
        for(int k=0;k<3;++k)mid[12*body+k]=(a[12*body+k]+b[12*body+k])/2;
        rotation_midpoint_small(a+12*body+3,b+12*body+3,mid+12*body+3);
      }
      auto vector=[&](const double* p,double* v){for(int k=0;k<3;++k)v[k]=p[12+k]+p[15+3*k+2]*package_half[drone]-p[k]+p[3+3*k+2]*.020;};
      for(const double* p:{a,b,static_cast<const double*>(mid)}) {
        double v[3];vector(p,v);double residual=std::abs(norm3(v)-.30),tilt=std::max(std::abs(std::atan2(p[10],p[11])),std::abs(std::asin(std::clamp(-p[9],-1.,1.))))*180/PI;
        if(!std::isfinite(residual)||!std::isfinite(tilt)||residual>2e-5||tilt>60)return;
        out[3]=std::max(out[3],residual);out[4]=std::max(out[4],tilt);
      }
      double half=angles[0]/2,pitch=std::abs(std::asin(std::clamp(-mid[9],-1.,1.)))+half+1e-12;
      double roll=pitch<PI/2?std::abs(std::atan2(mid[10],mid[11]))+half/std::cos(pitch)+1e-12:INF;
      double tilt=std::max(roll,pitch)*180/PI,va[3],vb[3],delta[3],near[3];vector(a,va);vector(b,vb);
      for(int k=0;k<3;++k)delta[k]=vb[k]-va[k];double squared=dot3(delta,delta);
      double f=squared?std::clamp(-dot3(va,delta)/squared,0.,1.):0.;for(int k=0;k<3;++k)near[k]=va[k]+f*delta[k];
      double deviation=(.020*angles[0]*angles[0]+package_half[drone]*angles[1]*angles[1])/8;
      double lower=std::max(0.,norm3(near)-deviation-1e-12),upper=std::max(norm3(va),norm3(vb))+deviation+1e-12;
      double residual=std::max(std::abs(lower-.30),std::abs(upper-.30));
      if(!std::isfinite(tilt)||!std::isfinite(residual)||tilt>60-1e-8||residual>2e-5-1e-10)return;
      out[3]=std::max(out[3],residual);out[4]=std::max(out[4],tilt);out[5]++;
    }
    out[0]=1;
  }
};
thread_local Context* active=nullptr;
void callback(const mjModel* m,mjData* d) {
  auto c=active;
  if(!c || m!=c->model.get() || d!=c->data.get() || !c->callback_error.empty()) return;
  try { require(c->rk<64,"unexpected RK callback count"); c->point(d,c->metrics.data()+c->rk*WIDTH,c->states.data()+c->rk*STATE); ++c->rk; }
  catch(const std::exception& e) { c->callback_error=e.what(); }
}
struct CallbackGuard {
  CallbackGuard(Context* c) { no_callbacks(); active=c; mjcb_control=callback; }
  ~CallbackGuard() { mjcb_control=nullptr; active=nullptr; }
};
template<class F> int guarded(F f) { try { error.clear(); f(); return 0; } catch(const std::exception& e) { error=e.what(); return -1; } }
}
extern "C" {
const char* da_error() { return error.c_str(); }
int da_version() { return mj_version(); }
void* da_create(const mjModel* m,const mjData* d) {
  Context* result=nullptr; guarded([&]{ require(m&&d,"model/data pointers required"); std::lock_guard<std::mutex> lock(callback_mutex); no_callbacks(); result=new Context(m,d); }); return result;
}
void* da_create_xml(const char* path) {
  Context* result=nullptr; guarded([&]{ std::lock_guard<std::mutex> lock(callback_mutex); no_callbacks(); char msg[2048]{}; Model m(mj_loadXML(path,nullptr,msg,sizeof(msg))); require(bool(m),msg);
    // Same authored runtime angular damping reconstructed by NativePointAuditor.
    for(int drone=1;drone<=2;++drone) {
      std::string prefix="drone-"+std::to_string(drone)+"/";
      int joint=id(m.get(),mjOBJ_JOINT,prefix+"vehicle_free");
      double damping=(.027+m->body_mass[id(m.get(),mjOBJ_BODY,prefix+"suspension_rod")]+m->body_mass[id(m.get(),mjOBJ_BODY,prefix+"payload")])*1e-5;
      for(int k=3;k<6;++k) m->dof_damping[m->jnt_dofadr[joint]+k]=damping;
    }
    Data d(mj_makeData(m.get())); mj_forward(m.get(),d.get()); result=new Context(m.get(),d.get()); }); return result;
}
void da_destroy(Context* c) { delete c; }
int da_ngeom(Context* c) { return c->model->ngeom; }
int da_state_size(Context* c) { return mj_stateSize(c->model.get(),mjSTATE_INTEGRATION); }
int da_model_size(Context* c) { return static_cast<int>(mj_sizeModel(c->model.get())); }
int da_model_bytes(Context* c,unsigned char* out) { return guarded([&]{mj_saveModel(c->model.get(),nullptr,out,mj_sizeModel(c->model.get()));}); }
int da_get_state(Context* c,double* state,int* warnings) { return guarded([&]{
  require(!c->failed,"failed native step cannot be checkpointed as complete");
  mj_getState(c->model.get(),c->data.get(),state,mjSTATE_INTEGRATION);
  for(int i=0;i<mjNWARNING;++i){warnings[2*i]=c->data->warning[i].lastinfo;warnings[2*i+1]=c->data->warning[i].number;}
}); }
int da_set_state(Context* c,const double* state,const int* warnings,const double* rpm) { return guarded([&]{
  std::lock_guard<std::mutex> lock(callback_mutex);no_callbacks();int count=mj_stateSize(c->model.get(),mjSTATE_INTEGRATION);
  require(finite(state,count),"finite complete integration state required");
  for(int i=0;i<mjNWARNING;++i)require(warnings[2*i]>=0&&warnings[2*i+1]>=0,"nonnegative warning state required");
  Data check(mj_makeData(c->model.get()));mj_setState(c->model.get(),check.get(),state,mjSTATE_INTEGRATION);
  require(check->time>=0 && zero(check->qfrc_applied,24)&&zero(check->xfrc_applied,6*c->model->nbody),"checkpoint has undeclared time/applied forces");
  for(int i=0;i<8;++i)require(std::isfinite(rpm[i])&&rpm[i]>=0&&rpm[i]<=1.8*HOVER && std::abs(check->ctrl[c->motors[i]]-KF*(rpm[i]*rpm[i]))<=1e-12,"checkpoint RPM/control authority mismatch");
  for(int j=0;j<c->model->njnt;++j) {
    int type=c->model->jnt_type[j];if(type!=mjJNT_FREE&&type!=mjJNT_BALL)continue;
    const double* q=check->qpos+c->model->jnt_qposadr[j]+(type==mjJNT_FREE?3:0);
    require(std::abs(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]-1)<1e-10,"checkpoint unit quaternion required");
  }
  mj_resetData(c->model.get(),c->data.get());mj_setState(c->model.get(),c->data.get(),state,mjSTATE_INTEGRATION);
  mj_forward(c->model.get(),c->data.get());
  // Reapply integration fields so any forward-solver warmstart updates cannot
  // replace the exact checkpoint values. Derived getters have been rebuilt.
  mj_setState(c->model.get(),c->data.get(),state,mjSTATE_INTEGRATION);
  for(int i=0;i<mjNWARNING;++i){c->data->warning[i].lastinfo=warnings[2*i];c->data->warning[i].number=warnings[2*i+1];}
  c->failed=false;
}); }
int da_copy_to(Context* c,const mjModel* model,mjData* data) { return guarded([&]{
  std::lock_guard<std::mutex> lock(callback_mutex);no_callbacks();require(!c->failed,"failed native step cannot publish data");
  auto size=mj_sizeModel(c->model.get());require(size==mj_sizeModel(model),"target runtime model differs");
  std::vector<unsigned char> a(size),b(size);mj_saveModel(c->model.get(),nullptr,a.data(),size);mj_saveModel(model,nullptr,b.data(),size);
  require(a==b,"target runtime model differs");mj_copyData(data,model,c->data.get());
}); }
int da_self_count(Context* c) { return static_cast<int>(c->self_pairs.size()); }
int da_self_pairs(Context* c,int* out) { return guarded([&]{ for(size_t i=0;i<c->self_pairs.size();++i) { out[2*i]=c->self_pairs[i].first; out[2*i+1]=c->self_pairs[i].second; } }); }
int da_package_offsets(Context* c,double* out) { return guarded([&]{ for(int d=0;d<2;++d) { out[3*d]=out[3*d+1]=0.; out[3*d+2]=c->package_half[d]; } }); }
int da_body_geometry(Context* c,double* out) { return guarded([&]{ std::copy(c->body_geometry.begin(),c->body_geometry.end(),out); }); }
int da_distance(Context* c,const double* positions,const double* rotations,int a,int b,double* out) {
  return guarded([&]{ std::lock_guard<std::mutex> lock(callback_mutex); no_callbacks(); int n=c->model->ngeom;
    require(a>=0 && a<n && b>=0 && b<n && finite(positions,3*n) && finite(rotations,9*n),"finite geometry query required");
    mju_copy(c->scratch->geom_xpos,positions,3*n);mju_copy(c->scratch->geom_xmat,rotations,9*n);
    *out=mj_geomDistance(c->model.get(),c->scratch.get(),a,b,100.,nullptr);
    require(std::isfinite(*out),"nonfinite native geometry query");
  });
}
int da_fast_interval(Context* c,const double* ga,const double* gb,const double* ba,const double* bb,double* out) {
  return guarded([&]{std::lock_guard<std::mutex> lock(callback_mutex);no_callbacks();c->fast_interval(ga,gb,ba,bb,out);});
}
int da_shapes(Context* c,double* out) { return guarded([&]{ for(int g=0;g<c->model->ngeom;++g) { auto s=c->shapes[g]; double* x=out+7*g; x[0]=s.owner; x[1]=s.rotor; x[2]=s.kind; for(int k=0;k<3;++k) x[3+k]=c->model->geom_size[3*g+k]; x[6]=s.body; } }); }
int da_inspect(Context* c,double* metrics,double* state) { return guarded([&]{ std::lock_guard<std::mutex> lock(callback_mutex); no_callbacks(); c->point(c->data.get(),metrics,state); }); }
int da_reset(Context* c,const double* qpos,const double* qvel,double time) { return guarded([&]{ std::lock_guard<std::mutex> lock(callback_mutex); no_callbacks(); require(finite(qpos,30)&&finite(qvel,24)&&std::isfinite(time)&&time>=0,"finite shadow reset required"); mj_resetData(c->model.get(),c->data.get()); mju_copy(c->data->qpos,qpos,30); mju_copy(c->data->qvel,qvel,24); c->data->time=time; mj_forward(c->model.get(),c->data.get()); c->failed=false; }); }
int da_step(Context* c,const double* rpm,double* metrics,double* states,double* geometry) {
  return guarded([&]{
    std::lock_guard<std::mutex> lock(callback_mutex);
    require(!c->failed,"failed shadow requires reset/reconstruction");
    for(int i=0;i<8;++i) require(std::isfinite(rpm[i]) && rpm[i]>=0 && rpm[i]<=1.8*HOVER,"original RPM authority required");
    CallbackGuard guard(c); c->rk=0; c->callback_error.clear(); c->failed=true;
    // Match the native Python command map KF * rpm**2, including operation order.
    for(int i=0;i<8;++i) c->data->ctrl[c->motors[i]]=KF*(rpm[i]*rpm[i]);
    // Match the original boundary forward schedule; callback only observes mj_step.
    mjcb_control=nullptr; c->boundary(0);
    for(int i=0;i<16;++i) {
      mjcb_control=callback; mj_step(c->model.get(),c->data.get());
      require(c->callback_error.empty(),c->callback_error.c_str());
      mjcb_control=nullptr; mj_forward(c->model.get(),c->data.get()); c->boundary(i+1);
    }
    mj_forward(c->model.get(),c->data.get()); require(c->rk==64,"exact64 RK points required");
    std::copy(c->metrics.begin(),c->metrics.end(),metrics); std::copy(c->states.begin(),c->states.end(),states);
    std::copy(c->geometry.begin(),c->geometry.end(),geometry); c->failed=false;
  });
}
}
