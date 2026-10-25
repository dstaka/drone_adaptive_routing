#!/usr/bin/env python
"""Plot a hash-verified diagnostic failure; never infer operational acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptive_drone_patrol.util import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--font',default='outputs/assets/NotoSansJP.ttf',type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the previous plot')
    if sha256(args.samples)!=args.sha256:
        raise ValueError('Diagnostic raw hash differs')
    rows=[json.loads(line) for line in args.samples.read_text().splitlines()]
    t=np.array([r['time_s'] for r in rows])
    if len(rows)<2 or not np.allclose(t,np.arange(len(t))*.01,atol=1e-9,rtol=0):
        raise ValueError('Incomplete diagnostic sample timing')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=str(args.font))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    figure,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    target=np.select([t<24,t<36],[1.,3.],default=1.)
    axes[0].step(t,target,where='post',label='機体の指令高度',color='#556677',linestyle='--')
    axes[0].plot(t,[r['drone_xyz'][2] for r in rows],label='機体中心',color='#17649a')
    axes[0].plot(t,[r['package_floor_m'] for r in rows],label='荷物の実最下点',color='#d47615')
    axes[0].axhline(.1,color='#b62732',label='安全下限0.10 m')
    axes[0].scatter(t[-1],rows[-1]['package_floor_m'],color='#b62732',zorder=5)
    axes[0].set_ylabel('高度 / 床余裕 [m]',fontproperties=font)
    axes[0].legend(prop=font,ncols=2)
    for body,label,color in [('drone','機体','#17649a'),('package','荷物','#d47615')]:
        axes[1].plot(t,[np.linalg.norm(r[body+'_velocity']) for r in rows],label=label,color=color)
    axes[1].set_ylabel('並進速度 [m/s]',fontproperties=font)
    axes[1].legend(prop=font)
    axes[2].semilogy(t,np.maximum([r['constraint_residual_m'] for r in rows],1e-10),color='#247b57',label='実attachment長残差')
    axes[2].axhline(2e-5,color='#b62732',linestyle='--',label='事前閾値2e-5 m')
    axes[2].set_ylabel('拘束残差 [m]',fontproperties=font)
    axes[2].set_xlabel('実シミュレーション時刻 [s]',fontproperties=font)
    axes[2].legend(prop=font)
    for axis in axes:
        axis.axvline(36,color='#555555',alpha=.4,linestyle=':')
        axis.set_xlim(0,t[-1])
    figure.suptitle('診断PD・質量比0.30・一辺0.030 m・初期揺れ0度\n39.35秒で荷物床余裕違反（学習方策の受入ではない）',fontproperties=font)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.out,dpi=160)
    plt.close(figure)
    print(args.out,sha256(args.out))


if __name__=='__main__':
    main()
