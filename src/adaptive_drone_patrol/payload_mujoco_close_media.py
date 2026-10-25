"""Close camera replay only: recorded native poses, never physics integration."""

from pathlib import Path
import json
import numpy as np
from functools import lru_cache
from .payload_solver_media import select_software_egl_device, goal_display_style


def load_close_camera(path="configs/payload_mujoco_close_camera.json"):
    c = json.loads(Path(path).read_text())
    if c["schema"] != "payload-mujoco-close-camera-v1" or c["fps"] != 100:
        raise ValueError("Frozen100Hz close camera required")
    if (c["main"]["width"], c["main"]["height"]) != (c["raw_width"], c["raw_height"]):
        raise ValueError("Main viewport must cover raw canvas")
    x, y = c["inset_xy"]
    w, h = c["overview"]["width"], c["overview"]["height"]
    if min(x, y) < 0 or x + w > c["raw_width"] or y + h > c["raw_height"]:
        raise ValueError("Overview must fit inside main viewport")
    return c


class CloseRenderer:
    def __init__(self, model_path, config):
        self.device = select_software_egl_device()
        import mujoco as mj
        from OpenGL import GL

        self.mj = mj
        self.config = config
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.model.vis.global_.offwidth = config["raw_width"]
        self.model.vis.global_.offheight = config["raw_height"]
        self.model.vis.quality.offsamples = 4
        self.model.vis.headlight.ambient[:] = [0.55] * 3
        self.model.vis.headlight.diffuse[:] = [0.75] * 3
        for name, color in [("ground", [0.28, 0.35, 0.44, 1]), ("vehicle_collision", [0, 0, 0, 0])]:
            i = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            if i >= 0:
                self.model.geom_rgba[i] = color
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH:
                self.model.geom_rgba[i] = [0.8, 0.88, 0.98, 1]
        self.renderers = {
            name: mj.Renderer(self.model, height=config[name]["height"], width=config[name]["width"])
            for name in ("main", "overview")
        }
        self.renderer_string = GL.glGetString(GL.GL_RENDERER).decode()
        if "llvmpipe" not in self.renderer_string.lower():
            self.close()
            raise ValueError("Software renderer required")
        self.bounds = []

    def close(self):
        for r in getattr(self, "renderers", {}).values():
            r.close()

    def render(self, raw, goals, priorities, completed, target_id):
        mj = self.mj
        c = self.config
        for key, width in [("qpos", self.model.nq), ("qvel", self.model.nv)]:
            a = np.asarray(raw[key], float)
            if a.shape != (width,) or not np.isfinite(a).all():
                raise ValueError("Invalid recorded native pose")
            getattr(self.data, key)[:] = a
        self.data.time = raw["time_s"]
        mj.set_mjcb_control(None)
        mj.mj_forward(self.model, self.data)
        if not np.array_equal(self.data.qpos, raw["qpos"]) or not np.array_equal(self.data.qvel, raw["qvel"]):
            raise ValueError("Display changed native state")
        for body, name in [("drone", "base_link"), ("package", "payload")]:
            i = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if not np.allclose(self.data.xpos[i], raw[body + "_xyz"], atol=1e-8, rtol=0):
                raise ValueError("Getter mismatch")
        views = {}
        for name, r in self.renderers.items():
            cam = mj.MjvCamera()
            mj.mjv_defaultCamera(cam)
            setting = c[name]
            cam.lookat[:] = (
                setting["lookat"]
                if name == "overview"
                else (np.asarray(raw["drone_xyz"]) + raw["package_xyz"]) / 2
            )
            for key in ("distance", "azimuth", "elevation"):
                setattr(cam, key, setting[key])
            r.update_scene(self.data, camera=cam)
            if name == "overview":
                for i, (goal, priority) in enumerate(zip(goals, priorities)):
                    label = chr(65 + i)
                    style = goal_display_style(
                        priority, visited=label in completed, selected=label == target_id
                    )
                    scene = r.scene
                    if scene.ngeom >= scene.maxgeom:
                        raise ValueError("Decorations overflow")
                    geom = scene.geoms[scene.ngeom]
                    rgba = [int(style["fill"][j : j + 2], 16) / 255 for j in (1, 3, 5)] + [1]
                    mj.mjv_initGeom(
                        geom,
                        mj.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, 0.07),
                        np.asarray(goal),
                        np.eye(3).ravel(),
                        rgba,
                    )
                    geom.label = label + ("+" if label in completed else "*" if label == target_id else "")
                    scene.ngeom += 1
            if name == "main":
                # Conservative projected bounding spheres for every visible body geom.
                eye = np.mean([v.pos for v in r.scene.camera], axis=0)
                forward = np.mean([v.forward for v in r.scene.camera], axis=0)
                forward /= np.linalg.norm(forward)
                up = np.mean([v.up for v in r.scene.camera], axis=0)
                up /= np.linalg.norm(up)
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                scale = c["raw_height"] / (2 * np.tan(np.deg2rad(self.model.vis.global_.fovy) / 2))
                limits = []
                for gi in range(self.model.ngeom):
                    if self.model.geom_bodyid[gi] == 0 or self.model.geom_rgba[gi, 3] == 0:
                        continue
                    delta = self.data.geom_xpos[gi] - eye
                    depth = float(delta @ forward)
                    radius = float(self.model.geom_rbound[gi])
                    if depth <= radius:
                        raise ValueError("Body intersects camera")
                    px = c["raw_width"] / 2 + scale * float(delta @ right) / depth
                    py = c["raw_height"] / 2 - scale * float(delta @ up) / depth
                    margin = scale * radius / (depth - radius)
                    limits.append([px - margin, py - margin, px + margin, py + margin])
                b = np.asarray(limits)
                bounds = [
                    float(b[:, 0].min()),
                    float(b[:, 1].min()),
                    float(b[:, 2].max()),
                    float(b[:, 3].max()),
                ]
                if (
                    bounds[0] < 8
                    or bounds[1] < 8
                    or bounds[2] > c["raw_width"] - 8
                    or bounds[3] > c["raw_height"] - 8
                ):
                    raise ValueError("Drone/suspension/package outside close main frame")
                ix, iy = c["inset_xy"]
                iw, ih = c["overview"]["width"], c["overview"]["height"]
                if not (
                    bounds[2] < ix - 2
                    or bounds[0] > ix + iw + 2
                    or bounds[3] < iy - 2
                    or bounds[1] > iy + ih + 2
                ):
                    raise ValueError("Overview would hide physical bodies")
                self.bounds.append(dict(time_s=raw["time_s"], bounds_px=bounds))
            views[name] = r.render().copy()
        result = views["main"]
        x, y = c["inset_xy"]
        inset = views["overview"]
        h, w = inset.shape[:2]
        result[y - 2 : y + h + 2, x - 2 : x + w + 2] = [12, 26, 42]
        result[y : y + h, x : x + w] = inset
        return result


@lru_cache(maxsize=12)
def _font(path, size):
    from PIL import ImageFont

    return ImageFont.truetype(str(path), size)


def close_hud(raw, state, native, payload, config, font, controller_hash, *, final=False):
    from PIL import Image, ImageDraw
    from .payload_contract import Payload, payload_metrics
    from .payload_solver_mujoco import FREE_MASS_KG
    from .payload_solver_trace import observation_sample

    p = Payload(FREE_MASS_KG, payload["mass_ratio"], payload["side_m"])
    m = payload_metrics(observation_sample(native, p.side_m), state["target"], p)
    w, h = config["raw_width"], config["raw_height"]
    image = Image.new("RGB", (w, h + config["hud_height"]), "#101d2d")
    image.paste(Image.fromarray(raw), (0, 0))
    d = ImageDraw.Draw(image)
    f, small = _font(str(font), 23), _font(str(font), 19)
    title = (
        "最終受入飛行・人手連続1倍速確認待ち"
        if final
        else "開発動画・既存成功rawの画角改善 / 新しい飛行成功には数えない"
    )
    lines = [
        title,
        f"{state['scenario_id']} | 物理時刻 {native['time_s']:.2f}s | 目標 {state['selected'] or '-'} | 完了 {len(state['visited'])}/10 | 完全1倍速・省略なし",
        "C-local-lqr-v1 / 飛行NN未使用 | MuJoCo3.15.0 float64 RK4 s16 | payload-settling-video-v1",
        f"荷物 {p.side_m * 1000:.0f}mm / 質量比 {p.mass_ratio:.3f} | 揺れ {m['swing_angle_deg']:.2f}度 | 全角速度 {m['package_spin_deg_s']:.2f}度/s",
        "静定：位置・傾き等と同時に1秒 / 荷物速度≤0.15m/s・振れ速度≤15度/s・箱全角速度≤30度/s",
        "  ".join(
            f"{chr(65 + i)}{'＋' if chr(65 + i) in state['visited'] else '＊' if chr(65 + i) == state['selected'] else '○'}:{v}"
            for i, v in enumerate(state["priorities"])
        ),
        state.get("event_text")
        or "主画面：機体と吊り荷の中点を固定倍率で追従 / 右上：全10地点 / ○未完了・＊目標・＋完了",
        f"制御設定SHA256 {controller_hash[:24]}… | 飛行rawの物理状態をそのまま再表示",
    ]
    for i, line in enumerate(lines):
        d.text(
            (16, h + 8 + i * 32), line, font=f if i < 2 else small, fill="#ffe59b" if i == 0 else "#e9f3fb"
        )
    result = np.asarray(image).copy()
    if not np.array_equal(result[:h], raw):
        raise ValueError("HUD changed native viewport")
    return result


def verify_caption_binding(trial, prior):
    """Independently match every fixed caption to hash-bound actual flight fields."""
    from .payload_hybrid_contract import VIDEO_PROFILE
    from .util import sha256

    trial = Path(trial)
    for rel, digest in prior["physical_artifacts"].items():
        if sha256(trial / rel) != digest:
            raise ValueError("Prior physical artifact changed: " + rel)
    meta = json.loads((trial / "mission/metadata.json").read_text())
    design = json.loads((trial / "flight/design.json").read_text())
    prop = json.loads((trial / "flight/runtime-properties.json").read_text())
    expected = dict(
        controller="C-local-lqr-v1",
        backend="MuJoCo3.15.0/float64/RK4/substeps16/N1",
        flight_nn="NOT_USED",
        settling_profile=VIDEO_PROFILE.id,
        settling_profile_sha256=VIDEO_PROFILE.sha256,
        aggregate_simulated_deadline_s=None,
        leg_deadline_s=30.0,
        mission_wall_deadline_s=600.0,
    )
    if any(meta.get(k) != v for k, v in expected.items()) or design.get("recipe") != "C-local-lqr-v1":
        raise ValueError("Fixed captions differ from actual controller/backend/profile/deadlines")
    if any(
        prop.get(k) != v
        for k, v in dict(
            backend="mujoco",
            version="3.15.0",
            precision="64",
            substeps=16,
            integrator=1,
            actual_dt_s=0.000625,
        ).items()
    ):
        raise ValueError("Fixed captions differ from actual compiled settings")
    if meta["checkpoint_sha256"] != sha256(trial / "flight/design.json"):
        raise ValueError("Actual controller design differs")
    return dict(
        schema="mujoco-close-caption-binding-v1",
        actual_labels=expected,
        actual_design_sha256=sha256(trial / "flight/design.json"),
        physical_artifacts=prior["physical_artifacts"],
        new_physical_success_count=0,
    )


def verify_display_proof(proof, trial, prior, prior_sha256, config):
    """Additional immutable label/inset/input checks for preserved first capture."""
    if (
        proof["prior_video_proof_sha256"] != prior_sha256
        or proof["independent_mission"] != prior["independent_mission"]
        or proof["physical_artifacts"] != prior["physical_artifacts"]
        or proof["new_physical_success_count"] != 0
    ):
        raise ValueError("Prior content and bound SHA differ")
    labels = verify_caption_binding(trial, prior)
    n = proof["lossless"]["frames"]
    bounds = proof["framing_all_rendered_frames"]
    if len(bounds) != 2 * n:
        raise ValueError("Both complete framing traversals required")
    x, y = config["inset_xy"]
    w, h = config["overview"]["width"], config["overview"]["height"]
    for j, row in enumerate(bounds):
        if abs(row["time_s"] - (j % n) / 100) > 1e-7:
            raise ValueError("Framing time coverage differs")
        left, top, right, bottom = row["bounds_px"]
        if not (right < x - 2 or left > x + w + 2 or bottom < y - 2 or top > y + h + 2):
            raise ValueError("Overview occludes physical bodies")
    return dict(
        schema="mujoco-close-display-supplement-v1",
        caption_binding=labels,
        complete_framing_traversals=2,
        frames_per_traversal=n,
        inset_occluded_frames=0,
        prior_content_hash_consistent=True,
        new_physical_success_count=0,
    )
