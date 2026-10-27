"""Native frame scheduling and camera-only replay of retained real qpos.

The renderer never steps physics. Display colors/markers/cameras are separate
from the qualified physical model; no visual-only edit grants physical credit.
"""

import copy
import ctypes
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=12)
def _hud_font(path,size):
    from PIL import ImageFont
    return ImageFont.truetype(path,size)


def native_frame_rows(records):
    """Emit each control instant exactly once, including the final real state."""
    previous, step = None, 0
    for row in records:
        if row["episode_id"] != 1:
            raise ValueError("A complete video cannot contain hidden episode resets")
        kind = row["kind"]
        if previous is None:
            if kind != "RESET" or row["step"] != 0 or row["actual"]["time_s"] != 0:
                raise ValueError("Initial native frame required")
        elif kind == "RETARGET":
            if row["step"] != step or row["actual"] != previous:
                raise ValueError("Retarget changed actual camera pose/time")
            continue
        elif kind == "STEP":
            step += 1
            if row["step"] != step or not np.isclose(row["actual"]["time_s"], step * 0.01, atol=1e-7, rtol=0):
                raise ValueError("Missing native video interval or nonphysical clock")
        else:
            raise ValueError("Unexpected reset or native video record")
        previous = copy.deepcopy(row["actual"])
        yield dict(step=step, actual=previous)
    if previous is None:
        raise ValueError("Empty native video cannot establish1x")


def select_software_egl_device():
    """Select only an explicitly software EGL device before context creation."""
    required = dict(
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LIBGL_ALWAYS_SOFTWARE="true", GALLIUM_DRIVER="llvmpipe"
    )
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError("Explicit CPU rendering environment required before imports")
    from mujoco.egl import egl_ext as EGL

    query = ctypes.CFUNCTYPE(ctypes.c_char_p, EGL.EGLDeviceEXT, EGL.EGLint)(
        EGL.eglGetProcAddress("eglQueryDeviceStringEXT")
    )
    devices = []
    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        extensions = query(device, 0x3055).decode()
        devices.append(dict(index=index, extensions=extensions))
        if "EGL_MESA_device_software" in extensions:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(index)
            return dict(selected_device=index, device_inventory=devices)
    raise ValueError("No explicit software renderer; no GPU context was created")


class NativeRenderer:
    """Two real native camera views from the exact recorded joint state."""

    def __init__(self, model_path, config, *, cpu_only=False):
        self.cpu_only = cpu_only
        self.device = select_software_egl_device() if cpu_only else None
        import mujoco as mj
        from OpenGL import GL

        self.mj = mj
        self.config = copy.deepcopy(config)
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.model.vis.global_.offwidth = config["overview"]["width"]
        self.model.vis.global_.offheight = config["overview"]["height"]
        self.model.vis.quality.offsamples = 4
        self.model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
        self.model.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
        # These colors affect this display-only model, never the source physics.
        for name, color in [("ground", [0.28, 0.35, 0.44, 1.0]), ("vehicle_collision", [0.0, 0.0, 0.0, 0.0])]:
            index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            if index >= 0:
                self.model.geom_rgba[index] = color
        for index in range(self.model.ngeom):
            if self.model.geom_type[index] == mj.mjtGeom.mjGEOM_MESH:
                self.model.geom_rgba[index] = [0.8, 0.88, 0.98, 1.0]
        self.renderers = {
            name: mj.Renderer(self.model, height=config[name]["height"], width=config[name]["width"])
            for name in ["overview", "detail"]
        }
        self.renderer_string = GL.glGetString(GL.GL_RENDERER).decode()
        self.vendor = GL.glGetString(GL.GL_VENDOR).decode()
        if cpu_only and "llvmpipe" not in self.renderer_string.lower():
            self.close()
            raise ValueError("CPU-only preview received a hardware context")
        self.closed = False

    def goal_label(self, index, goal, priority, style):
        return f"{style['symbol']} {chr(65 + index)} {priority}"

    def close(self):
        for renderer in getattr(self, "renderers", {}).values():
            renderer.close()
        self.closed = True

    def render(self, raw, goals, priorities, completed, target_id):
        mj = self.mj
        for key, width in [("qpos", self.model.nq), ("qvel", self.model.nv)]:
            values = np.asarray(raw[key], dtype=float)
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError("Actual finite native joint state required for every frame")
            getattr(self.data, key)[:] = values
        self.data.time = raw["time_s"]
        mj.set_mjcb_control(None)
        mj.mj_forward(self.model, self.data)
        if not np.array_equal(self.data.qpos, raw["qpos"]) or not np.array_equal(self.data.qvel, raw["qvel"]):
            raise ValueError("Display forward pass changed recorded native state")
        for body, name in [("drone", "base_link"), ("package", "payload")]:
            index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if not np.allclose(self.data.xpos[index], raw[body + "_xyz"], rtol=0, atol=1e-8):
                raise ValueError("Rendered body differs from actual native getter")
        views = []
        for name in ["overview", "detail"]:
            camera = mj.MjvCamera()
            mj.mjv_defaultCamera(camera)
            setting = self.config[name]
            camera.lookat[:] = (
                setting["lookat"]
                if name == "overview"
                else np.array(raw["drone_xyz"]) + setting["lookat_offset"]
            )
            for field in ["distance", "azimuth", "elevation"]:
                setattr(camera, field, setting[field])
            renderer = self.renderers[name]
            renderer.update_scene(self.data, camera=camera)
            for index, (goal, priority) in enumerate(zip(goals, priorities) if name == "overview" else []):
                label = chr(ord("A") + index)
                style = goal_display_style(priority, visited=label in completed, selected=label == target_id)
                color = [int(style['fill'][i:i+2],16)/255 for i in (1,3,5)] + [0.9]
                scene = renderer.scene
                if scene.ngeom >= scene.maxgeom:
                    raise ValueError("Camera decorative geometry capacity exceeded")
                geom = scene.geoms[scene.ngeom]
                mj.mjv_initGeom(
                    geom,
                    dict(pending=mj.mjtGeom.mjGEOM_SPHERE, selected=mj.mjtGeom.mjGEOM_CYLINDER,
                         visited=mj.mjtGeom.mjGEOM_BOX)[style['state']],
                    np.full(3, self.config["marker_radius_m"]),
                    np.asarray(goal),
                    np.eye(3).ravel(),
                    color,
                )
                geom.label = self.goal_label(index, goal, priority, style)
                scene.ngeom += 1
            views.append(renderer.render().copy())
        result = np.concatenate(views, axis=1)
        if result.shape != (self.config["raw_height"], self.config["raw_width"], 3):
            raise ValueError("Frozen native viewport dimensions differ")
        return result


def load_camera(path="configs/payload_solver_camera.json"):
    config = json.loads(Path(path).read_text())
    if (
        config["schema"] != "payload-solver-camera-v1"
        or config["fps"] != 100
        or config["raw_width"] != config["overview"]["width"] + config["detail"]["width"]
        or config["raw_height"] != config["overview"]["height"]
        or config["raw_height"] != config["detail"]["height"]
    ):
        raise ValueError("Frozen100FPS native camera contract differs")
    return config


def goal_display_style(priority, *, visited=False, selected=False):
    """Keep inherited1..100 colors separate from independent visit symbols."""
    from .scene import priority_color

    state = 'visited' if visited else 'selected' if selected else 'pending'
    return dict(fill=priority_color(priority,100), state=state,
                symbol=dict(visited='+',selected='*',pending='o')[state])


def compose_hud(raw, state, native, payload, config, font_path, model_sha256):
    """Append legible Japanese telemetry without writing into either raw view."""
    from PIL import Image, ImageDraw
    from .payload_contract import Payload, payload_metrics
    from .payload_solver_mujoco import FREE_MASS_KG
    from .payload_solver_trace import observation_sample

    raw = np.asarray(raw)
    h, w = config["raw_height"], config["raw_width"]
    if raw.shape != (h, w, 3) or raw.dtype != np.uint8:
        raise ValueError("Original full RGB viewport required")
    cube = Payload(FREE_MASS_KG, payload["mass_ratio"], payload["side_m"])
    metrics = payload_metrics(observation_sample(native, cube.side_m), state["target"], cube)
    canvas = Image.new("RGB", (w, h + config["hud_height"]), "#101d2d")
    canvas.paste(Image.fromarray(raw), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title = _hud_font(str(font_path), 28)
    face = _hud_font(str(font_path), 23)
    small = _hud_font(str(font_path), 20)
    names = dict(normal="通常巡回", emergency="緊急優先度変更", cost_shift="有向コスト変更")

    def text(x, y, value, font=face, fill="#e9f3fb"):
        draw.text((x, h + y), value, font=font, fill=fill)

    text(22, 10, f"全体視点  |  {names[state['scenario_id']]}  |  実測1倍速", title)
    text(config["overview"]["width"] + 15, 10, "同時刻の機体・吊り荷拡大", face)
    text(
        22,
        55,
        f"物理時刻 {native['time_s']:.2f} 秒   目標 {state['selected'] or '-'}   完了 {len(state['visited'])}/10",
    )
    text(850, 55, f"高度：機体 {native['drone_xyz'][2]:.3f} m / 荷物 {native['package_xyz'][2]:.3f} m")
    text(
        22,
        91,
        f"荷物 {cube.side_m * 1000:.0f} mm角 / 質量比 {cube.mass_ratio:.3f} / {cube.mass * 1000:.3f} g",
        small,
    )
    text(
        850,
        91,
        f"揺れ {metrics['swing_angle_deg']:.2f} 度 / 荷物spin {metrics['package_spin_deg_s']:.2f} 度/秒",
        small,
    )
    for index, priority in enumerate(state["priorities"]):
        label = chr(ord("A") + index)
        visited = label in state["visited"]
        selected = label == state["selected"] and not visited
        x, y = 22 + (index % 5) * 350, h + 140 + (index // 5) * 102
        style = goal_display_style(priority, visited=visited, selected=selected)
        outline = "#59d9a0" if visited else "#ffc557" if selected else "#7c9caf"
        draw.rounded_rectangle((x, y, x + 330, y + 84), radius=9, fill="#20374d", outline=outline, width=3)
        draw.rectangle((x+295,y+12,x+316,y+70),fill=style['fill'])
        draw.text((x + 15, y + 8), f"{label}   優先度 {priority}", font=face, fill="white")
        draw.text(
            (x + 15, y + 43),
            "＋ 訪問完了" if visited else "＊ 現在の目標" if selected else "○ 未完了",
            font=small,
            fill="#d9f7ed" if visited else "#fff1ce" if selected else "#b9d0df",
        )
    event = state.get("event_text") or "全地点は固定配置。機体と荷物の全条件を同時に1秒満たして完了。"
    text(22, 354, event, face, fill="#ffe59b")
    text(22, 397, "優先度の色：1（淡）→100（濃） / ○未完了・＊目標・＋完了 / 0.30m剛体リンク", small)
    text(22, 432, f"方策SHA-256 {model_sha256[:24]}…  |  最終frameまで省略なし", small, fill="#aac4d7")
    result = np.asarray(canvas).copy()
    if not np.array_equal(result[:h], raw):
        raise ValueError("Japanese HUD changed an original raw pixel")
    return result


def mission_display_state(scenario, events, step):
    """Fold only same-or-earlier real mission events into the visible state."""
    state = dict(
        scenario_id=scenario.scenario_id,
        selected=None,
        visited=[],
        priorities=scenario.priorities.copy(),
        target=scenario.coords[0],
        event_text=None,
    )
    for event in events:
        if event["simulation_step"] > step:
            break
        kind = event["type"]
        if kind == "TARGET_SELECTED":
            label = event["target_id"]
            state["selected"] = label
            state["target"] = scenario.coords[scenario.checkpoint_ids.index(label) + 1]
        elif kind == "SHARED_SETTLED":
            if event["target_id"] in state["visited"]:
                raise ValueError("Repeated physical completion cannot enter HUD")
            state["visited"].append(event["target_id"])
        elif kind == "PRIORITY_UPDATED":
            update = event["update"]
            label = update["checkpoint_id"]
            state["priorities"][scenario.checkpoint_ids.index(label)] = update["value"]
            state["event_text"] = (
                f"緊急：{label}の優先度 {update['old']} → {update['value']}（最初の実訪問完了後）"
            )
        elif kind == "COST_UPDATED":
            update = event["update"]
            edges = " / ".join(f"{a}→{b}" for a, b in update["edges"])
            state["event_text"] = (
                f"有向移動コスト {edges}：{update['old'][0]:g} → {update['value']:g}（他の辺は維持）"
            )
    return state
