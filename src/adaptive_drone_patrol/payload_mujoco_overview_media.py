"""Enlarged original fixed overview; original detail camera stays unchanged."""

import numpy as np
from .payload_mujoco_close_media import verify_caption_binding as verify_caption_binding
from .payload_solver_media import NativeRenderer, load_camera


def load_overview_camera(path="configs/payload_mujoco_overview_camera.json"):
    c = load_camera(path)
    if c["overview"]["width"] != 1280 or c["overview"]["height"] != 720 or c["overview"]["distance"] >= 10:
        raise ValueError("Enlarged original fixed overview required")
    original = load_camera()
    if c["detail"] != original["detail"]:
        raise ValueError("Original individual detail view must remain unchanged")
    return c


def project_spheres(camera, positions, radii, *, width, height, fovy):
    eye = np.mean([v.pos for v in camera], axis=0)
    forward = np.mean([v.forward for v in camera], axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean([v.up for v in camera], axis=0)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    scale = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    result = []
    for p, radius in zip(positions, radii):
        delta = np.asarray(p) - eye
        depth = float(delta @ forward)
        if depth <= radius:
            raise ValueError("Physical geometry intersects camera")
        x = width / 2 + scale * float(delta @ right) / depth
        y = height / 2 - scale * float(delta @ up) / depth
        margin = scale * radius / (depth - radius)
        result.append([x - margin, y - margin, x + margin, y + margin])
    a = np.asarray(result)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())]


class FixedOverviewRenderer(NativeRenderer):
    def __init__(self, model_path, config):
        super().__init__(model_path, config, cpu_only=True)
        self.bounds = []

    def render(self, raw, goals, priorities, completed, target_id):
        frame = super().render(raw, goals, priorities, completed, target_id)
        ids = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] != 0 and self.model.geom_rgba[i, 3] != 0
        ]
        cam = self.renderers["overview"].scene.camera
        kw = dict(
            width=self.config["overview"]["width"],
            height=self.config["overview"]["height"],
            fovy=self.model.vis.global_.fovy,
        )
        bodies = project_spheres(cam, self.data.geom_xpos[ids], self.model.geom_rbound[ids], **kw)
        targets = project_spheres(cam, goals, np.full(len(goals), self.config["marker_radius_m"]), **kw)
        for name, b, margin in [("physical bodies", bodies, 12), ("all ten destinations", targets, 40)]:
            if b[0] < margin or b[1] < margin or b[2] > kw["width"] - margin or b[3] > kw["height"] - margin:
                raise ValueError(name + " outside fixed overview: " + str(b))
        self.bounds.append(dict(time_s=raw["time_s"], bodies_px=bodies, all_ten_targets_px=targets))
        return frame


def verify_framing(bounds, *, frames, config):
    if type(frames) is not int or frames < 1 or len(bounds) != 2 * frames:
        raise ValueError("Two complete native framing traversals required")
    first = None
    for i, row in enumerate(bounds):
        if abs(row["time_s"] - (i % frames) / 100) > 1e-7:
            raise ValueError("Framing time coverage differs")
        for key, margin in [("bodies_px", 12), ("all_ten_targets_px", 40)]:
            b = np.asarray(row[key], float)
            if (
                b.shape != (4,)
                or not np.isfinite(b).all()
                or not (margin <= b[0] < b[2] <= 1280 - margin and margin <= b[1] < b[3] <= 720 - margin)
            ):
                raise ValueError("Fixed overview clips " + key)
        if first is None:
            first = row["all_ten_targets_px"]
        if not np.allclose(row["all_ten_targets_px"], first, rtol=0, atol=1e-8):
            raise ValueError("Overview camera or goals moved")
    return dict(
        complete_traversals=2,
        frames_per_traversal=frames,
        all_ten_destinations_visible=True,
        whole_drone_suspension_package_visible=True,
        overview_fixed=True,
        detail_camera_unchanged=True,
    )


def final_hud(raw, state, native, case, config, font, controller_hash):
    from PIL import Image, ImageDraw, ImageFont
    from .payload_solver_media import compose_hud

    image = Image.fromarray(compose_hud(raw, state, native, case, config, font, controller_hash))
    d = ImageDraw.Draw(image)
    h, w = config["raw_height"], config["raw_width"]
    d.rectangle((0, h + 390, w, h + config["hud_height"]), fill="#101d2d")
    d.text(
        (22, h + 397),
        "機械判定：最終受入済 / 人手連続1倍速レビュー待ち",
        font=ImageFont.truetype(str(font), 23),
        fill="#ffe59b",
    )
    d.text(
        (22, h + 432),
        "C-local-lqr-v1 / 飛行NN未使用 | MuJoCo3.15.0 float64 RK4 s16 | payload-settling-video-v1 | 完全1倍速",
        font=ImageFont.truetype(str(font), 20),
        fill="#c2dced",
    )
    result = np.asarray(image).copy()
    if not np.array_equal(result[:h], raw):
        raise ValueError("Final HUD changed native pixels")
    return result


def caption_regions(config):
    from .payload_solver_video import caption_regions as original

    if (config["raw_width"], config["raw_height"], config["hud_height"]) != (1792, 720, 480):
        raise ValueError("Original Japanese HUD dimensions required")
    return original()



