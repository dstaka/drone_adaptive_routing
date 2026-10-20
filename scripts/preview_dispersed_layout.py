#!/usr/bin/env python
"""Render measured geometry only; this is not a physical flight recording."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from adaptive_drone_patrol.analytical import demo_fixture
from adaptive_drone_patrol.scenario import INTEGER_PROFILE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--font", default="outputs/visible_20261015/remote/NotoSansJP.ttf")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    data = json.loads(Path(a.layout).read_text())
    old = np.array(demo_fixture(40029, INTEGER_PROFILE).coords)
    new = np.array(data["coords"])
    font = FontProperties(fname=a.font)
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11})
    from matplotlib.font_manager import fontManager

    fontManager.addfont(a.font)
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    for col, points, title in [
        (0, old, "旧配置：全地点の絶対高度 1.00 m"),
        (1, new, "新候補：乱数による実3D配置（人手確認前）"),
    ]:
        ax = fig.add_subplot(2, 2, col + 1)
        ax.set_title(title)
        ax.scatter(
            points[1:, 0],
            points[1:, 1],
            c=points[1:, 2],
            cmap="viridis",
            vmin=1,
            vmax=3,
            s=95,
            edgecolors="black",
        )
        for i, xyz in enumerate(points[1:]):
            label = chr(65 + i) if col == 0 else f"{chr(65 + i)}  z={xyz[2]:.2f}m"
            offset = (-85, 9) if xyz[0] > 2 else (5, 9)
            ax.annotate(label, xyz[:2], xytext=offset, textcoords="offset points", fontsize=11, color="black")
        ax.scatter([0], [0], marker="*", s=160, c="red")
        ax.annotate("S (z=1.00m)", (0, 0), xytext=(5, -18), textcoords="offset points")
        ax.set(xlim=(-3, 3), ylim=(-2.1, 2.1), xlabel="x [m]", ylabel="y [m]（上から見た水平分散）")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(2, 2, col + 3, projection="3d")
        ax.scatter(
            points[1:, 0], points[1:, 1], points[1:, 2], c=points[1:, 2], cmap="viridis", vmin=1, vmax=3, s=60
        )
        for i, (x, y, z) in enumerate(points[1:]):
            ax.plot([x, x], [y, y], [0, z], c="gray", alpha=0.45, linestyle="--")
            ax.text(x, y, z + 0.09, f"{chr(65 + i)} {z:.2f}m", fontsize=10)
        ax.scatter([0], [0], [1], marker="*", s=100, c="red")
        ax.set(
            xlim=(-2.9, 2.9),
            ylim=(-2, 2),
            zlim=(0, 3.2),
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="地面からの絶対高度 z [m]",
        )
        ax.view_init(elev=25, azim=-55)
        ax.set_box_aspect((5.4, 3.6, 3.2))
    c = data["checks"]
    fig.suptitle(
        "A〜Jの実目的地配置：水平分散・高低差の新旧比較\n"
        f"新候補の実span：x={c['spans_m'][0]:.3f} m / y={c['spans_m'][1]:.3f} m / z={c['spans_m'][2]:.3f} m",
        fontsize=18,
    )
    fig.savefig(out, dpi=130)
    plt.close(fig)
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    ax.axis("off")
    rows = [
        [chr(65 + i), f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", str(data["priorities"][i])]
        for i, (x, y, z) in enumerate(new[1:])
    ]
    table = ax.table(
        cellText=rows, colLabels=["ID", "x [m]", "y [m]", "z [m]", "重要度"], loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax.set_title("採用候補の座標と固定重要度（表示は丸め、実値はmanifest）")
    ax = fig.add_subplot(1, 2, 2)
    ax.axis("off")
    c = data["checks"]
    lines = [
        "配置プレビュー／飛行実績ではありません",
        f"配置seed {data['seed']} / 候補 {data['selected_attempt']}",
        f"重要度seed {data['priority_seed']}（旧配置と同じ）",
        "",
        "実際の広がり x / y / z",
        " / ".join(f"{x:.3f}" for x in c["spans_m"]) + " m",
        f"実高度 {c['height_min_max_m'][0]:.3f}〜{c['height_min_max_m'][1]:.3f} m",
        f"高さ3帯の点数 {c['height_band_counts']}",
        f"最小3D間隔 {c['min_separation_m']:.3f} m",
        f"Sからの最小距離 {c['min_base_distance_m']:.3f} m",
        f"実水平凸包面積 {c['convex_hull_xy_m2']:.3f} m²",
        "公称水平領域：5.4×3.6 m（旧領域の9倍）",
        "9倍は凸包面積や飛行距離の比ではありません。",
        "",
        "3シナリオ・全実行seedで同一座標を使用。",
        "数値条件：合格／配置の人手確認：未確認",
        "受入試行：未凍結・未実行",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=12, linespacing=1.6)
    fig.savefig(out.with_name(out.stem + "-coordinates.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
