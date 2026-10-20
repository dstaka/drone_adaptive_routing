"""Japanese viewer layout; physical timing is owned by the delivery scheduler."""

import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .scene import project, priority_color, marker_style, priority_text
from .annotations import same_frame_detail, dashed_arrow
from .video_delivery import optimality_captions

TITLES = {
    "normal": "シナリオA：通常巡回",
    "emergency": "シナリオB：緊急点検",
    "cost_shift": "シナリオC：移動コスト変化",
}
SUBTITLES = {
    "normal": "距離と優先度のバランス",
    "emergency": "地点Cの優先度上昇で残りの巡回を再計画",
    "cost_shift": "有向移動コストの増加で低コストな順序に変更",
}


def current_target_distance(scenario, pose, target):
    if target is None:
        return None
    position = scenario["coords"][scenario["checkpoint_ids"].index(target) + 1]
    return float(np.linalg.norm(np.asarray(position) - pose))


def state_at(record, frame):
    time = float(frame["sim_time_s"])
    state = {
        "selected": None,
        "visited": [],
        "priorities": list(record["scenario"]["priorities"]),
        "event_seq": 0,
        "update": None,
    }
    for event in record["events"]:
        if event["sim_time_s"] > time + 1e-9:
            break
        state["event_seq"] = event["event_seq"]
        kind = event["type"]
        if kind == "TARGET_SELECTED":
            state["selected"] = event["target_id"]
        if kind == "WAYPOINT_REACHED":
            state["visited"].append(event["target_id"])
            state["selected"] = None
        if kind == "PRIORITY_UPDATED":
            state["priorities"][event["update"]["checkpoint"]] = event["update"]["value"]
            state["update"] = event
        if kind == "COST_UPDATED":
            state["update"] = event
    return state


def provenance_values(mapping):
    kind = mapping["run_id"].rsplit("_seed_", 1)[0]
    code = ["normal", "emergency", "cost_shift"].index(kind) + 1
    state = {k: mapping[k] for k in ["visited", "target", "priorities"]}
    digest = int.from_bytes(hashlib.sha256(json.dumps(state, sort_keys=True).encode()).digest()[:2], "big")
    return [code, mapping["source_frame_index"], mapping["simulation_step"], mapping["event_seq"], digest]


def provenance_stamp(image, mapping):
    draw = ImageDraw.Draw(image)
    bits = "".join(f"{n:016b}" for n in provenance_values(mapping))
    draw.rectangle((0, 1060, 1919, 1079), fill="#102333")
    for i, bit in enumerate(bits):
        draw.rectangle((20 + i * 6, 1064, 24 + i * 6, 1075), fill="white" if bit == "1" else "black")


def order_text(order, ids):
    return " → ".join(ids[i] for i in order)


def card_specs(records, values):
    feature = next(
        v for v in values if v["run_id"] == records[-1]["path"].name and v["nearer_lower_priority_gate"]
    )
    ids = feature["snapshot"]["scenario"]["checkpoint_ids"]
    low, high = feature["selected_L"], feature["selected_H"]
    candidates = {x["id"]: x for x in feature["candidate_priorities_distances"]}
    rows = []
    for name in [low, high]:
        c = candidates[name]
        q = next(q for q in feature["forced_first"] if q["action"] == c["action"])
        rows.append(
            {
                "label": f"{'近いL' if name == low else '遠い高優先度H'}（地点{name}）を先に訪問した場合の最小コスト",
                "name": name,
                "candidate": c,
                "result": q,
            }
        )
    rows.append({"label": "AIの残り巡回予測（AIの予測）", "result": feature["policy_rollout"]})
    trade = {
        "title": "判断時点の比較（静止図）",
        "kind": "tradeoff",
        "run_id": feature["run_id"],
        "decision_id": feature["decision_id"],
        "boundary_hash": feature["boundary_hash"],
        "timestamp_s": next(r["schedule"]["dt"] for r in records if r["path"].name == feature["run_id"])
        * feature["physical_reference"]["simulation_step"],
        "feature": feature,
        "rows": rows,
    }
    cards = {
        "intro": {
            "title": "適応型ドローン巡回点検",
            "lines": [
                "A〜Jの全10地点を、同じ学習済みAIで実飛行・点検します。",
                "優先度は重要度であり、訪問順そのものではありません",
                "全地点：固定色と数値 ／ 選択：水色の輪 ／ 点検済み：✓",
                "実飛行はシミュレーション時間に対して1.0倍速。全編・提出用とも省略なし。",
                "説明は独立した静止カード。破線の比較順序は予測で、実飛行ではありません。",
            ],
        },
        "tradeoff": trade,
        "outro": {
            "title": "実測結果・限界・出典",
            "lines": [
                "固定9試行すべて成功。実点検90/90地点。動画は同じ実行seed 1の3シナリオ。",
                "同じ上位MaskablePPO／追加学習後の下位PPOを固定して使用。",
                "目的コストは論理モデル。物理時間・電力の削減を意味しません。",
                "開発失敗を保持。障害物・知覚・実機安全性・幅広い信頼性は未検証。",
                "Genesis v1.4.3（Apache-2.0）／Noto Sans JP（SIL OFL 1.1）",
            ],
        },
    }
    for record in records:
        kind = record["scenario"]["scenario_id"]
        summary = record["summary"]
        cards[kind + "_end"] = {
            "title": TITLES[kind] + "：完了",
            "lines": [
                f"実点検 {summary['completed_count']}/{summary['num_checkpoints']} 地点 ／省略なし",
                "実訪問順：" + " → ".join(summary["completed_order"]),
                f"物理時間 {record['schedule']['end']:.2f} 秒 ／実軌跡 {summary['measured_distance_m']:.3f} m",
                f"名目目的コスト {summary['analytical_cost']:.3f}（物理時間や電力とは異なる）",
            ],
        }
        if kind == "normal":
            continue
        event = next(e for e in record["events"] if e["type"] in ["PRIORITY_UPDATED", "COST_UPDATED"])
        comparison = event["comparison"]
        c = event["update"]
        update = (
            f"地点C：優先度 {priority_text(c['old'], record['scenario'].get('schema_version', 2), 3)} → {c['value']:g}"
            if kind == "emergency"
            else "有向辺 "
            + ", ".join(f"{a}→{b}" for a, b in c["edges"])
            + f"：倍率 {c['old'][0]:g} → {c['value']:g}"
        )
        cards[kind] = {
            "title": TITLES[kind] + "：判断時点の比較（静止図）",
            "kind": "event",
            "run_id": record["path"].name,
            "event_seq": event["event_seq"],
            "timestamp_s": event["sim_time_s"],
            "comparison": comparison,
            "lines": [
                f"物理時刻 {event['sim_time_s']:.2f} 秒、最初の点検完了直後：" + update,
                "更新なしの予測：" + order_text(comparison["no_event_order"], ids),
                "更新後の予測：" + order_text(comparison["post_event_order"], ids),
                f"更新後条件で再評価：旧順序 {comparison['retained_order_post_event']['cost']:.3f} → 新順序 {comparison['replanned_order_post_event']['cost']:.3f}",
                f"名目目的コストの改善 {comparison['improvement']:.3f}。物理時間・エネルギーの削減ではありません。",
                "同じ境界snapshot・同じ学習済み方策の比較。実飛行は次の映像で確認。",
            ],
        }
    for card in cards.values():
        card["scenario_schema_version"] = records[0]["scenario"].get("schema_version", 2)
    if records[0]["scenario"].get("schema_version") == 3:
        cards["intro"]["lines"][1] = "重要度は1〜100の整数。大きいほど重要で、訪問順そのものではありません"
    return cards


class Renderer:
    def __init__(self, font_path):
        self.fonts = {}
        for size in [20, 23, 25, 28, 30, 34, 40, 46]:
            face = ImageFont.truetype(str(font_path), size)
            try:
                face.set_variation_by_axes([500])
            except (OSError, AttributeError):
                pass
            self.fonts[size] = face

    def text(self, draw, xy, value, size=28, fill="white", stroke=0):
        draw.text(
            xy, str(value), font=self.fonts[size], fill=fill, stroke_width=stroke, stroke_fill="#102333"
        )

    def dispersed_badges(self, draw, record, state):
        """Keep ID, integer weight and actual height together in full-size badges."""
        schema=record['scenario'].get('schema_version',2)
        maximum=100 if schema==3 else 5
        for j, (marker, rect) in enumerate(zip(record["scene"]["markers"], record["scene"]["badge_rects"])):
            x, y = project(marker["world_xyz"], record["scene"]["camera"])
            a, b, c, d = [int(v) for v in rect]
            color = priority_color(state["priorities"][j], maximum)
            ink = marker_style(state["priorities"][j], priority_max=maximum)["ink"]
            draw.line([(x + 20, y + 145), ((a + c) / 2 + 20, (b + d) / 2 + 145)], fill="#d5e4ec", width=2)
            draw.rectangle((a + 20, b + 145, c + 20, d + 145), fill=color, outline="white", width=2)
            self.text(draw, (a + 28, b + 148), f"{marker['id']}  {priority_text(state['priorities'][j],schema)}", 23, fill=ink)
            self.text(draw, (a + 28, b + 175), f"z={marker['world_xyz'][2]:.3f}m", 20, fill=ink)
            status = (
                "AIの選択"
                if marker["id"] == state["selected"]
                else "✓ 点検済み"
                if marker["id"] in state["visited"]
                else "未点検"
            )
            self.text(draw, (a + 28, b + 201), status, 20, fill=ink)

    def legend(self, draw, y=995, schema_version=2):
        self.text(
            draw,
            (25, y),
            "重要度：1（低）〜100（高）" if schema_version == 3 else "優先度：1（低）〜5（高）",
            25,
        )
        for index, i in enumerate([1, 25, 50, 75, 100] if schema_version == 3 else range(1, 6)):
            x = 395 + index * 105
            draw.rectangle((x, y, x + 46, y + 32), fill=priority_color(i, 100 if schema_version == 3 else 5))
            self.text(draw, (x + 55, y), i, 25)
        self.text(draw, (925, y), "水色の輪＝AIの選択 ／ ✓＝点検済み ／ 線＝実飛行", 25)

    def card(self, spec):
        image = Image.new("RGB", (1920, 1080), "#102333")
        draw = ImageDraw.Draw(image)
        self.text(draw, (65, 45), spec["title"], 46)
        self.text(draw, (65, 110), "説明用の静止カード／シミュレーション時計は進みません", 25, fill="#86dbe6")
        if spec.get("kind") == "tradeoff":
            self.tradeoff(draw, spec)
        else:
            for i, line in enumerate(spec.get("lines", [])):
                self.text(draw, (70, 220 + i * 98), line, 30)
        self.legend(draw, schema_version=spec.get("scenario_schema_version", 2))
        return image

    def tradeoff(self, draw, spec):
        value = spec["feature"]
        ids = value["snapshot"]["scenario"]["checkpoint_ids"]
        rows = spec["rows"]
        self.text(
            draw,
            (65, 164),
            f"{spec['run_id']} / {spec['decision_id']} / 物理時刻 {spec['timestamp_s']:.2f}秒",
            25,
        )
        self.text(draw, (65, 207), "現在の条件を固定した残り巡回（これからの更新は仮定しない）", 30)
        for i, row in enumerate(rows):
            y = 266 + i * 165
            r = row["result"]
            self.text(draw, (65, y), row["label"], 28, fill="#86dbe6")
            if i < 2:
                c = row["candidate"]
                self.text(
                    draw,
                    (65, y + 39),
                    f"優先度 {priority_text(c['priority'], value['snapshot']['scenario'].get('schema_version', 2), 3)} ／計画距離 {c['distance_m']:.3f} m ／実位置から {c['pose_distance_m']:.3f} m",
                    25,
                )
            self.text(
                draw,
                (65, y + 78),
                (f"総コスト {r['cost']:.3f} ＝ 移動 {r['travel_cost']:.3f} ＋ 0.05 × 重要度時刻 {r['completion_penalty']:.3f}（{.05 * r['completion_penalty']:.3f}）"
                 if value['snapshot']['scenario'].get('schema_version') == 3 else
                 f"総コスト {r['cost']:.3f} ＝ 移動 {value['snapshot']['scenario']['alpha'] * r['travel_cost']:.3f} ＋ 優先度・完了時刻 {value['snapshot']['scenario']['beta'] * r['completion_penalty']:.3f}"),
                28,
            )
            label = "条件付き最適解（比較用）" if i < 2 else "AIの予測"
            self.text(draw, (65, y + 119), label + "：" + order_text(r["order"], ids), 25)
        for i, row in enumerate(rows[:2]):
            y0 = 300 + i * 230
            self.text(draw, (1450, y0 - 38), "条件付き最適解（比較用）", 20, fill="#86dbe6")
            coords = np.asarray(value["snapshot"]["scenario"]["coords"])
            low = coords[:, :2].min(axis=0) - 0.2
            high = coords[:, :2].max(axis=0) + 0.2

            def xy(p):
                t = (np.asarray(p)[:2] - low) / (high - low)
                return 1460 + t[0] * 370, y0 + 160 - t[1] * 155

            route = [value["snapshot"]["current"]] + [a + 1 for a in row["result"]["order"]]
            for a, b in zip(route, route[1:]):
                dashed_arrow(draw, xy(coords[a]), xy(coords[b]), "#729da8")
            for j, name in enumerate(ids):
                x, y = xy(coords[j + 1])
                color = priority_color(value["snapshot"]["scenario"]["priorities"][j], 100 if value["snapshot"]["scenario"].get("schema_version") == 3 else 5)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white")
                if name == value["selected_L"]:
                    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline="#00e5ff", width=2)
                self.text(draw, (x - 6, y + 9), name, 20)
        captions = optimality_captions(value)
        delta = rows[1]["result"]["cost"] - rows[0]["result"]["cost"]
        self.text(
            draw,
            (65, 773),
            f"Q*(H) − Q*(L) = {delta:.3f} ／ J* = {value['exact']['cost']:.3f}",
            28,
            fill="#ffdc91",
        )
        self.text(draw, (65, 816), " ／ ".join(captions), 28, fill="#ffdc91")
        self.text(
            draw,
            (65, 861),
            "高優先度でも、近い地点を先に回る方が残り全地点の総コストを小さくできる場合があります",
            28,
        )
        s = value["snapshot"]["scenario"]
        self.text(
            draw,
            (65, 908),
            f"総コスト＝移動コスト＋優先度を重みとした完了時刻のペナルティ（小さいほど良い）α={s['alpha']:g} β={s['beta']:g}",
            25,
        )
        self.text(
            draw,
            (65, 949),
            "入力と出力のコスト分析であり、AIの内部思考や物理エネルギーを示すものではありません。",
            23,
        )

    def flight(self, camera, record, frame):
        image = Image.new("RGB", (1920, 1080), "#102333")
        draw = ImageDraw.Draw(image)
        state = state_at(record, frame)
        kind = record["scenario"]["scenario_id"]
        time = float(frame["sim_time_s"])
        self.text(draw, (25, 16), TITLES[kind], 40)
        self.text(draw, (25, 77), SUBTITLES[kind], 30, fill="#b4d9e5")
        image.paste(camera, (20, 145))
        ids = record["scenario"]["checkpoint_ids"]
        n = len(ids)
        telemetry = record["telemetry"]
        step = int(frame["simulation_step"])
        sample = telemetry[step - 1] if step else None
        pose = [float(sample[k]) for k in ["x", "y", "z"]] if sample else record["initial"]
        # The image stays at native resolution: no stretching or hidden crop.
        for j, marker in enumerate(record["scene"]["markers"]):
            x, y = project(marker["world_xyz"], record["scene"]["camera"])
            x += 20
            y += 145
            color = priority_color(state["priorities"][j], 100 if record["scenario"].get("schema_version") == 3 else 5)
            ink = marker_style(state["priorities"][j], priority_max=100 if record["scenario"].get("schema_version") == 3 else 5)["ink"]
            draw.ellipse((x - 17, y - 17, x + 17, y + 17), outline=color, width=6)
            if marker["id"] == state["selected"]:
                draw.ellipse((x - 25, y - 25, x + 25, y + 25), outline="#00e5ff", width=4)
                if not record['scene'].get('badge_rects'):
                    self.text(draw, (x - 53, y - 63), "AIの選択", 20, fill="#00e5ff", stroke=1)
            if record['scene'].get('badge_rects'):
                continue
            label = f"{marker['id']}  {priority_text(state['priorities'][j], record['scenario'].get('schema_version', 2))}"
            w = draw.textlength(label, font=self.fonts[23]) + 18
            draw.rectangle((x - w / 2, y + 23, x + w / 2, y + 56), fill=color, outline="white", width=1)
            self.text(draw, (x - w / 2 + 9, y + 22), label, 23, fill=ink)
            if marker["id"] in state["visited"]:
                self.text(draw, (x + 21, y - 27), "✓", 30, stroke=1)
        feature = next(
            (
                v
                for v in record.get("_explanations", [])
                if v["nearer_lower_priority_gate"]
                and v["physical_reference"]["simulation_step"] <= step
                and ids[v["policy_action"]] == state["selected"]
            ),
            None,
        )
        if record['scene'].get('badge_rects'):
            self.dispersed_badges(draw,record,state)
            self.text(draw,(25,112),'斜め上方からの3D投影／z＝地面からの実高度 [m]／引出線＝実地点',20,fill='#86dbe6')
        if feature and not record['scene'].get('badge_rects'):
            for role, name in [("L", feature["selected_L"]), ("H", feature["selected_H"])]:
                x, y = project(record["scenario"]["coords"][ids.index(name) + 1], record["scene"]["camera"])
                self.text(
                    draw, (x - 25 + 20, y - 90 + 145), role + "：地点" + name, 23, fill="#ffdc91", stroke=1
                )
            self.text(
                draw,
                (25, 112),
                f"比較元 {feature['decision_id']}（{feature['physical_reference']['simulation_step'] * record['schedule']['dt']:.2f}秒）／数値・条件付き最適解は独立した静止カードを参照",
                20,
                fill="#ffdc91",
            )
        self.text(draw, (1340, 150), f"シミュレーション時間 {time:.2f} 秒", 30)
        self.text(
            draw, (1340, 199), f"実飛行 1.0倍速 ／ 点検 {len(state['visited'])}/{n}", 30, fill="#86dbe6"
        )
        target = state["selected"] or ("完了" if len(state["visited"]) == n else "待機")
        self.text(draw, (1340, 247), "現在の目標：" + target, 30)
        if sample:
            self.text(draw, (1340, 294), f"実速度 {float(sample['speed_mps']):.2f} m/s", 25)
            distance = current_target_distance(record["scenario"], pose, state["selected"])
            label = (
                f"実位置から目標 {distance:.3f} m"
                if distance is not None
                else f"直前の点検位置との距離 {float(sample['distance_to_goal_m']):.3f} m"
                if len(state["visited"]) == n
                else f"待機位置との距離 {float(sample['distance_to_goal_m']):.3f} m"
            )
            self.text(draw, (1340, 333), label, 25)
        self.map(draw, record, state, pose, time)
        for j, name in enumerate(ids):
            x = 1340 + (j // 5) * 278
            y = 726 + (j % 5) * 43
            draw.rectangle((x, y + 5, x + 26, y + 30), fill=priority_color(state["priorities"][j], 100 if record["scenario"].get("schema_version") == 3 else 5))
            self.text(draw, (x + 37, y), f"{name}  優先度 {priority_text(state['priorities'][j], record['scenario'].get('schema_version', 2))}", 23)
        point = project(pose, record["scene"]["camera"])
        detail = same_frame_detail(camera, point, record['scene'].get('detail_crop_size_px',(96,72))).resize((200, 150))
        image.paste(detail, (1070, 180))
        draw.rectangle((1069, 179, 1271, 331), outline="#86dbe6", width=2)
        self.text(draw, (1050, 145), "同じ実フレームの機体拡大", 20)
        self.text(draw, (25, 882), "実飛行の軌跡・全地点表示 ／ 全区間保持・省略なし", 28)
        if state["update"]:
            event = state["update"]
            u = event["update"]
            text = (
                f"地点Cの優先度 {priority_text(u['old'], record['scenario'].get('schema_version', 2), 3)} → {u['value']:g}"
                if event["type"] == "PRIORITY_UPDATED"
                else "有向辺 "
                + ", ".join(f"{a}→{b}" for a, b in u["edges"])
                + f" 倍率 {u['old'][0]:g} → {u['value']:g}"
            )
            self.text(draw, (25, 930), text + f"（更新 {event['sim_time_s']:.2f}秒）", 28, fill="#ffdc91")
        else:
            self.text(
                draw, (25, 930), "優先度は重要度であり、訪問順そのものではありません", 28, fill="#ffdc91"
            )
        self.legend(draw, 1017, record["scenario"].get("schema_version", 2))
        return image, state

    def map(self, draw, record, state, pose, time):
        coords = np.asarray(record["scenario"]["coords"])
        lo = coords[:, :2].min(axis=0) - 0.2
        hi = coords[:, :2].max(axis=0) + 0.2
        dispersed = bool(record['scene'].get('badge_rects'))

        def xy(p):
            ratio = (np.asarray(p)[:2] - lo) / (hi - lo)
            return 1375 + ratio[0] * 470, (548-ratio[1]*110) if dispersed else (650-ratio[1]*220)

        draw.rectangle((1335, 392, 1895, 708), fill="#1c3547")
        self.text(draw, (1345, 396), "XY全地点マップ（m）／線：実飛行" if dispersed else "全地点マップ（m）／線：実飛行", 23)
        trajectory = [
            xy([float(r["x"]), float(r["y"])])
            for r in record["telemetry"]
            if float(r["sim_time_s"]) <= time + 1e-9
        ]
        if len(trajectory) > 1:
            draw.line(trajectory, fill="#58d5ce", width=3)
        ids = record["scenario"]["checkpoint_ids"]
        for i, name in enumerate(ids):
            x, y = xy(coords[i + 1])
            draw.ellipse(
                (x - 9, y - 9, x + 9, y + 9), fill=priority_color(state["priorities"][i], 100 if record["scenario"].get("schema_version") == 3 else 5), outline="white"
            )
            if name == state["selected"]:
                draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline="#00e5ff", width=3)
            self.text(draw, (x - 13, y + 12), name + ("✓" if name in state["visited"] else ""), 20)
        if state["update"] and state["update"]["type"] == "COST_UPDATED":
            for a, b in state["update"]["update"]["edges"]:
                nodes = ["S"] + ids
                ai = nodes.index(a) if isinstance(a, str) else a
                bi = nodes.index(b) if isinstance(b, str) else b
                dashed_arrow(draw, xy(coords[ai]), xy(coords[bi]), "#ffdc91")
        x, y = xy(pose)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white")
        if dispersed:
            self.text(draw,(1345,580),'XZ補助投影：横x／地面からの実高度z [m]',20)
            for z in [1,2,3]:
                yy=696-(z-1)*40
                draw.line([(1385,yy),(1870,yy)],fill='#476170',width=1)
                self.text(draw,(1345,yy-12),str(z),20)
            for j,p in enumerate(coords[1:]):
                xx=1390+(p[0]-lo[0])/(hi[0]-lo[0])*465
                yy=696-(p[2]-1)*40
                draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=priority_color(state['priorities'][j],100 if record['scenario'].get('schema_version')==3 else 5),outline='white')
                if j in [int(np.argmin(coords[1:,2])),int(np.argmax(coords[1:,2]))]:
                    self.text(draw,(xx+7,yy-20),ids[j],20)
            xx=1390+(pose[0]-lo[0])/(hi[0]-lo[0])*465
            yy=696-(pose[2]-1)*40
            draw.ellipse((xx-4,yy-4,xx+4,yy+4),fill='#00e5ff')
