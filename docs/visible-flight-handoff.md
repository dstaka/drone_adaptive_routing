# 10地点物理飛行の引き継ぎ

承認済みの[仕様](../specs/2026-10-14-visible-multigoal-flight/spec.md)を、scalable-adaptive-routing のマージ済みコード ff346a0 から実装した記録です。実装先はリポジトリルートです。後続の realtime-demo-video は対象に含めていません。

実装と物理受入を完了しました。確定した受入結果・転送照合・GPU総時間は以下に記載します。

## 保存先と環境

公開参照はリポジトリ相対です。大きいモデル・動画・ログ・接続先依存の設定は、Git管理外の outputs/visible_20261015/ に保存します。旧モデル・既存成果物・他PR用specは保持しています。

- 飛行環境: Linux Python 3.12.14、.venv-flight。[flight lock](../requirements/flight-linux.lock.txt)の169固定版が一致。
- CPU環境: Linux Python 3.12.14、.venv-cpu。[CPU lock](../requirements/routing-scalable-linux.lock.txt)の33固定版が一致。
- uv の初回解決は追加index優先のため失敗。版を変えず、明示されたPyPI/PyTorch両indexを検索する --index-strategy unsafe-best-match で再現しました。既存環境は変更していません。
- ルーティングモデルは既存採用seed42を継続。SHA-256: c4c2a937d26650c09a7810073382263e79ba58fcf238a752387fd41d45a2bc2a。
- 元の飛行モデル SHA-256: 5d9d193a6fc4f6066c6fcb66a41e7ccc7203d450d2515c95a874deb3830d33d7。上書きせず保持。
- 追加学習後 model_1300.pt SHA-256: 4a6f3b333735f3c10ed4e51f2ddd3466308cf90b7dfb3bec289b519a084a539f。
- 追加学習設定 cfgs.pkl SHA-256: 25720a5b59b1edd769ed36ecc29be16febde3ad6088dcc6bdd878f71d2f8fff1。

## 実装と検証の判断

任意NのID・mask・完走判定へ対応し、N=10/12をCPUで検証しました。物理到着は実位置・速度による0.20m、0.30m/s、0.20秒連続確認、0.50秒連続点検のままです。中断時は滞在をリセットし、失敗・reset・target mismatchは完了に数えません。傾斜は実quaternionから算出し、上限60度・最低高度0.1mを維持します。

出発前に実ホバー2秒を撮影して全地点・初期優先度を示します。地点は固定・非衝突のGenesisランドマークで、旧単一target描画は無効です。優先度は世界座標へ投影した数値バッジと赤の固定スケール、選択は追加輪郭、訪問済みはチェック、緊急は警告記号で区別します。元rawは保持し、注釈版は別ファイルです。動的材質変更APIは利用せず、specで許可されたneutral landmark方式を明示しています。

物理decisionごとに論理境界snapshotと実poseを保存し、厳密残りコストはCPUでオフライン再計算します。L/H条件は計画距離と実pose距離の両方を確認します。J*とQ*は固定した観測条件の巡回目的関数であり、実軌跡・エネルギー・未知イベントに対する最適性ではありません。

## 開発試行と追加学習

固定9試行とは別に、すべての開発失敗を保持します。

1. 元モデルの200ステップ基準飛行・カメラ・全デコードは成功、reset 0。
2. 初回投影照合は、診断コードでOpenGL投影とOpenCV外部行列を混在させて飛行前に失敗。固定版Genesisのintrinsics/extrinsicsを確認して修正しました。
3. 元モデルの長距離B→J切替は、183ステップ目に傾斜66.710922度で安全停止。マーカー有無で同じ状態・停止が再現し、相対位置観測は飽和していません。adapterはpose・速度・motor historyを変更していません。
4. 同じ17観測・4モータ・物理モデル・目標範囲で1001反復を追加学習。学習seed7、256環境、元iteration300から1300へ進めました。300ステップ周期の全範囲retargetと連続ホバーを学習し、学習側もroll/pitch上限60度にしました。GPUジョブ722.873721秒。
5. 新モデルは長距離・反転・同地点継続を含む8連続操作を完了。マーカー有無の1508サンプルで位置・速度・quaternion・モータ出力が完全一致。最大相対位置観測0.628198。
6. 開発ミッション初回は、step前の未初期化upstream base_eulerを読む不具合で起動失敗。実quaternionから算出する修正と回帰テストを追加しました。転送前のアーカイブ生成完了を待たなかった失敗もログに保持しています。
7. 修正後の開発normal seed11は10地点を17.82シミュレーション秒で完走。seed11は固定受入9試行の分母に含めません。

独立レビューで実pose距離と計画距離の混同、およびモデル・形状・カメラ・ソースの固定照合不足を指摘され、失敗するテストを追加して修正しました。未解決の重大コード指摘はありません。

## 環境の再構築

新しいLinux GPU作業ディレクトリへソース・両モデル・設定を転送し、サイズ・SHA-256を照合してください。既存出力先を再利用せず、実行設定はその作業ディレクトリで作り直します。

~~~bash
export PYTHONPATH="$PWD/src"
uv venv --python 3.12 .venv-flight
uv venv --python 3.12 .venv-cpu
uv pip install --index-strategy unsafe-best-match --python .venv-flight/bin/python -r requirements/flight-linux.lock.txt
uv pip install --index-strategy unsafe-best-match --python .venv-cpu/bin/python -r requirements/routing-scalable-linux.lock.txt
.venv-cpu/bin/python -m pytest tests -m 'not gpu' -q
.venv-cpu/bin/python -m ruff check src scripts tests
~~~

GPU実行は必ず最新の累計台帳を引き継いで adaptive_drone_patrol.jobs から起動します。履歴8時間枠をリセットしません。モデル・形状・カメラ・timeout・到着設定を変えた場合は、比較試験9件を改めて固定して実行します。形状やroutingモデルを変えた場合は、前PRの影響するrouting受入も再評価します。

## 検証記録

受入9試行、オフライン検証、表示QAと転送照合を分けて記録しています。

### 固定受入の確定結果

固定モデル・同じ初期形状・到着設定・カメラで、各シナリオのseed1・2・3を一度ずつ実行しました。9件すべてを分母に含め、置き換え・成功試行の選別はありません。

| シナリオ | 成功/試行 | 各seedの実シミュレーション時間 | 実訪問 |
| --- | ---: | ---: | ---: |
| normal | 3/3 | 17.82秒 | 各10地点 |
| emergency | 3/3 | 17.23秒 | 各10地点 |
| cost_shift | 3/3 | 17.23秒 | 各10地点 |

90/90点検が連続0.50秒の距離・速度条件を満たし、reset・終了・fall・target mismatch・非有限状態・timeoutは0件です。滞在中の最大残差は0.19975528263849074m、最大速度は0.2997811212976476m/s。全legの最大roll/pitchは33.52745254722287度、最低高度は0.8046350479125977mです。安全条件は緩めていません。

B完了後のイベントで、emergencyはCだけの優先度2.660865759308→5、cost_shiftはB→Aだけの倍率1→5を更新し、実際の次選択がA→Cへ変わりました。両者とも変更後の予測suffixは C D F J I H G E A。物理選択の記録と、仮想counterfactualの記録を区別しています。post-eventの残り目的関数改善はemergencyで25.219822682394、cost_shiftで7.035441364304です。cost_shiftの移動コスト成分は7.8→5.4です。

全90判断から physical 説明を再生成し、42判断が実pose距離を含むL/H条件を満たしました。normal seed1 のdecision-2はE/Hの例です。計画距離は0.60/1.341640786500m、優先度は2.120156088347/2.693373135601。J*=Q*(E)=147.981512600810、Q*(H)=165.512743820382、J_policy=147.981512600810です。実pose由来の距離は各boundaryと各表示フレームに別保存しています。675完全巡回の独立再計算の最大成分差は5.684341886080802e-14でした。

9本すべてを全デコードし、7995個の実フレームの数・PTS・simulation stepを照合しました。全フレームの補助コントラスト監査で10ランドマークの不足0、全telemetryで機体の画面外0。seed1の3シナリオについて、開始・全10回の選択・イベント・終了のraw/注釈フレームを目視しました。優先度は訪問後も保持され、緊急時にCだけが濃赤・5へ変わり、有向コストは破線矢印と旧新倍率で表示されます。読みやすい常時overviewに加え、同じrawフレームから切り出した機体拡大を常時表示します。

GPU使用の累計は2803.57274048205秒、残り25996.42725951795秒です。継承分1233.7003595418064秒を含み、今回の増分は1569.8723809402436秒。固定9試行とGPU契約テストの監督ジョブは484.7642950396985秒でした。GPUジョブはすべて終了しており、インスタンス契約・停止操作は行っていません。

これは固定初期状態の決定論的デモ安定性の結果です。3シードの軌跡・結果は一致しました。擾乱下の一般的な頑健性評価は未実施です。最終編集・リタイミング・提出用合成は、指定どおり後続PRへ残しています。

### 再実行設定の生成

以下はリポジトリルートから、転送済み成果物をそのまま配置したLinux作業環境で実行します。相対参照から現在の作業環境に合うパスを作り、既存設定を上書きしません。

~~~bash
export PYTHONPATH="$PWD/src"
.venv-cpu/bin/python - <<'PY'
from pathlib import Path
import hashlib
import yaml
root = Path.cwd()
asset = root / "outputs/visible_20261015/remote"
flight = asset / "outputs/visible/finetune_v1"
routing = asset / "outputs/scalable_20261015/normalized-src/outputs/training/seed_42.zip"
config = {
    "flight_checkpoint": str(flight / "model_1300.pt"),
    "flight_cfg": str(flight / "cfgs.pkl"),
    "routing_model": str(routing),
    "hover_source": str(root / "third_party/genesis_drone/hover_env.py"),
    "planner_python": str(root / ".venv-cpu/bin/python"),
    "scenario_files": [str(root / f"configs/demo_scenarios_scalable/{s}.yaml")
                       for s in ["normal", "emergency", "cost_shift"]],
    "flight_eval_config": str(root / "configs/visible_flight_eval.yaml"),
    "budget_ledger": "outputs/gpu_ledger.json",
    "gpu_budget_s": 28800,
}
for path_key, hash_key in [
    ("flight_checkpoint", "flight_checkpoint_hash"), ("flight_cfg", "flight_cfg_hash"),
    ("routing_model", "routing_model_hash"), ("hover_source", "hover_source_hash")
]:
    config[hash_key] = hashlib.sha256(Path(config[path_key]).read_bytes()).hexdigest()
with (root / "configs/execution_visible_rerun.yaml").open("x") as stream:
    yaml.safe_dump(config, stream)
PY
.venv-flight/bin/python -m adaptive_drone_patrol.jobs   --out outputs/visible_rerun/job --ledger outputs/gpu_ledger.json   --budget-s 28800 --timeout 5400 --   .venv-flight/bin/python -u scripts/final_gpu_validation.py   --config configs/execution_visible_rerun.yaml --out outputs/visible_rerun/acceptance
.venv-cpu/bin/python scripts/verify_visible_flight.py   --root outputs/visible_rerun/acceptance/genesis   --model outputs/visible_20261015/remote/outputs/scalable_20261015/normalized-src/outputs/training/seed_42.zip   --model-hash c4c2a937d26650c09a7810073382263e79ba58fcf238a752387fd41d45a2bc2a   --font outputs/visible_20261015/remote/NotoSansJP.ttf   --out outputs/visible_rerun/verification --annotate-seed 1
.venv-cpu/bin/python scripts/inspect_visible_frames.py   --root outputs/visible_rerun/acceptance/genesis   --annotations outputs/visible_rerun/verification   --font outputs/visible_20261015/remote/NotoSansJP.ttf   --out outputs/visible_rerun/visual_qa
~~~

最新の累計台帳を先に配置してください。すでに別ジョブで更新された台帳を、古いコピーで上書きしてはいけません。jobs は残予算からtimeoutを制限し、同台帳でのGPU同時実行を拒否します。

今回実際に使ったソースは、受入開始前の implementation-frozen.tar.gz と最終の final-source.tar.gz を両方保持します。受入後の差はオフライン証拠照合・注釈表示・回帰テストです。物理ミッションの条件・routing/flightモデル・raw記録は変更していません。

### ローカル成果物の主要参照

- モデル: outputs/visible_20261015/remote/outputs/visible/finetune_v1/model_1300.pt、同じディレクトリの cfgs.pkl。
- 固定条件: outputs/visible_20261015/remote/outputs/visible/frozen_suite.json。
- 全9件: outputs/visible_20261015/remote/outputs/visible/acceptance_v1/genesis/。各runに scenario.json、events.jsonl、decisions.jsonl、telemetry.csv、frames.csv、raw.mp4、scene_manifest.json、mission_summary.json。
- 全物理説明・受入: outputs/visible_20261015/remote/outputs/visible/verification_v2/。
- 最終注釈動画: outputs/visible_20261015/remote/outputs/visible/annotations_v3/{normal,emergency,cost_shift}_seed_1.annotated.mp4。
- 最終実フレーム: outputs/visible_20261015/remote/outputs/visible/visual_qa_v3/。
- 開発失敗・監督ログ: outputs/visible_20261015/remote/outputs/jobs/ と outputs/visible_20261015/remote/outputs/visible/。
- 原実行設定は outputs/visible_20261015/remote/configs/execution_visible*.yaml。元サーバ依存のパスを含むため、上の生成手順で作り直してください。

確定転送は465ファイル、展開後91,402,466 bytes、アーカイブ60,684,795 bytesです。アーカイブSHA-256は d6cba7b79cfeb60cdb9c29b7f94889fb69d6c650f974508dba4317a332d9a88f。各ファイルのサイズ・SHA-256とローカル照合結果は、[manifest](evidence/visible/delivery-manifest.json)と[receipt](evidence/visible/delivery-receipt.json)に記録しています。

ローカル転送照合は465/465件PASSです。ローカルとLinux固定CPU環境はいずれも123 passed、GPU契約テストは1 passed、Ruff・差分空白検査はPASSです。公開文面・差分・commitメタデータから個人名、ホスト固有絶対パス、接続情報、秘密情報を除きます。納品revisionは、この記録を含む1件のcommitです。commit後のreceiptはGit管理外の outputs/visible_20261015/final-commit.json に保存します。
