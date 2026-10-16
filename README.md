# 適応型ドローン巡回点検

A〜Jの10地点を、同じ学習済みルーティング方策と飛行方策で巡回します。点検完了後の境界で、地点Cの優先度上昇や有向辺B→Aのコスト増加を受け取り、次の未訪問地点を選び直します。実際のGenesis映像、位置・速度、点検・更新ログを同じ実行で記録しています。

[今回の承認済み仕様](specs/2026-10-14-realtime-demo-video/spec.md)、[実装計画](docs/superpowers/plans/2026-10-16-realtime-demo-video.md)、[成果物・検証・再実行の引き継ぎ](docs/realtime-video-handoff.md)を参照してください。先行する[経路最適化](docs/scalable-routing-handoff.md)と[全地点物理飛行](docs/visible-flight-handoff.md)の結果も保持しています。

## 実測結果と動画

固定3シード×3シナリオは全9試行成功、90/90地点の点検を完了しました。seed 1の3本を選び、全飛行を省略せず1倍速で編集します。通常17.82秒、緊急17.23秒、コスト変化17.23秒が実シミュレーション時間です。動画の長さは出力フレーム単位に量子化し、末尾の実点検完了状態を表示します。

成果物ルートは `outputs/realtime_20261016_v2/`（Git管理外）です。

- `sources/{normal,emergency,cost_shift}_seed_1/`：元raw・時刻表・物理ログの完全コピー。
- `raw/*/raw.mp4`：実物理時刻へ正規化した1倍速raw。元の境界追加撮影との対応を `frames.csv` と `timing.json` に記録。
- `full_normal.mp4`、`full_emergency.mp4`、`full_cost_shift.mp4`：全10点検の日本語HUD付き全編。
- `demo_full.mp4`、`submission.mp4`：全3シナリオと独立した説明カード。今回の素材は全飛行が収まるため、省略はありません。
- 各動画の `.timeline.json`、`.selection.json`、`.verification.json`、成果物全体の `manifest.json`。
- `report/report.pdf`：今回の日本語レポート。

飛行はシミュレーション時間に対して1.0倍速です。計算に要した実時間とは別です。説明カード中はシミュレーション時計を進めません。優先度1〜5の固定色と数値、目標の輪、点検済みのチェックを独立して表示します。予測順序と実飛行は区別し、総コストを物理時間・電力と呼びません。

上位はCPUのMaskablePPO、下位はGenesis/RSL-RLのPPOです。ルーティング採用seedは42、観測は `routing-v2-n10-324`、324要素・10行動です。飛行には追加学習後の `model_1300.pt` を使用し、旧モデルへ戻していません。モデルハッシュと保存先は引き継ぎ記録に記載しています。

全フレーム・PTSの独立検証、静止フレームとPDFの目視に加え、2026年10月17日にユーザーから動画の目視OKを受領しました。残っていた人手確認を完了として記録し、今回の仕様受入はPASSです。改善案は別spec／PRで扱います。

## 判断と物理判定

目的関数は α×移動コスト＋β×優先度を重みとした名目完了時刻のペナルティです。今回はα=β=1、名目速度1 m/s、サービス0.50秒、基地帰還なしです。有向辺倍率は移動コスト成分だけに適用し、物理速度や名目距離を変更しません。

コスト変化の物理判断 `cost_shift_seed_1 / decision-1` では、近いL＝地点Cを先にする条件付き最適コスト149.097に対し、遠い高優先度H＝地点Hは168.796です。同じsnapshotの厳密解J*とAI予測も149.097です。この例の最適性は「現在の条件を固定した残り巡回」に限ります。一般評価では常に最適ではなく、採用seedの静的100例の平均AI/厳密解比は約1.030、動的50例中48適用例の改善は2例です。

実点検は半径0.20 m以内・速度0.30 m/s以下を連続0.20秒維持し、合計0.50秒滞在して確定します。逸脱で滞在をリセットします。落下、reset、非有限値、目標不一致、通信異常、timeoutは成功に数えません。固定初期状態の決定論的デモであり、擾乱、障害物認識、実機安全性、実際の設備不具合検出は未検証です。

## CPUで再生成する

既存のローカルCPU環境 `.venv-routing311` を使用します。Linuxの飛行・ルーティングロックは別環境用です。今回、依存版変更、再学習、追加撮影、GPU利用はありません。CPU・GPUの試験値を混同しません。

```bash
export PYTHONPATH=src
.venv-routing311/bin/python -m pytest -m 'not gpu' -q
.venv-routing311/bin/python -m ruff check src scripts tests

# 出力先は未使用の名前にする。fullは自然尺、submissionの上限は速度を変えない。
.venv-routing311/bin/python scripts/build_video.py --config configs/video.yaml \
  --input-root outputs/visible_20261015/remote/outputs/visible/acceptance_v1/genesis \
  --explanations outputs/visible_20261015/remote/outputs/visible/verification_v2/decision_explanations.json \
  --mode full --out outputs/realtime_new/demo_full.mp4
.venv-routing311/bin/python scripts/build_video.py --config configs/video.yaml \
  --input-root outputs/visible_20261015/remote/outputs/visible/acceptance_v1/genesis \
  --explanations outputs/visible_20261015/remote/outputs/visible/verification_v2/decision_explanations.json \
  --mode submission --duration 120 --out outputs/realtime_new/submission.mp4

# ffprobeの実行ファイルを指定する。既存の検証結果は上書きしない。
.venv-routing311/bin/python scripts/check_realtime_media.py --root outputs/realtime_new \
  --ffprobe outputs/realtime_20261016_v2/tools/ffprobe
.venv-routing311/bin/python scripts/build_report.py --source docs/report_ja.md \
  --out outputs/realtime_new/report/report.pdf
```

`configs/video.yaml` の `fps` は25/30、`playback_rate` は1.0のみです。`submission_max_s` と `--duration` は提出尺の上限であり、飛行速度を計算する値ではありません。素材とカードが上限に収まらなければ明示的に失敗します。旧timeline v1と `scripts/check_media.py` は履歴資料用で、新1倍速受入とは分離しています。

今回の独立検証は、元の物理step・実timestep・元rawのPTS・出力の符号化PTSから時刻を再計算します。全デコード、各飛行フレームの参照情報・日本語HUD・元映像由来の機体拡大、イベント、末尾coverage、全地点を照合します。動画の `playback_rate` 宣言だけでは合格しません。

## 保存・予算・出典

大きいモデル・動画・ログは `outputs/` に保存し、Gitにはコード、仕様、テスト、文書、少数の圧縮証拠画像だけを含めます。歴史資料 `adaptive_drone_patrol_20261013/` と過去成果物は実装先にせず保護します。公開参照はリポジトリ相対パスです。

GPU累計は2,803.57274048205秒、残り25,996.42725951795秒（8時間枠）です。今回の増分は0秒。新サーバは使っていません。追加物理撮影時は [前回の再構築・台帳手順](docs/visible-flight-handoff.md) に従い、現在の台帳と新しい実験名・出力先を使います。

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にGenesis、提供ノートブック、SB3、RSL-RL、Noto Sans JPの継承元・ライセンスを記載しています。動画公開、コース提出、push、PR作成・マージは行いません。
