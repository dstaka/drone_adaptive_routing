# 2機NN協調：作業継続用handoff

全体は未完了。push・PR作成・merge・公開は禁止。今回の関連変更のみlocal commitする。既存のマージ済みspec、以前の単機成果物、無関係な未追跡ファイルを保存する。

Runのローカル絶対パス：
`outputs/dual_payload_coordination_20261027T144754Z`

- `progress-ja.md`：日本語の工程別状態。
- `progress.json`：更新日時と各workerの状態。600秒超の更新停止はstale／未確認。
- `timing/events.jsonl`：追記式の試行・時間記録。
- `timing/work-items.json`：派生表。未観測時間はnot recorded。
- `logs/`、`receipts/`、`audits/`、`recovered/`：失敗を含む確認結果。

Sidechat確認文：
「outputs/dual_payload_coordination_20261027T144754Z のprogress-ja.md／progress.jsonとtiming記録を読み、最終更新時刻、現在の作業、開始からの経過時間、完了数、待機・失敗理由、次の工程を教えて。古い情報や未計測値はその旨を示して。」

承認済み仕様は [spec](../specs/2026-10-27-dual-drone-learned-coordination/spec.md)、計画は [plan](superpowers/plans/2026-10-27-dual-drone-coordination.md)。今回の保存要件変更はspec §11.3。TB級の全raw保存・回収は不要。安全監査の省略や精度低下は承認されていない。

GPUサーバは利用者が今回指定した接続先を使い、`/workspace/dual_payload_coordination_20261027T144754Z/` に隔離。接続情報はrunのサーバreceiptを参照。ソース・tests・configs・spec・物理asset・固定制御器の転送は承認済み。旧Genesis飛行NN等3ファイルの追加転送は承認されておらず、利用者がGenesisテストをスキップするよう指示したため実施しない。サーバは再作成でデータを失う領域なので、必要物はローカルへ回収・サイズ／SHA照合する。

次の依存工程：compact記録とonline監査の一致・native再現、全native資格（330秒×7、混合36、受動／powered・制動・精度）、CPU/GPU実測と環境選択。その後のみ3seed固定予算学習、共通validation300選定、joint freeze、未使用routing300と別physical127へ進む。候補なしなら学習未達として停止し、held-outを開かない。

最終3本の新規全編1倍速動画を作成した時点でローカルパス・SHA・実測値を提示する。人手確認の明示的な完了前は全体完了にしない。現時点では最終動画も人手確認もない。

最新の実行状態（必ずprogressの更新時刻を確認）：

- 最新765file固定コピーの全体回帰は1,732passed・48skip・1除外、532.10秒。C++ SDKのinclude環境を明示した。関連Ruffも成功。直前の5失敗／10fixtureエラーも保存し、修正理由を結果文書へ記録した。
- `qualification-batch-001` は各330秒を実行後、lazy importの依存集合漏れにより7失敗・74未着手。3,825.29秒。失敗を合格へ変換しない。事前importと実1tick preflightを追加した新規 `qualification-batch-002` が進行中。必要結果39files437,028bytesは `recovered/delivery-qualification001-results` へ照合回収。最初の大きな回収はSSH切断により不完全なまま残るので、完全回収と数えない。
- 実測用の正式な入力版はサーバ `run/experiment-inputs-v2`、実行ソースは `source-production-015` に015/016 overlayを適用したもの。binding SHAは `3d404dfc89b463add16eff0c6aa68b41578afa797e9b73eac6ac4ef5dd7c2cca`。学習許可receiptではない。v1は保存パスに拘束されたライブラリをコピー先から読み込もうとして失敗したsetup版であり、本番に使わない。
- C++ libraryは元の `run/native-build-009/dual_native_audit.so` を使う。SHA `c24de2fe6829e2a3512ae1ba31d3f4fb370c3adc39e3dcbad5e0d5ac87ea0ada`。build receiptは元の絶対パス・headers・sourceを照合する。`delivery-build-009` は回収packetであり、そのまま実行パスに置換できない。
- `benchmark-pipeline-002` は81条件完了と入力固定を待ち、quota内の1/2/4/7 workers × CPU/CUDAの有限計測を順次実行する。warm-up4096＋3×4096、各5epochで、本番遷移0。補助資格試験は `qualification-supplements-002`。途中状態や単独のcase合格から学習を許可しない。
- 学習／validationのケースは最初に宣言したローカル `declaration-v1` のbytesを使う。サーバ再生成には5index・27初期quaternion scalarsで最大3.47e-18の差があり、元のtraining JSONL SHA `b19b3a92834ce96ae85171b511e0d33885aa14a0ebd94178cfeadc0620331f85` を `configs/dual_coordination_cases` に固定した。両生成版と差分を保存。validationは一致した。
- 選定はN42/N43/N44と両baselineの300件ずつ、計1,500件。選定後に同じ300件×両ablationの600診断を実施する専用runnerを追加した。診断結果を再選定へ返さない。projectionはこの600件も含む。
- 開発用B-static動画は `recovered/development-video-002/video/view.mp4`、SHA `13adbb23c1c0cfd4108ad2ebd98670414ac4a389b25bf12e5908f0c4058a282d`。25.66秒、各5地点、変更0、最小間隔3.38093252m、安全hold0。最終NN動画／人手確認とは別である。
