# 2機NN協調：作業継続用handoff

2026-10-29 15:15 SGT追記: 学習PID38003と監視PID43940の実cmdlineを再確認。15:14 SGT時点seed42=48／256更新、196,608遷移以上・stage1、他seed0。親pipelineはtraining、heldout-cohorts-001は存在しない。新たに `run/recovered/delivery-frozen-inputs-001/` を回収（185files／51,921,409bytes、送受信サイズ・SHA一致）。`root/` 以下に全176個のLearningLock固定ファイルとreplay参照・補助情報がある。`manifest.json` のrestoration_rootへ同じ相対構造で復元する情報であり、lockを書き換えたローカル実験を作ったわけではない。整合性receiptは `run/receipts/frozen-input-closure-verification-001.json`。最終terminal3本・joint freeze・評価結果・動画は引き続き必要。回収1回目のSSH失敗、2回目成功を別試行として保存。準備時間だけはtimer API指定誤りによりnot recorded。

2026-10-29 12:42 SGT追記: 全CPU構成完了、中央値1／2／4／7=2270.579068／1259.379690／687.722418／407.591166秒。7ワーカーで本番学習PID38003が稼働中。LearningLockは `run/cpu-production-pipeline-004/learning-lock.json`、SHA `eccdd3f24346c0745c2ea054ca206b8297519ab74f6baa8b8a58415ac803fa92`、固定日時2026-10-29T01:48:48.649348Z。概算全工程8.56日、学習3.62日、保存17.54GB（類似実測からの推定、ETA保証なし）。最新確認はseed42更新24／256・98,304遷移以上、seed43／44は0。学習を再起動しない。親pipelineのproduction_transitions=0は全seed終了時に更新される欄なので、途中カウントには使用しない。

学習カウントはローカル `run/remote-progress.json` の `snapshot.jobs` 内 `production-training-observer-001`（server PID43940、60秒更新）を読む。`progress.json` の `remote_snapshot_path` から辿れる。source timestampとheartbeatを必ず確認し、部分rolloutはnot recorded。監視だけの失敗は学習停止を意味しない。固定source018を変更せず、外部script `run/observe_dual_training_001.py` SHA `1bbf4e675260fda2c7248bb2f0551f1b7663524f1089316b1beeb8bdd9902eea` を起動した。関連67テスト成功。初回launchはPopen後のreceipt保存先不足でCLI失敗したが、既存PIDを確認してreceiptのみ修復し、二重起動していない。

初回再開用状態・LearningLock・全CPU比較metadataはローカル `run/recovered/delivery-production-start-001/`、41files／12,390,496bytes送受信サイズ・SHA照合済み、送信元保持。モデルは更新6・部分rollout476時点、selectable=falseなのでvalidationには使わない。最終的に必要な固定ソース／設定／受入結果／terminal3本の回収inventoryは引き続き照合する。途中時間表は `run/timing/report-001/per-attempt-ja.md`（723試行、完了主張なし）。観測の中断・通信切断の未計測内訳を実行時間へ埋めない。評価待機34139、履歴圧縮34074、ローカルmirror48044を実PIDで再確認してから状態を扱う。

最終時間表の必須追加: `run/timing/clock-audit-001.json` のローカル10試行はUTC spanとmonotonicが不一致。元の値を保存したままappend-only注記でunclassified spanに分類した。暫定表001はこの分類前のsnapshotなので最終unionに再利用しない。最終時は新しい両ホストjournalから再集計し、UTC span／monotonic計測／差を別欄にし、未計測の中断内訳をnot recordedとする。学習・サーバ側の計算はこの観測中断中も継続していた。差の原因はnot recorded。

2026-10-29 08:36 SGT追記: CPU2完了、中央値1259.379690秒、構成全体5032.897539秒。測定反復は1..3（1259.379690／1260.999822／1258.687686）、0はwarm-up1248.556209秒なので中央値から除外。完了記録9files／180,903bytesを `run/recovered/delivery-cpu2-benchmark-001/` に独立サイズ・SHA照合して回収、元データ保持。CPU4の実PID35326、親32398を確認。CPU7未開始、学習待機34138／評価待機34139を維持。本番学習0、未使用held-out未開封。残る実測後のcost reduction／LearningLock／固定3seedは既存pipeline004、動画は既存成功手順のpostpipeline002に従う。

2026-10-29 07:58 SGT追記: 最新指示により動画は実績のある既存手順を再利用し、CPU/GPU描画の追加実験・GL識別調査は中止。予定した描画probeとラッパーは未実行のまま外した。物理計算・NN推論/学習はCPUを維持する。CPU2の測定1回目4096遷移+5epochは1259.379690秒、warmupは1248.556209秒。CPU4/7を含む残り計測後に固定3seedへ進む。学習待機004、学習後待機002が稼働中。本番学習は未開始。

2026-10-29 07:47 SGT追記: ユーザーがGPU制限の用途を明確化。物理シミュレーション・NN推論/学習はCPU、動画のOpenGL/EGL描画にはGPUを使用可。旧動画のGL識別はnot recordedのため断定しない。今回の描画はNVIDIA EGLのvendor/rendererを別途記録する。学習前の待機処理を003へ更新し、学習プロセスのCUDA可視性は空に保つ。描画と学習は同時実行しない。本番学習は未開始。

2026-10-28T23:15UTC更新: ユーザー指示で残工程をCPU専用へ変更。GPU物理は未適格・全体速度未測定であり、加速不能とは断定しない。native物理1ワーカー+CPU NNの中央値2270.579秒、同GPU NNは2232.989秒（約1.66%短縮、warmupを除いた3回）。追加GPU開発・資格試験を停止し、候補006の331件成功と1周期36条件の局所成功、途中停止したstrict003を保存。strict003は最終レポートなしの未完了、passive32未実施。CPU1実測を保持し、2/4/7構成を新しいbenchmark-pipeline-006-cpu-onlyで実測中。本番学習0、validation/held-out未実施。CPU専用の明示記録を要求する管理reducer変更はbenchmark/campaign回帰114件成功148.60秒。既存source018の物理・学習runtimeは変更しない。


2026-10-28 20時UTCの更新：native資格はreport006の81/81・補足10項目合格を維持。本番学習・validation・held-out・最終NN動画は未着手。CPU1構成の有限計測（warm-up4096＋測定4096×3、各5epoch）は完走し、測定反復中央値2270.579秒、構成全体9101.446秒だった。その他CPU/CUDA構成は継続中。容量測定は起動順レースによる旧試行失敗を保存して再実行し、再開用最大履歴の合成probe284,394,527bytes／atomic保存1.749秒・readback1.296秒、実compact archive4,773bytesを確認した。これらは未学習の容量見積りで、学習済みモデルではない。

GPUの既存native36条件に対する30秒閉ループ比較は全件settling・完了時間差≤5%を通過し、最大差0.14782%、計測271.989秒だった。最初1秒の同一motor入力比較も別途36件通過済み。ただし全内部形状・区間安全、受動系、力・仕事、完全再開、4096＋5epochのGPUパイプライン計測は残り、GPUは未承認。候補001の97pass／2失敗はcuda表記とcuda:0の比較不整合、候補002の243pass／12失敗は同種の補助処理と設定propertyの厳密inventory不整合だった。新候補003の関連テストは実CUDA76pass／4失敗まで進み、残る4件は空Warp配列のptr=Noneを未対応とする設定・再開処理で、修正中。失敗を物理合格に読み替えない。003テストは後続native+CUDA構成の未測定warm-up中に終了し、測定3反復との重複なしを保存した。

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

- **予測時間の修正後が現在の実行版。** spec §4.3が要求する最悪資格停止時間＋0.20秒に対し、旧ソースは現在の停止軌道だけから2秒などを選び得た。実測最悪3.35秒との不一致を確認し、global下限4.0秒と独立した実測比較gateを追加した。旧ソースで失敗する静止時／2秒より先の交差テストを保存した。関連26合格3skip、実MuJoCo事前47合格28.64秒。修正後の全体回帰は**1,735合格・48skip・1除外・失敗0／エラー0、523.09秒**。
- 一時テストコピーが過去の回帰用入力4ファイルを欠いた先行試行は、1,717合格・5失敗・13fixtureエラー、543.66秒。入力455,723bytesだけをサイズ／SHA照合して添付し、コードを変えずに失敗18件が26.68秒で合格、その後の全体再実行も合格した。過去の2機物理評価として流用したものではなく、旧ソフトウェア回帰の入力である。両試行とsetup不備を保存した。
- 現在の固定入力は `run/experiment-inputs-v3`、binding SHA `ae224409dbbd8a245393d0d189eaf78cf07df368060f1c2d44184933175a3595`、ソースは `source-production-018`。`qualification-batch-003` が新規81条件を実行する。`qualification-supplements-003` は新ソースの停止7条件等と開発pilot005／compiled replay006を実行し、停止集計は`qualification-stopping-reduction-003`、総合判定`qualification-report-006`が81/81・10checks・errors0で合格した。実測は`benchmark-pipeline-005`、容量計測待ちは`storage-measurement-pipeline-004`。report004は集合のJSON保存失敗、005は宣言済みpassive-half32を16と誤認した集計不整合。修正版は凍結source018の外の`run/qualification_reducer_006.py`で、そのSHAをreportへ保持する。過去の失敗報告は削除しない。
- 旧`qualification-batch-002`は81件とも完走したが、global予測時間の不備があるv2ソースの診断結果として保存し、正式資格へ流用しない。`benchmark-pipeline-002`と`qualification-report-pipeline-003`は実測・判定前に停止した。旧v2に本番遷移やheld-out開封はない。
- サーバの利用期限・費用上限は利用者が「期限・上限なし」と明示した。有限3seed・各1,048,576遷移、資格・選定・held-outの前提は維持する。今回の実装はlocal commit `55ef0bf` に保存済み。
- 2026-10-28T13:51Z時点で資格batch002は61/81合格、7実行中、13未着手、失敗0。最初の330秒×7条件は全件完走。補足11コマンドは終了コード0だが、7停止条件の独立trace再構成と全81条件の総合判定は別工程である。資格全体の合格を推測しない。
- `qualification-stopping-reduction-002` は実traceを再構成中。`qualification-report-pipeline-003` は81条件と停止再構成の合格を待って、固定入力v2に対する総合判定を行う。pipeline002はjournal引数誤りで起動失敗し、ログを保存した。失敗試行の計算時間はnot recorded。後継003は別ファイル・別出力であり、資格caseを再実行したものではない。
- 開発B-dynamic 3件はnormal/priorityが各27.66秒で成功し、availability-delayが52.40秒で`leg_timeout:1`、担当完了数[3,5]。失敗も保存した。各wallは100.1270／100.5822／146.0790秒で、資格試験と同時実行した開発コスト類例であり、専有benchmarkや受入結果ではない。
- `recovered/delivery-timing-snapshot-003` は13:47:08Z時点のサーバtimingコピーであり、最終journalではない。5files1,355,846bytes、開発3件32files6,766,023bytes、再現補足2files41,995bytesを送受信サイズ/SHA一致で回収済み。進行後のjournalも再回収する。
- 最新765file固定コピーの全体回帰は1,732passed・48skip・1除外、532.10秒。C++ SDKのinclude環境を明示した。関連Ruffも成功。直前の5失敗／10fixtureエラーも保存し、修正理由を結果文書へ記録した。
- `qualification-batch-001` は各330秒を実行後、lazy importの依存集合漏れにより7失敗・74未着手。3,825.29秒。失敗を合格へ変換しない。事前importと実1tick preflightを追加した新規 `qualification-batch-002` が進行中。必要結果39files437,028bytesは `recovered/delivery-qualification001-results` へ照合回収。最初の大きな回収はSSH切断により不完全なまま残るので、完全回収と数えない。
- 保存済みの旧入力v2はサーバ `run/experiment-inputs-v2`、旧ソース`source-production-015`。binding SHAは `3d404dfc89b463add16eff0c6aa68b41578afa797e9b73eac6ac4ef5dd7c2cca`。予測時間修正前なので本番に使わない。v1も保存パスに拘束されたライブラリをコピー先から読み込もうとして失敗したsetup版であり、本番に使わない。
- C++ libraryは元の `run/native-build-009/dual_native_audit.so` を使う。SHA `c24de2fe6829e2a3512ae1ba31d3f4fb370c3adc39e3dcbad5e0d5ac87ea0ada`。build receiptは元の絶対パス・headers・sourceを照合する。`delivery-build-009` は回収packetであり、そのまま実行パスに置換できない。
- `benchmark-pipeline-003` は総合資格004と入力v3を待ち、quota内の1/2/4/7 workers × CPU/CUDAの有限計測を順次実行する。warm-up4096＋3×4096、各5epochで、本番遷移0。途中状態や単独のcase合格から学習を許可しない。
- 学習／validationのケースは最初に宣言したローカル `declaration-v1` のbytesを使う。サーバ再生成には5index・27初期quaternion scalarsで最大3.47e-18の差があり、元のtraining JSONL SHA `b19b3a92834ce96ae85171b511e0d33885aa14a0ebd94178cfeadc0620331f85` を `configs/dual_coordination_cases` に固定した。両生成版と差分を保存。validationは一致した。
- 選定はN42/N43/N44と両baselineの300件ずつ、計1,500件。選定後に同じ300件×両ablationの600診断を実施する専用runnerを追加した。診断結果を再選定へ返さない。projectionはこの600件も含む。
- 開発用B-static動画は `recovered/development-video-002/video/view.mp4`、SHA `13adbb23c1c0cfd4108ad2ebd98670414ac4a389b25bf12e5908f0c4058a282d`。25.66秒、各5地点、変更0、最小間隔3.38093252m、安全hold0。最終NN動画／人手確認とは別である。
