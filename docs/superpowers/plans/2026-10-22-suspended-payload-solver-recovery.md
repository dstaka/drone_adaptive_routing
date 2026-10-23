# Suspended-Payload Solver Recovery Implementation Plan

> **Execution closed:** 2026-10-23のユーザー指示により旧方針の追加実験を終了する。数値資格は達成、飛行・最終受入は未達。新specの実装を始めず、全データ回収・終了時handoff・1件のlocal commitで区切る。未完checkboxは歴史的な未達として残す。

**Goal:** 同じ吊り荷物理モデルを正しく計算できるbackendを選び、実飛行・公平な学習・254評価・全編動画まで完了する。

**Architecture:** 診断・数値参照・backend実装と、独立した資格判定を分ける。既存recovery runtime/mission/sample部品を利用し、選んだ物理・観測・action契約を全経路で共有する。計算量より先に、実backendで小さい飛行課題が成立することを確認する。

**Tech Stack:** Python 3.12、最初は既存lockのGenesis 1.4.3、必要に応じて宣言済みpatch/更新/backendと新lock、PyTorch/RSL PPO、独立CPU数値参照、既存routing・動画検証。

**Spec:** [新spec](../../../specs/2026-10-22-suspended-payload-solver-recovery/spec.md)。specがこの計画より優先する。候補に応じた実装詳細は次回セッションで具体化し、通常の設定・エンジン変更について承認待ちしない。

## 実行状況

| 段階 | 実結果 |
| --- | --- |
| 数値修理・資格 | MuJoCo3.15/float64/RK4/N1/control.01/substeps16を採用。受動32後にpowered18/refinement12、motor15、実reset30、初期姿勢reset30を完了し独立raw検査通過。早期barrier前の短powered試行は除外して再実行。 |
| 学習飛行 | 小pilot5回計1.02Mと最初の非pilot共通pairが各3M完了、計7.02M、両方式shared0。randomizedの全.3M/1M/3M native prefix照合は通過したが経験条件は未達。 |
| 共通比較 | 両方式3M終了と全raw配送・全member照合・永続化は完了。全.3M/1M/3Mで両方式各0/6、全36native照合通過。元v19は別コピーの次sourceとは分け、この監査・比較終了まで変更しない。 |
| 次source | 真のgoal終了と期限の同時発生時のPPO bootstrapだけlocal修正。旧raw schemaは維持。局所40検査、広いCPU1087検査と独立レビュー通過。現サーバv19へは未配置。 |
| 最終段階 | final選択・fresh最終学習・validation12/120・freeze・254件・採用動画3本・新規人手確認は未達のまま終了。全データ回収と終了時文書・PDF検証を完了し、指定identityの1件のlocal commitへ保存する。 |

## Global Constraints

- 起点想定e39cb098a9edc47007c5d427ae0d220939075dda、ルートで作業。先行変更・準備文書・履歴・個人ファイルを保護。
- GPU累積上限なし、使用時間trackingなし、旧ledger更新なし。1GPU jobとdeadlock保護は維持。
- 実物理・0.30mリンク・3寸法/独立質量0.10～0.30・反作用・元motor権限を維持。dt_control=.01。
- 通常判断で設定/精度/積分器/対象を絞ったengine patch/版更新/別backendを検証可能。実物理や受入閾値の緩和は別承認。
- 全32受動→powered、全62＋motor15＋実reset30→optimizer。新しい受動energy収支5%条件も独立再構成。
- 高度候補0/.25/.50/.75/1m、developmentで根拠がある最小共通値だけを両最終学習前に固定。
- sample ladder .3M/1M/3Mはdevelopmentの出発点。両最終方策は共通S/N/T/seed7・同一model4296移植・fresh optimizer。
- validation12/120→freeze→両方策127件ずつ、合計254。randomized9/9・18/18・95/100以上。
- 全10地点shared settling1秒、floor>.10m、roll/pitch<=60度、leg30/mission120/wall600秒。raw/HUD3本と新規人手1x確認。
- 説明・字幕は日本語、コード/comment/docstringは英語。新規local commit1件、指定identity。push/PR/公開/提出なし。

## Review Focus

- 旧budget helperが残り秒・ledger不在で新経路を止めないか。Task1で負例を検査する。
- 別backend・patch・精度の実source/物性/仕事が自己申告や旧PASSに置き換わらないか。Task2/3の束縛破壊試験。
- reset/step例外、有限異常、inactive slotからrawや遷移が失われないか。Task4/5で実getter・mask・quarantineを検査する。
- pilotやheld-outによる最終比較の汚染がないか。Task6/7で出自・freeze・各総sampleを照合する。
- CPU doubles・単なるhover・メタデータ1xを本番の実遷移/完走/動画へ読み替えないか。Task4/7/8で実データを再構成する。

## Task 1: 新しい実行経路と入力保護

**Files:** 新規 `src/adaptive_drone_patrol/payload_solver_jobs.py`、`tests/test_payload_solver_jobs.py`。既存参照は `src/adaptive_drone_patrol/evidence.py`、`scripts/run_payload_retry_probes.py`、`payload_recovery_sampling.py`。新規出力は `outputs/suspended_solver_<date>_<unique>/`。

**Interfaces:** `run_job(command, *, output, timeout_s)` は排他・進捗・終了コード・失敗artifactを管理する。GPU総時間・残量・ledger引数を持たない。timeoutはjob停止用で、missionの600秒とは分離する。

- [x] git HEAD/branch/history/index/untrackedを読み、準備ファイルを保護。入力manifestと必要artifactのsize/hashを照合。
- [x] ledgerなし・旧ledger枯渇でも実行可能、旧ledger bytes不変、競合job拒否、timeout時raw保持のテストを先に失敗させる。
- [x] 新launcherを実装しGREENを確認。旧helperを変える場合も旧履歴の意味を変更せず、新経路は残時間計算を呼ばない。
- [x] GPUが必要な時点でのみ非公開の現接続先を使い、新workspace/隔離環境を用意。初期lock照合と転送hashを保存。

## Task 2: 小さい数値診断と修理候補選択

**Files:** `payload_recovery_planar.py` を再利用し、新規 `src/adaptive_drone_patrol/payload_solver_reference.py`、`scripts/diagnose_payload_solver.py`、`tests/test_payload_solver_reference.py`。候補宣言は出力配下 `candidates/<id>/declaration.json`。必要なpatchは `patches/payload-solver/<id>.patch`、新lockは `requirements/payload-solver-flight-linux.lock.txt` / `requirements/payload-reference-linux.lock.txt`。

**Interfaces:** `compare_candidate(declaration, artifact_locations)` は実状態・source・物性を読み、短時間収束/energy/work/拘束と失敗理由を返す。宣言と実値が異なる場合は拒否。候補比較receiptだけでtrainingを許可しない。

- [x] 保存済みのV10・128/256 source/rawを検証し、今回sourceとの差を記録。既存の独立平面診断を再現。
- [x] 独立高次参照の既知解/刻み幅収束、誤ったCOM/frame/慣性、符号の違うwork、source差替えをテスト。
- [x] 90度代表・170度nominalの小診断を宣言し、対応する積分器/精度/dtを比較。必要な候補だけ実行する。
- [x] 改善しなければ、数値機構に対応するpatch/更新/別backendを宣言して検証。参照自身の収束、質量/慣性/motor/反作用、実installed sourceを確認。
- [x] 3刻み幅で短時間収束と長時間energy/workを検査。13%損失を隠したまま次段へ送らない。全失敗を保存し採用候補を固定。

## Task 3: 受動barrier、実飛行診断、完全資格

**Files:** `payload_recovery_qualification.py`、`payload_recovery_passive_binding.py`、`scripts/probe_payload_ball.py`を再利用/拡張。新規 `scripts/qualify_payload_solver.py`、`tests/test_payload_solver_qualification.py`、選択backend adapter。新規 `configs/payload_solver_probe_manifest.json`。motor/resetは既存auditを新sourceへ明示適用。

**Interfaces:** `verify_solver_qualification(receipt, actual_binding, locations)` は全raw/actual settings/engine/workを再構成し、完全資格だけを返す。`actual_binding`にはbackend、source/lock/build、dt/精度、asset/物性、batch、観測/actionが入る。

- [x] 短trace、重複case、未観測force、欠損substep、旧署名、損失超過でadmissionが失敗するテストを先に追加。
- [x] 選択sourceの受動全32を実行、独立検査。失敗ならTask2へ戻りpowered/optimizerを開始しない。
- [x] 受動通過後に診断controllerによるhover・短往復・retargetを実測。floor/tilt/反作用/入力飽和を保存し、制御不良と積分不良を分離。
- [x] 同一RPM replayとpowered18/refine12、motor15、30回以上の実resetを完了。N=1では他logical環境の隔離を達成したと主張せず、batch選択時に隣接環境付きpartial resetを追加。
- [x] 仕事上界/参照の前提を新積分器で導出・測定。最終sourceと実行sourceを一致させ、独立receiptを作成。

## Task 4: 共通runtimeと最初の学習飛行

**Files:** `payload_recovery_runtime.py`、`payload_recovery_mission.py`、`payload_retry_contract.py`、`payload_retry_checkpoint.py`、新規 `scripts/train_payload_recovery.py`、`tests/test_payload_recovery_training.py`。既存runtime/missionテストを拡張。

**Interfaces:** 既存 `RecoveryPayloadRuntime` のreset/retarget/step/actual rawを選択backendへ適合。`require_recovery_runtime_admission()` のhard blockはTask3の実再構成を実装して置換し、常にtrue等の解除をしない。

- [x] 異なるbackend/source/物性/観測checkpointをinference前に拒否するテストと、reset/retarget/actionの実時系列テストを追加。
- [x] 実N=1 backendでmodel4296の移植tensor・出力・即時action・clip/historyを照合。親configも検査。
- [x] runtime統合が資格対象source/settings/action経路を変更した場合はTask3へ戻り、実行sourceと一致する資格を取得してからoptimizerへ進む。
- [x] 資格済みN=1でdevelopment seed/cases/Sを宣言し、小規模な学習hover・往復を実行。実motionとshared settlingを独立評価。
- [ ] 失敗なら実軌道・reward成分・行動飽和・観測scaleから修正案を選び、共通reward/curriculum/PPOをdevelopmentで検証。未資格physicsを学習で補わない。
- [x] training/validation/mission/captureが同じcoreを使う入口を用意。以降の全S学習より前に実retarget連続性を確認。

## Task 5: 必要なbatchだけを選択

**Files:** `payload_recovery_batch.py`、`payload_recovery_sampling.py`、既存batch/sampleテスト、新規 `scripts/benchmark_payload_solver.py`。

**Interfaces:** 選択Nのbindingと、実有効transition/s・memory・各処理区間のprofile。GPU累計/残量forecastは作らない。source/settingsが変わればTask3の選択構成資格を取り直す。

- [ ] base128/half256等の設定上限、実geometry、独立episode抽選、slot/shape、例外時rawのテストを追加。
- [ ] 対応するN1/8/32/128を計測。N1との差、実geometry/慣性、30partial reset、隣接環境不変、invalid rollout隔離を確認。
- [x] 正しさ・memory・連続episode経験を満たすNを選び、選択Nで完全資格。PPO計測は資格成立後。
- [x] 正しいbatchが得られない場合は資格済みN1でTask6へ進む。並列化自体を完成条件にしない。

N>1の実batchは未資格で採用していない。記録込み実有効throughputを測った完全資格済みN1を継続する。高度はdelta_z=0を維持し、現在の推力不足による落下を共通高度変更の根拠へ読み替えない。

## Task 6: developmentによるsample選択と公平な最終学習

**Files:** `payload_recovery_sampling.py`、新training CLI、trainingテスト。出力は `development/`、`training-plan.json`、`final/randomized/`、`final/fixed/` を分離。

**Interfaces:** `training-plan.json` はS/N/T/K、seed7、移植tensor hash、architecture/PPO/reward/curriculum、stage経験、checkpoint選択規則、layout/physical bindingを両方策共通に固定。

- [x] pilot再開checkpoint混入、sample過少計上、方法別丸め、無効slot消費、rollout境界reset、stage経験不足を拒否するテストを追加。
- [ ] development用ケース/seed/評価規則を事前固定し、両方式の.3M/1M/3Mチェックポイントを比較。必要な次rungは実行前に宣言し、共通に増やす。
- [x] 物理成立後に必要な場合だけ高度候補を診断し、base hash/共通transformとrouting・観測・HUD整合を検証。
- [ ] development結果でS等を凍結。両最終学習を同じmodel4296移植とfresh optimizerから実行。pilotやmodel2000を初期値にしない。
- [ ] 両方策の実有効sample、optimizer更新、全stage/連続target経験とcheckpoint由来を独立集計。追加GPU時間の許可や残量計算は不要。

## Task 7: validation・freeze・全254受入

**Files:** 新規 `scripts/validate_payload_recovery.py`、`scripts/evaluate_payload_recovery.py`、`scripts/check_payload_recovery_trials.py`、`tests/test_payload_recovery_evaluation.py`。既存cases・mission・routing説明/統計部品を利用。

**Interfaces:** validation receiptは12条件/120実target遷移のrawを束縛。freezeは両model/source/engine/layout/cases等を束縛し、254case結果はfreeze hashと対応。

- [x] 静止sampleのtargetラベル変更、100瞬間sample、retarget reset、stale freeze、case省略/重複、評価後再抽選を拒否するテストを追加。
- [ ] 両方策のvalidation12/120を実行し独立再構成。モデル選択は事前規則に従い、held-outを開く前にfreeze。
- [ ] 両方策それぞれ9/18/100、合計254を新規実行。ケース単位600秒wallを維持し、GPU総時間とは別に判定。
- [ ] failure/分母/subgroup/Wilson95%と、実位置からのQ*/J*/suffix/反実仮想/コストを保存。固定方策が良くてもそのまま報告。

## Task 8: 全編動画・文書・最終保存

**Files:** 新規 `scripts/capture_payload_recovery.py`、既存media/HUD/検査部品と関連テスト、README、`docs/report_ja.md`、新handoff、`docs/evidence/suspended-solver/`、実行結果の英文PR description。

**Interfaces:** 採用randomized・seed1・nominalの3完全raw/HUDと、全frame検証receipt、final hash、人手確認記録。全gate未達も明記する。

- [ ] 代表実frameで日本語HUDのgeometry/文字/圧縮を確認し、合格した方策の全10地点shared settlingまで3scenarioを撮影。
- [ ] 全frame decode・PTS・元画素・字幕/telemetryを検査し、具体的イベント前～次settling区間を指定して新規1x連続視聴を依頼。
- [ ] 待機中にCPU検証・資料・PDFを完了。全PDFページを描画/目視。人手未確認はpending。
- [ ] 全外部成果物をローカルへ転送しsize/hash照合。新spec/準備資料/索引を実装・テスト・必要文書・compact evidenceと同じ1件の新local commitへ含める。
- [ ] stage allowlistと秘密/絶対path/個人ファイル/巨大成果物の混入を検査。指定Author/Committerでcommit、実IDと全gateを日本語報告。英文PR descriptionは新specへリンク。GPU使用時間集計を要求しない。

## 実行中の判断

物理や実データが不足すれば依存段階を止め、次の根拠ある候補を調べる。時間がかかること、N1であること、30万遷移で足りないことは終了理由にしない。未承認の変更が必要になった場合だけ、失敗証拠と具体的な修正specを提示する。サーバに接続できない場合も、独立CPU統合・テスト・文書を進める。
