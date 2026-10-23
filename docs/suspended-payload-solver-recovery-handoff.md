# 2026-10-23 終了時の引き継ぎ

ユーザーが新しいハイブリッド制御・Genesis優先specの精査と旧作業の区切りを指示したため、旧方針の追加学習を終了した。旧specの完成条件を満たしたという意味ではない。新specの実装・新規実験は今回は開始していない。

- 数値物理: MuJoCo3.15 float64 RK4、N1、control.01、substeps16で全62/motor15/reset30/初期姿勢reset30を独立再構成して通過。
- 実学習: 5pilot計1,020,000＋randomized3,000,000＋fixed3,000,000＝7,020,000有効遷移。共有settling完了0。全checkpointは不採用。
- 実development: 30万・100万・300万で各方式6条件、計36件。すべて最初のhover30秒timeout、0target完了。全36native再検査は通過。正式validation/254件は未開始で、developmentを受入へ読み替えない。
- [全rung比較とhash](evidence/suspended-solver/paired-development-curve.json)、[実学習量](evidence/suspended-solver/training-progress.json)、[配送一覧](evidence/suspended-solver/transfers.json)を保存。
- fresh最終学習、validation12/120、freeze、元draw254件、採用動画3本、今回の連続1x人手確認は未実施。held-out未開封。delta_z=0、物理・motor・閾値は維持。
- 次の判断: [新spec](../specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md)は方向性として妥当。[レビュー](suspended-payload-hybrid-genesis-review_ja.md)にcontroller-only最終採用とN1 fallbackの終端条件など、次plan前の確認点を記録。旧自動helperを再起動して追加学習しない。

終了時コードレビューではP0/P1の追加確定findingはなかったが、[未修正P2](evidence/suspended-solver/final-code-review.json)を1件保存した。旧exposure checkerはload/pose RNGとnative/action/stateを照合する一方、宣言されたcommand RNG・target順序までは独立再構成していない。実20k pilotの一時コピーをnonpilotへmetadataだけ再ラベルすると当該監査を通過した。実P6の宣言違反や物理資格偽造を示す再現ではないが、完全なcurriculum証明とは扱えない。今回全経験gateは不適格、final/adoptionは未発行であり、新specのcheckerはこの制限を引き継がないこと。

最終回収では、当作業のremote 5847fileを列挙し、4775fileを既存ローカルarchiveの同一size/SHAへ結合した。残りはcoverage記録を含む1073member・195,194,880Bの追加archiveへ回収し、全memberと永続化を検証した。両方式の全3rung native監査は終了、すべて経験不適格だった。[fixed監査](evidence/suspended-solver/long-credit-fixed-prefix-audits.json)と[全配送・coverage](evidence/suspended-solver/transfers.json)を参照する。原rawと過去の失敗を削除していない。

終了時の実process確認では、当作業30jobはすべて終了し、元PIDも終了済みだった。物理・optimizer jobと残prefix監査は0、転送用一時descriptorもrelease済み。remoteには閉じた両学習の読み取り専用RAM保管keeperが2processだけ残る。追加学習や自動再開は行わない。全bytesはローカル永続goldへ回収済みだが、元pathはこのkeeperに依存するため、停止前に同bytes regular fileへ復元する。空きdiskを確認し、復元なしでkillしない。[終了状態](evidence/suspended-solver/closure-state.json)を参照する。接続先・PID等はGit外のprivate proofにだけ保存した。サーバの契約変更・破棄はしていない。

終了時PDFは `outputs/suspended_solver_20261022_e265f60d/report_ja_solver_closure_v1.pdf`。全7ページを描画し個別に目視確認した。[PDF照合記録](evidence/suspended-solver/pdf-closure.json)にはsource・font・PDF・全page imageのhashがある。過去のv1〜v4 snapshotは上書きしていない。

以下の過去の記述は実行途中・準備時点の履歴を含む。上記終了時結果と新しい証拠を優先し、古い「実行中」を現在の指示として扱わない。

# 吊り荷solver recovery：実行記録と引き継ぎ

2026-10-22に準備し、人からの[実行指示](suspended-payload-solver-recovery-instructions_ja.md)に従って実装・実験を開始した。[仕様](../specs/2026-10-22-suspended-payload-solver-recovery/spec.md)と[計画](superpowers/plans/2026-10-22-suspended-payload-solver-recovery.md)に基づき、一般承認を再度待たずに実行を継続する。

## 実行中の状態（2026-10-23）

出力は `outputs/suspended_solver_20261022_e265f60d/`。branchは `feature/suspended-payload-solver-recovery`、起点HEADは `e39cb098a9edc47007c5d427ae0d220939075dda`。学習・受入・動画の成功はまだ成立していない。

| 段階 | 実測と状態 |
| --- | --- |
| 新実行経路 | 旧ledgerを読まない排他launcherを実装。台帳の不在・枯渇、競合・timeoutの回帰を通過。旧ledgerは保存。 |
| 小診断・参照 | Genesisの対応設定を比較し、独立RK4参照の収束と3刻み幅を検証。MuJoCo3.15.0、float64、RK4、control.01秒、base16/half32を選択。 |
| 物性修正 | 最初の候補は角減衰の質量基準が不一致だったため不採用。元のURDF質量.027kgを基準に修正。荷物質量の基準.02700047753751278kgは維持。 |
| 受動32 | 修正候補の全32を新規実行し独立検算済み。 |
| powered18・refinement12 | 全traceを新規記録し独立数値検算済み。powered最大energy–work残差5.2021e-11J、最低床距離.43793m、最大roll/pitch25.969度。6組の同一RPM刻み幅比較も通過。 |
| motor・reset | motor15、実reset30、共通runtimeの初期姿勢reset30を通過。N1の範囲。 |
| 完全資格 | 版番号付きsolver libraryのhash漏れをレビューで修正。試験前の全installed inventoryと両環境が完全一致。新schema-v2で全raw再構成に合格し、学習consumerでも再検証。旧schema receiptは学習へ使用していない。 |
| 共通runtime | 29観測・即時4motor・retargetの実状態/history維持を検証。36秒の診断hover/短往復を完了。学習済み方策の合格ではない。 |
| 学習 | 初回20,000に続き、lambda.99で200,000、bounded-spin/lambda.95で200,000を完了。角速度cost300kと位置二乗cost300kも完了し累計1,020,000有効遷移。5回ともdevelopment pilot、shared完了0。位置costの3checkpoint同条件meanもhover30秒timeout。両方式の非pilot共通3Mが各方式で完了し、完了済み学習は計7,020,000遷移。両方式shared0。全.3M/1M/3Mで各0/6、全36飛行が初回hover30秒timeout、全native照合通過。fresh最終学習は未実施。 |
| batch | N1の実物性を検証。N1/8/32/128のserial診断を完了し、有効throughputは約219/243/244/240遷移/sだった。N>1はneighbor隔離を検査したが、vector PPOの採用資格はなくN1を維持する。 |
| validation・受入 | development pilotのmean-action評価は最初のhoverを30秒以内にshared完了できず0/3。native actor全3,002記録の照合は通過。正式validation0/12、受入0/254。 |
| 動画・人手確認 | 撮影0/3。software EGL/llvmpipeの実native2視点と日本語HUD静止画を確認。lossless/視聴用codecの全frame検査を小実動画で通過。合格方策の全編動画・新規連続1x視聴はpending。 |
| 保存 | 不採用候補の全2,347項目を6,736,281,600Bのarchiveへ保存・ローカル展開。SHA-256一致後にremote重複2ディレクトリだけを整理。後続pilotの3archiveも全size/SHAを照合・展開済み。local commitは未作成。 |

pre-optimizer reviewで、初期化・sample消費・rollout連続性を確認した。実solverは `libmujoco.so.3.15.0`、SHA-256 `d3a1476dd52e220031aca40b2b00f69d1febce2ee85c57d5015f68b0cc72afa9`。試験前の全inventoryは `f9daa00f36cdd85f9cccb9fccda15971c00bdbeb6a251e250f0633429a1db519`、上流commitは `9ea3cdfcae93bf2cc4dc0e1a1627c5a39a1e06e5`。reference/flightの全inventoryとpackage RECORDも一致した。数値traceの通過だけでoptimizerを解除せず、新schemaの全再構成を必須にする。

完全資格raw manifestは `d40d67855b46d56658eab4c097b57270cbc13b7b4d0f504c099139ac3c1d2fd1`。初回pilot checkpointは `01fa85abccaaf8ac696d3f2eed378a57f28e6f904b03ed26133d7a3c7016b325`、実有効throughputは180.93遷移/秒。これは計算性能であり飛行合格ではない。mean評価後半500stepの平均機体誤差.26709m・荷物spin300.06度/秒が未達要因。別荷物を含む学習前半/後半の集計差だけで学習悪化とは断定せず、同一checkpoint/初期状態のnoise倍率0/.5/1診断はいずれもshared完了0であり、noise除去だけでは未達を解消しなかった。pilotの.15m目標差は到達半径.20m内なので、将来の3目標完了も実往復距離と分けて記録する。

追加pilotでは、lambda.99の200,000遷移で52episode中38安全失敗、bounded-spinの200,000遷移で17episode中0安全失敗だった。checkpoint20k/100k/最終の同一初期条件mean-action評価はすべてhover30秒timeoutまたは安全失敗で、受入には使用しない。bounded-spin20kは後半の位置誤差.35411m・荷物spin43.98度/秒で、spin低下だけをshared到着へ読み替えない。

初回20kの独立raw調査では、少なくとも1motorがclipされるstep68.93%、機体pitch角速度のlag1相関−.783で、action履歴との照合誤差は全20kで0だった。初期actorと20k actorを同じ訪問観測へ固定再評価してもmean出力の範囲超過は約69%。noiseだけの問題ではない。ただしclip率をPPOの学習不能率とは扱わない。角速度cost300kの最後20kではclip step率.04%、機体角速度norm平均.391rad/sまで下がった一方、位置誤差平均.665mが残った。raw action超過costは現時点で優先せず、目標から遠い場所の安定平衡にも戻りを促す位置二乗costを次の一変数仮説として宣言した。元motor clipping・権限・観測は変更しない。

未採用final freezerの独立レビューで、consumerが保存済みsuccess/exposureを実rawへ再結合する必要、正当な失敗prefixと壊れた証拠を区別する必要を確認した。final学習前に修正する。heldout freezerのrouting/camera参照についてはsnapshotとの対応と既知hashの再検査を追加した。これらは新規CPU実装の検査であり、実validationや受入を代替しない。

実行初期のbackend回帰で、最初の受動barrier前に短いpowered実行が入った。この試行は資格から除外し、受動32通過後にmotor/reset等をすべて再実行した。early-flight-v1のruntime例外と不採用減衰候補も、失敗記録として保存している。

### 最新のdevelopmentと独立consumer（2026-10-23）

P5はrewardv5（P4への位置二乗cost1項追加）、seed71、fresh4296、300,000遷移を完了した。全prefixのv17監査とmean20k/100k/300kのnative各3,002記録が通過し、飛行は全3件とも最初のhover30秒timeoutだった。最終meanの後半500STEPは機体誤差0.277550m、荷物角速度38.323479度/秒。受入には採用しない。実source archiveは `1a3e240e0116caffd0bee7fef68fb2d89475e86ee243aaf0191740f84e0f3cf6`。

次の非pilot両方式は、fresh4296・fresh optimizer、seed71、N1/T100、S3M、checkpoint.3M/1M/3M、報酬v5、gamma.999/lambda.99、curriculum_steps1.8Mで事前宣言した。初期5stageには各約300k、stage5には残り約1.5Mを与える。これは合わせたrecipeの試験であり、単独の因果比較ではない。randomized・fixedとも3Mが完了し、初期checkpoint bytesとcommon recipeの一致もローカルで確認した。randomized全29member・8,182,179,840B、fixed全29member・8,168,898,560Bは全size/SHAと永続化を確認した。[30万/100万/300万の全prefix独立raw監査](evidence/suspended-solver/long-credit-randomized-prefix-audits.json)は通過したが、経験条件は不適格。fixed全30万/100万/300万のnative監査も通過したが、経験不適格だった。全.3M/1M/3M実比較は両方式0/6で全36件が最初のhover timeout、全native再検査は通過した。source archive v19は `5dac14951fb089fd100f5474a57253dd530eaaa99995c2a606ab899f6f15908b`、333files。両方式のrung評価6条件と全stage経験が満たされるまでfinal計画は発行しない。

独立consumerは、native全getter schema・contactと安全aliasの一致、method/seedからの荷物・初期姿勢draw再生成、最終run rootの全raw・fresh初期値・exposure監査、元caseとcanonical scenarioの一致を必須にした。最終raw全体をfreeze snapshotへ含める。held-out claimは元case SHAをkeyとするhost-local guardに変更し、別freezeでも同streamの再開封を拒否する。別hostへの配送後の読取再検査は保存済み開封証拠を照合する。hostをまたぐ新実行は別途claim連携が必要で、今回の実物理評価は一つの実行hostに限定する。

P1–P5の旧learning dependency bindingは、RSL PythonとTorch native build/versionを含むがTorch optimizer Python inventoryを含まなかった。この不足を後付けで修復したとは主張しない。全pilotは非採用であり、v19の新非pilotと今後の最終学習ではTorch全Python sourceをoptimizer前とcheckpoint時にhash照合する。

MP4の全体誤差だけでは字幕局所消失を見逃すため、全29字幕・telemetry領域とforeground24pixel tileをluminance MAE4で全frame検査する。正当な日本語HUDもCRF17では超過したので、閾値を維持しCRF0/yuv420pへ変更した。実NotoJPの3frame codec診断は最大領域誤差1.919564／foreground2.686168で通過。優先度色は既存1–100 paletteへ戻し、訪問状態は独立symbol・輪郭で示す。これらは合格方策の動画ではない。

localの広いCPU回帰は1087pass/18skip/1GPU除外、新PPO設定の関連24pass、実P1の厳格prefix・改変検査8pass、nativeとderivedを同時に改変する2検査も通過した。remote v19のfocused CPUは94pass。保存helperの合成Linux CPU13検査では、sealの書込拒否、独立read offset、部分置換割込み・ready失敗・BrokenPipe・復元読取失敗時の保持、同bytes regular復元を検査した。さらに完了P1の実21file・55,983,475Bで可逆検証を行い、20k全native prefixの監査結果が切替前後で一致した。全fileは元のregular・size・SHAへ復元済み、keeperはexit0。このP1検証では資格rawと学習中のP6には適用していない。[実保管検証](evidence/suspended-solver/storage-real-pilot.json)は物理資格や新たな学習を意味しない。

単一の3M rawは専用RAM filesystemで記録する。終了processのreapと全gzip closeを確認し、ローカルgoldの全size/SHA照合後に限り、sealed/read-only RAM copyで同じ相対leafを読み続けられる保存経路を用意した。keeperは永続goldの代わりではなく、終了時は同bytes regular復元が必要である。完全資格rawはregularのまま保持し、移動しない。実行中PPOのbound sourceと物理/runtime sourceは保存helper追加の前後でも一致した。

P4/P5の同じ30万遷移checkpoint・同一pilot条件を全100Hz記録から比較した[図](evidence/suspended-solver/pilot-matched-300k.png)では、位置復帰と荷物静止の両方が未達だった。P5荷物の主振動数は約4.207Hzで、固定支点の微小角rod/cubeモデルの高速固有振動数4.20746Hzと整合した。これは失敗診断であり、物理資格・制御可能性の証明ではない。[診断JSON](evidence/suspended-solver/pilot-dynamics.json)はsample数と完全区間数を区別する。

最新progress PDF v4は4.02M遷移完了・同条件fixed実学習中・randomized全prefix監査通過と、実100Hz軌道のspin診断図を含むsnapshotで、全7ページを描画・個別目視した。日本語・図・footerに欠けや重なりはない。[PDF照合記録](evidence/suspended-solver/pdf-progress-v4.json)を保存し、v1/v2/v3は履歴として残した。最終結果後にPDFを改めて整合させる。

下記は準備時点の判断・旧実験の履歴であり、上表の実行結果と区別する。旧model2000は初期値・再開に使用していない。

## 今回のユーザー変更と適用順

ユーザーは「次回以降はGPU時間制限なし、使用時間tracking不要」と明示した。旧spec/指示/実行helperにある8時間枠・残量・phase配分・ledger更新・後段が枠に収まるかによる停止は、今後のこの作業に適用しない。古いledgerの数字を新しい残量へ読み替えず、台帳は歴史として保存する。新規GPU使用量集計は作らない。

ユーザーが今回求めたのは、直前の推奨案に沿う次回実装の入力整備である。推奨したエンジン変更の許容とsample量のdevelopment決定を新specに具体化した。新セッションでは計画提示後に実行を続け、これらの通常判断で改めて一般承認を求めない。

一方、制御器＋学習補正への変更、90～170度試験の除外、leg/missionの延長、受入閾値の緩和は推奨案の条件付き代替案であり、このpackageでは採用していない。必要になったら、その変更だけ具体的に提案する。

優先順は「人の新しい実行指示→新spec→旧specの非矛盾部分」。文書準備を実行指示と誤認しない。既存の通常missionのwall600秒・1GPU job・deadlock検出はGPU総時間制限とは異なり残る。新規レンタル・契約変更・サーバ破棄は許可していない。

## 起点と保護

- 想定HEAD：`e39cb098a9edc47007c5d427ae0d220939075dda`。実際のHEAD/branch/history/index/untrackedを確認し、進んでいてもresetしない。
- この準備時点のbranch：`feature/suspended-payload-flight-recovery`。次回は実状態を優先する。
- 新spec、本handoff、実行指示、実装計画、spec索引は未commitの可能性がある。保護し、次回の実装・テストと同じ新規local commitへ含める。
- `adaptive_drone_patrol_20261013/`、個人PNG、`.DS_Store`等を混ぜない。旧spec/証拠/失敗checkpointを上書きしない。旧model2000を再開・初期化に使わない。
- 接続先は次回、別の非公開メッセージで受け取る。保存済みprivateファイルに旧接続先があっても現在有効とは仮定しない。
- 準備時点でルートの `spec.md` は存在せず、既存索引にはその歴史参照が残っている。歴史コピー配下には `spec.md` があるが、ルートの現行specとしてコピー/改変しない。新しい実装は新specと明示した先行spec群に従う。

## 既知の結果と限界

[前回の実測handoff](suspended-payload-flight-recovery-handoff.md)と `docs/evidence/suspended-recovery/` が主な入口。前回出力は `outputs/suspended_recovery_20261022_112649_abce93/`。

| 事実 | 次回の解釈 |
| --- | --- |
| V10代表rawを新GPUでhashまで同一再現、正energy最大530.703% | 元の異常を再現できた。修理成功ではない。 |
| 独立平面の局所積分で増分差約8.92e-8J | 当該traceの離散化を強く支持。全3Dの証明ではない。 |
| 128/256substepの完全受動3traceは正増加5%内 | 3/32、全体3/62のみ。170度約13%損失が残る。 |
| 90度の30秒比較に最大約0.642m/180度の差 | 短時間収束とenergy/workを追加診断。受動比較にpowered閾値を流用しない。 |
| CPU901passed、Linux3.12でも901passed | CPU実装の検証。新motor0/15、reset0/30、学習0、受入0/254、動画0/3。 |

旧V10の失敗は、大角度14/16 energy増加、床8件、powered dt pair0/6。前回の受動3traceだけでこれらを解消したことにはならない。旧時間不足による停止理由は今後は撤廃されたが、物理未資格は引き続きoptimizerを止める。

新specは正energy増加だけでなく、独立再構成した物理仕事を差し引いた受動energy収支の絶対誤差5%を明示した。前回の20%損失緊急停止値未満であることを資格へ読み替えない。現在の約13%損失を隠して採用しない。

## 必ず取り違えないsourceと成果物

| 対象 | SHA-256 |
| --- | --- |
| 旧V10実source archive | `ca27fbdff020ef6a890ef98aeefb8af360803029ddea861c527ed4abae128a3f` |
| 旧V10完全receipt | `2ca4f9b3b0d52bafeca38c8fc723d160af21b228ef48e1d74260e0c7d1cf49ce` |
| 前回baseline再現sample | `8601f5f5d8b6b4e6694b9ed2d1631c3ddfa81fe66cfb9308d0a16bcc4066b8c4` |
| 前回candidate-s128-source-v2.tar.gz | `c8dddaa2a807e743beaaee6f0afedf9e5d867ff0468832af89ef5271dbc9fcbf` |
| 前回candidate物理署名 | `93794e24c9b1b688169bad71ba99bda73d27ca0d5de97b907c3b6ef6a21c47b7` |
| 前回final-delivery.tar.gz（7270159 bytes、83payload members） | `3910f05507ac572cb3febd76d232d391ef310ef05552388383d85d867ad5e3b2` |

最終CPU sourceにはcandidate実行後のguard/runtime/batch等の変更がある。[source-bindings](evidence/suspended-recovery/source-bindings.json)を確認し、新sourceに旧GPU PASSを差し替えない。各artifactの実在・size/hashは次回も確認。大きいraw/modelがGitにないことは正常だが、不足なら回収する。

親model4296/cfg/routing seed44のsize/hashは新specに再掲。実パスは[元payload spec](../specs/2026-10-17-suspended-payload-flight/spec.md)と[分散配置handoff](dispersed-destination-layout-handoff.md)から辿る。転送bundleには `third_party/genesis_drone/hover_env.py`、`hover_eval.py`、`hover_train.py` も含める。前回はこの3つの漏れによりLinux検証初回900pass/1fail、修正bundleで901passとなった。

## 再利用する部品と未完成部分

| 現在の入口 | 再利用できるもの／次回の仕事 |
| --- | --- |
| `payload_recovery_planar.py` / `diagnose_payload_planar.py` | 独立平面モデルと局所再生。高次参照・3刻み幅・実substep比較を追加。 |
| `payload_recovery_qualification.py` / `payload_recovery_passive_binding.py` | 62manifest、32passiveのraw/source/初期状態/force再構成。新数値品質とbackendを適合。 |
| `scripts/probe_payload_ball.py` | powered前barrier。別backend/仕事観測/全資格は新宣言へ統合。 |
| `payload_recovery_runtime.py` | current-action、reset/retarget、raw、例外時quarantine。admissionは意図的hard blockであり完成していない。 |
| `payload_recovery_mission.py` | shared100区間、全10地点、イベント/実位置routing説明。CPU doublesだけでなく実backendで検証。 |
| `payload_recovery_batch.py` | 3寸法bucket・logical episode・actual getter・mask。実geometry/隣接隔離/reset未測定、128上限をhalf256へ対応。 |
| `payload_recovery_sampling.py` | S=N*T*K、RNG、stage経験。旧BudgetForecastを新実行の許可条件に使わない。 |
| `scripts/train_payload_retry.py` / `payload_retry_training.py` | 単一env・旧物理bindingが残る。新資格/共通runtimeへ実装し直す対象。 |
| validation/acceptance/capture | 完全な新物理共通入口は未統合。新規実装が必要。 |

優先する順序は、小診断で候補選択→受動32→診断controllerでhover/往復→powered/motor/reset完全資格→実N1で小規模学習飛行→必要なbatchと共通sample計画→最終比較である。CPUコードやテスト数を増やすだけで実飛行確認を後回しにしない。

## 次回が通常判断で進められること

- 対応する積分器/精度/dt、根拠のある対象限定engine patch、必要な依存更新、別backend/reference。旧lockを残し新lock/source/patch/build hashを固定し、同じ実物理を再資格する。
- developmentの.3M/1M/3M checkpointと、その先の共通sample増量。新しいrungは実行前に宣言する。GPU残量との照合は不要。
- development内の共通reward/curriculum/PPO調整。両最終方策のS/N/T/seed7/architecture/初期tensor等を学習前に同一freeze。
- 条件が成立した場合だけ、SとA-J全点の共通高度候補0～1m。両最終学習より前に最小根拠値を固定。
- N1継続。並列化は実験を速くする手段であり、それ自体が飛行成功の前提ではない。

pilotはdevelopment内で継続可能だが、両最終学習の初期値にしない。両方とも正しいmodel4296移植とfresh optimizerから始める。29観測・4motorの意味を維持し、actorに質量/サイズ/IDを渡さない。学習量を変えるなら両最終学習を同じ新計画でやり直し、片側だけ追加しない。

## 次回の出力と報告

新しい `outputs/suspended_solver_<date>_<unique>/` と `docs/evidence/suspended-solver/` を使う。候補・失敗・source/物性/仕事・全gate・学習sample・比較・動画・人手確認・転送hashを保存する。GPU使用秒/残量の報告は不要。missionの時刻や1x判定は維持する。

新spec/計画/このhandoff/実行指示/索引を実装・テスト・必要文書・compact evidenceと一緒に1件の新local commitへ含める。Author/Committerは `dstaka <22057509+dstaka@users.noreply.github.com>`。stageにprivate情報・機械固有絶対path・model/video/巨大log・個人ファイルを混ぜない。日本語の全gate報告と、新specリンク付き英文PR descriptionを出す。未達・人手pendingを完了へ読み替えない。

## 準備資料の検査範囲

新しい4文書のローカルリンク、spec索引の追加リンク、公開情報の秘密/絶対path混入、Markdown構造と空白を検査する。前回compact evidenceが指すsource/転送/CPU/PDF関連24件の実artifactは準備時にsize/hash一致を確認した。親source3件も実在する。これは次回の実物理資格ではない。既存索引のルートspec欠落は上記のとおり記録し、旧証拠を生成して補っていない。

### 実保管・配送検証の追加記録

完了P1のsealed transfer archive（56,033,280B、SHA-256 `efb22e2346a61ee0a73619380879a76ed90bed9db01e6e1a10fc398c650b2ecc`）を実際に転送し、全22memberと展開後の全size/SHAを照合した。元rawはこの配送検証中ずっとregularのまま保持した。追加レビューで永続化要求の不足を指摘し、archive本体のfsync、Darwin full flush、親directoryのfsync、gold/receiptのatomicな書込みと同じflushを追加した。実P1で陽性を確認し、file/full/directory flushの注入失敗3件はすべてgold/receiptを発行せず停止した。

この永続化済みproofと全member照合の後にだけ、完了学習rawのsealed/read-only caretakerへの切替を許す。新private coordinatorは、randomized終了・配送・切替、宣言済みと同じfixed3M、両方式の全3rung監査・development評価を直列に進めるため起動した。最終採用やheld-outはこのcoordinatorの対象ではない。起動済みと実際の各段階の完了を区別し、物理実行は常に同時1jobとする。

### 受入・撮影checker v20と共通学習の記録

[独立レビュー記録](evidence/suspended-solver/checker-review-v20.json)の修正は、公開撮影APIが全254件をfresh再検査し、randomizedの元trial・case・model・native・元artifact集合へ動画を結合する。描画前後のsnapshot自体も元のaccepted mapと照合し、次の動画のXMLが途中で変わった場合を拒否する。全成功イベントをstep/time/pose/順序/目標/完了順と実poseの費用比較まで独立再構成し、同じ誤イベントで字幕を再描画するだけでは合格にしない。実サーバ48件、このchecker変更直後のlocal全回帰1083件を通過した。変更した6ファイルはv19 PPOの全bound source・完全資格のphysical/runtimeと非重複であることを配置前後に照合し、旧6sourceもarchiveへ残した。

最初の共通randomized3Mは無効0、9631閉episode中9579安全失敗、shared0だった。stage4では記録上7008閉episodeすべて失敗、最長50stepで第2目標への試行も0だった。全300万のnative独立監査は通過し、stage4の連続target区間の最大は53stepだった。これはstage4で開始して閉じたepisodeの最長50stepとは集計単位が異なる。全stage経験と連続10地点経験は不適格であり、sample数を技能へ読み替えない。追加payload列は実checkpointの[合成静的Jacobian診断](evidence/suspended-solver/long-credit-actor-sensitivity.json)で非ゼロへ学習されていたが、減衰の符号・位相、閉ループ安定性やmean飛行合格は未判定である。criticとactorは別々にgradient clippingされるため、critic大gradientが共通norm経由でactorを抑える仮説は採用しない。

### 次sourceだけの終了境界修正

[実rawの独立診断](evidence/suspended-solver/long-credit-feedback-diagnosis.json)は、stage0の位置保持と未達のfast transverse angular settlingを区別した。末尾100秒の1episodeでspin平均95.833度/秒、直接spin costのcap分率99.36%をrootでも再計算した。stage4の抽出107完全episodeは全件、最初のcornerで支持推力不足のまま40～49stepで荷物床条件に達した。忘却や一貫したanti-dampingは未確定。次にcap形状だけを比較する仮説は保存したが未採用・未実装であり、現paired3Mと全rung評価が先である。[PPO接続の独立照合](evidence/suspended-solver/ppo-integration-review.json)では、先頭20kの報酬12項、実移植tensor、Adam13状態と更新数が一致し、今回shared0を説明する接続ミスは再現されなかった。

独立CPU調査で、10個目のshared完了とepisode12000step期限が同時の場合、真の成功終了にもPPOのtimeout bootstrapが加わる境界不具合を再現した。[局所修正記録](evidence/suspended-solver/terminal-timeout-boundary.json)のとおり、先にspecを更新し、旧コードの否定検査1fail/3passを確認してから、真の終了をbootstrap対象から除外した。関連40検査と次sourceの全CPU回帰1087件、独立レビューは通過した。端末rawのtimeoutフラグは「期限にも到達した」という事実として保持する。今回のrandomized3Mはshared0のため発火0であり、飛行未達の原因とは主張しない。これは次sourceだけの修正で、完了した両学習と宣言済みv19比較は変更しない。比較終了後に新source bindingへ反映する。

次sourceは元v19とは別のコピーへ全360fileのsize/SHAを照合して配置し、同一のpinned環境でLinux CPU40件を通過した。元v19の全bound sourceと静的な資格済み物理bindingは配置・検査の前後で一致した。これは新たな物理実行・方策学習・recipe採用ではない。CPU証拠と実source archiveを含むmetadata配送v7の全7memberをローカルでsize/SHA照合・展開した。今後の変更も別sourceで束縛し、元のv19を監査終了前に書き換えない。

[割引報酬の独立算術診断](evidence/suspended-solver/discounted-reward-diagnosis.json)では、stage4の107失敗episodeすべてで、実returnが「初期機体位置・最初の遠方目標を固定し、水平姿勢・垂直荷物・速度0で留まる」構成状態の比較値を上回った。rootでも平均実return−201.47161、比較値−210.58703を再計算した。この比較は実800step retargetを含む停滞方策のreturnではなく、意図・最適性・実PPO advantage・係数提案を意味しない。spin項だけを同じ軌道上でuncappedに置換しても107件すべてで比較値を上回り、fast-mode capの問題とは区別した。未採用の仮説として保存し、現比較を先に完了する。
