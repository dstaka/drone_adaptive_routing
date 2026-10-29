# 2機NN協調：実装中の結果

2026-10-29 12:42 SGT追記: CPU1／2／4／7の測定反復中央値は2270.579068／1259.379690／687.722418／407.591166秒（各4096遷移＋5epoch、warm-up除外）。実測により7ワーカー・native-cpuを選定し、LearningLockを2026-10-29T01:48:48.649348Zに固定、本番学習PID38003を開始した。lock SHA-256は `eccdd3f24346c0745c2ea054ca206b8297519ab74f6baa8b8a58415ac803fa92`。CPU環境のCUDA_VISIBLE_DEVICESは空、未使用held-outは未生成。全工程の概算739,507.187秒（約8.56日）、うち学習313,030.016秒（約3.62日）、保存17,540,424,422bytes・1.25倍headroomを実測空き22,827,016,192bytesに対して判定した。評価コストは3本の開発ミッションの類似実測、動画25.66秒の実績などからの推定で、受入ミッションの実測や完了日時ではない。

同時点の確定済み学習はseed42の24／256更新・98,304遷移以上、seed43／44は未開始。外部の読み取り専用監視PID43940が60秒ごとに曲線のカウントを公開する。部分rolloutはnot recorded、terminal認証・受入は未達。監視・progress・artifact回帰67件成功（pytest0.55秒）、REDの未実装8エラーも保存。監視の起動receipt保存先不足は、既存プロセスを確認して保存先だけ修復し、再起動していない。固定source018は変更していない。初回の再開用checkpoint（seed42更新6・部分rollout476、評価対象外）を含む41files／12,390,496bytesを `run/recovered/delivery-production-start-001/` に送受信サイズ・SHA照合して回収、送信元保持。checkpoint11,228,001bytesのSHAは `8d4b9c724ac63964b97af0c98a0d2318d6d6b1b5608767cc26d9355f3b46908b`。723試行の暫定時間表は `run/timing/report-001/per-attempt-ja.md`、最終時に再集計する。

時間監査の追加注意: ローカル10試行でUTCのstart/end差とプロセスmonotonic経過の差が2秒超だった。例えば全体回帰reducer-fix-full-regression-001はUTC span4503.727859秒、wrapper monotonic851.437170秒、pytest内849.63秒。元の計測値は変更せず、journalへclock provenance注記を追記し、該当spanの実行／待機／中断の内訳はnot recordedとした。原因をホスト休止や時計補正と断定しない。暫定表001はこの発見前の分類なので、そのunionを最終active時間に使わず、最終表ではUTC spanとmonotonic値を別欄にする。全体経過は開始からのUTC差を維持する。上記8.56日はbenchmark集計の計算工程見積りであり、既に費やした実装・待機を含む完了日時の予報ではない。

2026-10-29 08:36 SGT追記: CPU2の有限計測が完了。warm-up1248.556209秒、測定3回1259.379690／1260.999822／1258.687686秒、中央値1259.379690秒、構成全体5032.897539秒。CPU1中央値2270.579068秒に対して約1.80倍。これは4096遷移＋5epochの開発計測であり、本番学習・巡回性能の合格ではない。完了記録9files／180,903bytesを `run/recovered/delivery-cpu2-benchmark-001/` に回収し、送受信サイズ・SHA-256を照合済み、元データ保持。CPU4を開始し、CPU7は待機。本番学習0、validation／held-out未実施。

2026-10-29 07:58 SGT追記: 最新指示により動画は実績のある既存手順を再利用し、CPU/GPU描画の追加実験・GL識別調査は中止。予定した描画probeとラッパーは未実行のまま外した。物理計算・NN推論/学習はCPUを維持する。CPU2の測定1回目4096遷移+5epochは1259.379690秒、warmupは1248.556209秒。CPU4/7を含む残り計測後に固定3seedへ進む。学習待機004、学習後待機002が稼働中。本番学習は未開始。

2026-10-29 07:47 SGT追記: ユーザーがGPU制限の用途を明確化。物理シミュレーション・NN推論/学習はCPU、動画のOpenGL/EGL描画にはGPUを使用可。旧動画のGL識別はnot recordedのため断定しない。今回の描画はNVIDIA EGLのvendor/rendererを別途記録する。学習前の待機処理を003へ更新し、学習プロセスのCUDA可視性は空に保つ。描画と学習は同時実行しない。本番学習は未開始。

2026-10-28T23:15UTC更新: ユーザー指示で残工程をCPU専用へ変更。GPU物理は未適格・全体速度未測定であり、加速不能とは断定しない。native物理1ワーカー+CPU NNの中央値2270.579秒、同GPU NNは2232.989秒（約1.66%短縮、warmupを除いた3回）。追加GPU開発・資格試験を停止し、候補006の331件成功と1周期36条件の局所成功、途中停止したstrict003を保存。strict003は最終レポートなしの未完了、passive32未実施。CPU1実測を保持し、2/4/7構成を新しいbenchmark-pipeline-006-cpu-onlyで実測中。本番学習0、validation/held-out未実施。CPU専用の明示記録を要求する管理reducer変更はbenchmark/campaign回帰114件成功148.60秒。既存source018の物理・学習runtimeは変更しない。

2026-10-28 20時UTCの更新：native資格はreport006の81/81・補足10項目合格を維持。本番学習・validation・held-out・最終NN動画は未着手。CPU1構成の有限計測（warm-up4096＋測定4096×3、各5epoch）は完走し、測定反復中央値2270.579秒、構成全体9101.446秒だった。その他CPU/CUDA構成は継続中。容量測定は起動順レースによる旧試行失敗を保存して再実行し、再開用最大履歴の合成probe284,394,527bytes／atomic保存1.749秒・readback1.296秒、実compact archive4,773bytesを確認した。これらは未学習の容量見積りで、学習済みモデルではない。

GPUの既存native36条件に対する30秒閉ループ比較は全件settling・完了時間差≤5%を通過し、最大差0.14782%、計測271.989秒だった。最初1秒の同一motor入力比較も別途36件通過済み。ただし全内部形状・区間安全、受動系、力・仕事、完全再開、4096＋5epochのGPUパイプライン計測は残り、GPUは未承認。候補001の97pass／2失敗はcuda表記とcuda:0の比較不整合、候補002の243pass／12失敗は同種の補助処理と設定propertyの厳密inventory不整合だった。新候補003の関連テストは実CUDA76pass／4失敗まで進み、残る4件は空Warp配列のptr=Noneを未対応とする設定・再開処理で、修正中。失敗を物理合格に読み替えない。003テストは後続native+CUDA構成の未測定warm-up中に終了し、測定3反復との重複なしを保存した。

**未完了。学習済み協調NN、routing held-out 300、物理受入127、最終動画の合格はまだない。** 既存の単機結果を2機の合格根拠に使用しない。

2026-10-28 14時UTCの訂正：旧v2ソースの81条件は全件完走したが、安全監視の予測時間がspec §4.3の「最悪資格停止時間＋0.20秒」を満たさない場合があると判明した。実測最悪3.35秒に対し、旧計算は現在の軌道だけを見て2秒などを選び得た。総合資格・benchmarkへの進行を停止し、既存結果を診断として保存した。global下限4.0秒と独立した実測比較gateを実装し、長い現在の停止軌道にはさらに延長する。新ソースの関連26件と実MuJoCo事前47件が合格し、全体回帰と81条件を別試行で再実行中。正式資格、3seed学習、held-outは未実施。以下の以前の時点の結果を、新ソースの合格に読み替えない。

2026-10-28 17時UTC：source018／inputsv3の81条件は全件実行通過し、report006で正式な81/81・補足10項目・errors0を確認した。report004の集合JSON保存失敗、report005の半刻み32試験に16を要求する集計不整合も保持した。物理ソース・結果・閾値を変更せず、別途SHAを持つ修正版集計器が宣言された刻み幅と各tickの数値監視件数を検証した。修正版を含む全体テストは1741passed／48skip／Genesis対象1deselected（849.63s）、独立レビュー重大・重要指摘0。CPU1実測benchmarkを開始、本番学習0。GPU試作の統合比較77件がCUDAで合格し、既存native36条件と同じ入力を使う最初1秒の比較は全通過（最大位置差0.000253324m、最大角度差0.0206083度）。30秒閉ループ、全内部安全／受動系資格、4096＋5epoch計測は未完了なのでGPU backendは未承認。

対象は [承認済みspec](../specs/2026-10-27-dual-drone-learned-coordination/spec.md)。専用runは `outputs/dual_payload_coordination_20261027T144754Z/`。本書は途中経過であり、最新の試行結果はrunの `progress.json` と個別receiptを参照する。

予測時間修正後の固定ソースは、全体回帰**1,735合格・48skip・1除外・失敗0／エラー0、523.09秒**、実MuJoCo事前47合格28.64秒。先行の一時コピーでは過去の回帰入力4ファイルを添付し忘れ、5失敗・13fixtureエラー、543.66秒となった。必要入力455,723bytesをSHA照合して添付し、コードを変えずに失敗18件が26.68秒で合格した後、全体も再実行した。失敗ログと環境是正receiptを保存した。旧モデル・ケースはソフトウェア回帰入力に限り、2機の物理受入には使用しない。

同一MuJoCo世界に、吊り荷を持つ2機、30 qpos、24 qvel、8モータを実装した。native MuJoCo3.15.0・float64・RK4・16内部step・100 Hz制御を維持する。担当変更、WAIT、1秒の担当ロック、検査dwell、30秒区間期限、300秒team期限、イベント順序を分離した。width128・2attention block・242行動のNNとPPO部品、baseline、選定計算、freeze前held-out拒否、進捗・時間・回収検証を実装中。

実測済みの限定的な確認：

- 同一世界・独立reset・衝突・従来モータ力との比較のnative5テストが合格。
- compact記録・状態再開を含む環境／missionのnative42テストが合格（pytest9.69秒、親プロセス計測13.17秒）。
- 10秒hoverは1000制御step、capture32.30秒、全stage gzip80,139,774 bytes。独立再構成監査は82,002状態と16,000区間、検出違反0、保存状態再構成誤差0。最小機体全体間隔9.7554369228 m、床余裕の最小値0.6346879099 m。これだけでは330秒・端点・混合荷重・制動の資格完了ではない。
- Warp32世界の1秒予備比較は位置差最大2.7395e-6 m。物理stepのみのnative1世界0.0654秒、GPU32世界1.1865秒。初期compileを含む最初の能力確認は約98.54秒。全36忠実度試験・完全4096 rolloutとPPOの測定前なのでGPU／CPUの採用は未決定。

独立レビューで、成功判定が他機の期限超過を隠す境界、checkpointの欠落時計、並列workerの古さ、失敗記録の上書き、journalの末尾破損、probe初期化失敗の記録漏れを発見した。再現テストを追加して修正した。失敗した試行をrunに保存する。

2026-10-28の利用者指示で、全内部step rawの恒久保存を不要とした。spec §11.3に反映済み。実行時監査とcompact再現記録に移行し、モデル・再実行入力・全ケース結果・動画を保存する。数値精度、安全閾値、学習seed数、評価分母は変更しない。元の全rawを遡及調査する場合は再実行が必要になる。

固定ソースコピーでの全体回帰は **1437合格・36skip・1除外・失敗0、449.89秒**。この固定コピー作成後のonline監査統合は別途native62件・46.84秒で確認した。先行試行はGenesis未導入と、実装中のソース変更を検知したハッシュ保護により2件失敗した。検査を弱めず、固定コピーで再実行した。

Genesis GPUテストは旧単機の回帰試験であり、今回の物理合格の条件ではない。利用者の指示により今後は `pytest -q -m 'not gpu'` とし、Genesisの追加転送・検証を進めない。MuJoCoのnative試験は別途継続する。

時間記録は2026-10-27T14:47:54Zから開始。それ以前は **not recorded**。親implementation工程は中断を含む経過spanであり、全時間を実行時間とみなさない。個別試行の実行・待機・失敗・再実行は `timing/events.jsonl` と `timing/work-items.json` に保存する。完了日時は未設定。最終報告では個別表、重複を除いた時間、全体経過、失敗や重複試験・環境同期不足などの振り返りを更新する。


2026-10-28T01:41:13+00:00時点の追加確認（正式資格・学習の合格ではない）：

- 開発B-static canonical pilot④は25.66秒、各5地点、100Hz最小機体全体間隔3.38093928196 m、床最小0.63499111733 mで完了した。pilot①〜③の失敗も保存した。
- 同じ129判断を再計算したnative再実行で259保存状態のqpos/qvel/rpm/ctrl誤差0、結果・報酬・巡回時間一致。全100Hz/64RK/16native区間監査を維持し、compact再現ログは7,111,813から771,097 bytesへ減った。
- 最新の関連native82テストが42.71秒で合格。別の資格試験v2・nativeカメラ18テストが3.96秒で合格。source-bound runtime/manifest回帰61合格12native未実行、terminalの異なるlearning-lock混入拒否3合格。これらは最終ソース全体回帰の代替ではない。
- 開発カメラ全2,567frame・3panel・5%borderを確認した。draft1の球marker遮蔽を静止画像で発見し保存、draft3でdetail表示を修正、日本語HUD初期/終端画像を確認した。最終動画や新しい1倍速人手確認はまだない。
- 同じnative物理のC++内部監査候補を実装中。Python監査へのfallbackと状態・checkpoint照合を検証している。未資格でありproductionには採用していない。

全native資格81条件と補助測定、CPU/GPU end-to-end4096測定、3seed学習、validation1500missions、jointfreeze、routing1500missions、physical127は未完了。学習遷移0、held-out未開封を維持する。GPU能力確認の残存ローカルSSH待ちは通信失敗として閉じ、remote実計算約100.38秒と区別した。中断・待ちの内訳を測定できないumbrella spanはnot recordedとして残す。

2026-10-28T10:24:06.808697+00:00 継続確認：C++候補はKF*(rpm*rpm)へ元実装と同じ演算順序を修正した後、全129判断・259保存状態を誤差0で再現した（物理25.66秒、wall53.1047147601秒、compact716,090bytes）。先行C++再現001の微小誤差失敗と、002のSSH通信失敗／remote実計算成功は別記録で保存・回収照合した。閾値は変更していない。新しいenv計測のnative同一性は追加検証中。

全81物理資格条件は専用ソースコピーsource-qualification-010で開始、最大7並列、失敗後未着手停止方式。開始receiptはrun/receipts/qualification-batch-001-launch.json。開発用動画001はruntime内tupleとJSON配列表現の相違で描画前に失敗し保存、値の厳密一致を維持した修正で002を開始した。最終NN動画ではない。production0・held-out未開封。

2026-10-28T10:31:57.352538+00:00 開発動画②をローカル回収済み。B-static、巡回25.66秒、各5地点、担当変更0、全機体最小間隔3.38093252006 m、床最小0.63499102302 m、安全hold0tick、検出違反0。全2,567frame・100fps・1倍速を機械検査。view.mp4 2,371,294bytes SHA-256 `13adbb23c1c0cfd4108ad2ebd98670414ac4a389b25bf12e5908f0c4058a282d`。FFV1 master318,416,357bytes SHA-256 `327d229c0a81980d3b71f199cd85d497d432c1418b020a1d8dfee5abc9963fa6`。回収先はrun/recovered/development-video-002/video/、12files324,910,849bytesの送受信照合一致。最終NN動画・正式資格・人手確認の根拠ではない。最新物理profileの再現③は129判断259状態誤差0、wall117.6721秒（資格試験と同時実行のため専有benchmarkに使わない）。追加native41合格1skip62.88秒。

2026-10-28T12:09:37.258142+00:00 資格batch001は63分45秒（3,825.2878s）で7件failed・74件未着手停止。各330秒/33,000tickを完走後、開始時に省かれた2つのlazy importで依存集合19→21となった終了検査失敗。物理値を変更せず、事前importと実1tick preflightを追加しbatch002を新規開始した。失敗を合格に読み替えない。元条件JSONとの差は5index・27個の初期姿勢quaternionの最終bit、最大3.47e-18。最初に宣言したローカルtraining JSONL47,080,006bytes/SHA b19b3a92834ce96ae85171b511e0d33885aa14a0ebd94178cfeadc0620331f85をサーバへ設定として固定し、両生成版と比較receiptを保存した。validationはbytes/SHAとも一致。

2026-10-28T12:34:12.006661+00:00 最新765file固定コピーの全体回帰：**1,732passed・48skipped・1deselected・0failed/0error、532.10秒**。Genesis GPUを利用者指示で除外、C++ SDKROOT/CPLUS_INCLUDE_PATHを明示。直前1706pass/5fail/10fixtureerrors478.61秒を保存し、学習workerのRSS初期化4件、import順序依存1件を修正、C++ヘッダー環境を是正した。最新source/src一式とstorage実測CLIを含む全テスト。native Python未導入によるskipはサーバの物理資格とは別扱い。run/receipts/final-current-full-regression-result.jsonにsnapshot・ログSHAを記録。
