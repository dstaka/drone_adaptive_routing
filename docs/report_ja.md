# 吊り荷飛行の物理回復診断

2026年10月22日 / 運用域資格は未完 / 学習・254受入・撮影は未実行

## 1. 今回の結論と継承条件

起点e3c145aから、specs/2026-10-22-suspended-payload-flight-recovery/spec.mdに基づいて実装した。現在のHEADと未commitの新spec・公開指示・handoff・索引を保護し、古いrevisionへのresetやamendは行っていない。実装先はルートで、歴史コピーと個人ファイルは対象外である。

旧失敗の新規再現と積分誤差の原因診断は進んだが、吊り荷の物理修理・運用資格・学習成功を達成したとは報告しない。新候補は3/62条件の受動screenだけであり、完全な32受動barrier、powered、motor、reset、選択batchの資格は未達である。

0.30mの圧縮も伝える固定リンク、機体下面(0,0,-0.020)、荷物上面attachment、実cube質量・慣性・衝突、受動関節と機体への反作用を維持した。荷物サイズ0.030/0.045/0.060m、質量比0.10～0.30は変更しない。直接荷物制御、姿勢投影、強い追加damping、motor権限の拡大は使っていない。

配置は全精度のSとA-Jを維持し、delta_z=0である。高度を上げるための前提となる物理資格が未成立なので、高度候補の開発sweepは実行していない。床>0.10m、傾斜60度、leg30秒・mission120秒・wall600秒、全10地点で共有1秒settlingの条件も不変である。

両方策の学習は各0/300000総遷移、validationは0/12条件・0/120実遷移、受入は0/254件、新規全編raw/HUD動画は0/3本である。成功率・Wilson区間・比較優位性・人手承認は存在しない。

<!-- pagebreak -->

## 2. 旧V10の再構成と独立な原因診断

V10のsource archive、完全receipt、約818MBのdelivery archiveを再hashした。全1631payload member、55traceのraw/sample hash、357176sampleを照合し、欠落・不一致はなかった。元の53command path表記差も時間・job同一性から照合し、両累積台帳を一致させた。

新規GPUで、90度・一辺0.060m・質量比0.10・motor0の30秒caseを再現した。3001sampleのSHA-256は旧V10と完全一致した。初期energy0.008888015284Jに対して最大0.056057001217J、増加530.703%、最初の5%超過は1.07秒だった。正常process終了は記録保存の成功であり、物理PASSではない。

worldXZ平面の剛体rodと独立回転cubeについて、実COM質量・慣性から2自由度Lagrangianを別実装した。角度はrod/cubeの絶対下向き軸であり、下側jointの相対角ではない。COM位置・速度の最大再構成差は2.124e-7m・2.845e-6m/sだった。

各実測0.01秒区間を初期値として2回のvelocity-first dt0.005秒更新を計算した。実energy増分0.046955007284Jに対し、局所再生の増分合計0.046954918107Jで、差は8.92e-8J、区間最大誤差は2.997e-8Jだった。この平面traceでは、保存的な意図モデルの積分離散化が観測増分をほぼ説明する。

これは3D全般のengine欠陥や、修理済みという証明ではない。0.01秒の速度差分はsubstep平均なので、その残差をそのまま未知の瞬間外力へ読み替えない。再現可能なCPU入口はscripts/diagnose_payload_planar.py、全計算と入力hashはcompact evidenceから参照する。

<!-- pagebreak -->

## 3. sourceに基づく細かいdtの実診断

pinned Genesis1.4.3の実sourceを確認した。standalone自由剛体用midpointは、DOF子孫やconnect/weld拘束を持つ物体を除外する。別CONNECT表現へ変えるだけでその積分経路を使えるという根拠はない。engine実装bytes・依存版は変更していない。

CPUの30秒縮約モデルでは8/32substepsに不合格条件が残った。そこで128substeps（物理dt0.000078125秒）を新署名・事前宣言・有限timeoutで診断した。3秒の小診断後、同sourceで30秒の90度・170度、半dt256substepsの90度を測定した。全sample・実物性・初期姿勢・source・engine264ファイル・実ゼロ外力場を独立に照合した。

90度・0.060m/0.10・128：正energy最大0.7403%、最小energyは初期の95.1066%。170度・nominal・128：最大1.1683%、最小86.9668%。90度・同荷物・256：最大0.6901%、最小97.9775%。いずれも3001sample/30秒で、正energy5%等の受動screenを通っただけである。

![実rawからの受動energy診断](docs/evidence/suspended-recovery/energy-diagnosis.png)

90度のdt/半dt同時刻比較では、荷物位置差最大0.642253m・姿勢差約179.993度、energy差最大0.000466602Jが残る。170度の約13%損失も数値品質の不確定要素である。受動比較へpoweredの2cm/2度判定を流用せず、powered dt検証や全資格の代替にはしない。

<!-- pagebreak -->

## 4. CPU実装と未統合の境界

62条件manifestは、継承55条件を保持して7つの受動半dt条件を追加する。case IDの旧数値名だけに依存せず、実substepsとdt/半dtを明示する。32受動traceの実raw・初期状態・物性・ゼロ外力・全sourceを再構成するbarrierを追加し、powered/refinement CLIはGPU初期化前にそれを要求する。短時間診断やstatus手編集で回避できない。

native coreはpayload29-ideal-world-current-action-v2、即時action適用、現在actionを次観測historyへ入れる順序を維持する。retargetでbody/joint/速度/historyを保持し、resetは物性を再測定する。異常はclip/観測/PPO前に実getterと直前raw windowを保存し、恒久quarantineする。reset・step後の変換例外でも現状態と完了不明を残す。

荷物専用missionは、機体と荷物の全条件が100個の完全な0.01秒区間で同時成立して初めて訪問完了とする。最初の新shared completion後だけC43→100、またはF→Aだけ1→5を適用する。実pose・残タスクからQ*/J*/suffix・反実仮想・成分コストを再計算し、採用routingのnominal入力とは区別する。今回の実イベント時刻は未測定である。

3寸法それぞれに実N環境sceneを構築するbatch backendも追加した。logical rowとepisode ID、個別mass/inertia/reset/historyを分け、RPMはfull[N,4]、inactiveを遷移へ数えない。全slotの数値異常でrolloutを隔離する。ただしGPUでの実geometry/隔離・30partial reset・N1等価性・memory/速度は未検証である。現batchのsubsteps上限128では、base128の半dt256比較も未対応であり、完成したqualification runnerではない。

sample計数はS=N*T*Kとvector step/optimizer更新を区別し、割切れない予算・invalid/paddingの混入を拒否する。seed7とlogical episode IDからmass/sizeを別streamで抽選する。これらはCPU準備であり、学習済み方策や測定された並列高速化ではない。完全62 admission、training共通化、validation/freeze、254coordinator、camera/HUD/全frame検査は未統合のままで、production入口は閉じている。

<!-- pagebreak -->

## 5. GPU予算・実行source・転送

新規workspaceにPython3.12.3のflight169版・CPU routing33版を既存lockどおり構築し、全版を照合した。最初の容量不足とvenv再作成失敗も保存し、hardlink同期で復旧した。これらはCUDA初期化前のCPU環境構築であり、GPU job時間には混入させず別記録とした。旧環境を流用せず、追加レンタル・契約変更は行っていない。

GPUは常に1job・有限timeout。今回6jobはbaseline60.015728秒、小診断21.564073/21.506851秒、30秒診断125.492373/125.790767/237.286058秒。増分591.655848817秒、累計11415.783737010秒、残り17384.216262990秒、200job・active0である。初期化やjob内のCPU/記録待ちを含むwall時間を全計上した。

実記録を有効にした30秒traceのloop wallは、128で115.624808/115.963804秒、256で227.382655秒だった。初期化等を別に戻した新規N1全62の線形予測は約18645.3秒、既存3traceを再利用できる仮定でも残59で約18156.3秒となる。これはmotor/reset、学習、validation、254件、撮影を含まない楽観的予測で、残予算を既に上回る。

未資格batchの速度を仮定して後段予算を成立したことにせず、specの停止規定に従って依存GPU作業を止めた。22%稼働率から倍率を推定していない。N1/8/32/128比較・PPO timing・両最終学習の共通予算freezeは未達である。

実GPU sourceはsource-startとcandidate-s128-source-v2のarchiveへ固定した。最終sourceの追加stageguard・native runtime/mission/batch等はCPU検証であり、V2物理結果を差し替えない。最終転送archiveは7270159bytes、SHA-256 3910f05507ac572cb3febd76d232d391ef310ef05552388383d85d867ad5e3b2。全83payload memberのサイズ/hashをローカルで照合した。大容量raw・モデル・動画・private pathはGit外である。

<!-- pagebreak -->

## 6. 全gateと次の具体案

物理：新sourceの完全62は未完、受動32中3traceのみ。新規motor15、30以上partial reset、選択batch・powered仕事上界の再資格は未達。元free vehicleの実質量0.02700047753751278kgとrotor/sourceは旧実測・同一pinned sourceを照合したが、今回の新15motor再測定とは区別する。

学習と比較：両方策とも0/300000総遷移。validation0/12条件・0/120遷移。randomized基本0/9・端点0/18・held-out0/100、fixed比較0/127、合計0/254。未実行の成功率・Wilson95%区間・優劣は報告しない。

動画：新しい全編raw/HUDは0/3本。全10地点shared settling、実1x・全decode/画素/字幕/telemetry照合、最終hash、人手連続視聴は未達。旧無荷物の承認を転用せず、映像がない状態で視聴確認を依頼していない。

CPU：最終local suiteは901passed・GPU1除外、lint/compile/空白検査は成功。Linux3.12の最終source検証、PDF全page描画/目視とartifact hashはverification.jsonに記録する。CPU成功を実物理・学習・受入へ読み替えない。

次は、保存した独立平面モデルと実rawを使って、数値散逸とdt軌道差を先に評価する。支持された設定/表現で、同じモデル・物性・仕事を保つ有限な修理案と全後段の実測予算を準備する必要がある。CONNECTへの変更だけによるmidpoint効果は見込めない。coupled integratorのengine patchや依存版変更が必要なら、具体的な修正specとその変更だけの明示承認を先に得る。今回はそれを実施していない。

公開文書・stageには接続情報・マシン固有絶対path・個人ファイルを含めない。指定identityの新規local commitへ、新spec・公開指示・handoff・索引・コード・テスト・必要文書・compact evidenceをまとめる。push、PR作成/merge、動画公開、提出、統合demo_full/提出用再編集は実行していない。
