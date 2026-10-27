# 高度を上げた吊り荷巡回の実装・評価結果

本変更は [Revision2仕様](../specs/2026-10-25-elevated-payload-destinations-detail-zoom/spec.md) に基づく。実装、3seed有限学習、validation選定、joint freeze後の未使用routing300件と別の物理127件、最終動画3本の機械検証が完了した。最終3本の全編1倍速人手確認も利用者返信により合格した。必要全成果物の回収照合も完了し、今回分をlocal commitへまとめる。

## 変更と保持した条件

A–Jの高度だけを元座標の `z_new=2+1.5*(z_old-1)` で変換した。範囲2.126634138–4.867951029m、平均3.263839668m、高度差2.741316891m。x/y、S、床、初期状態分布、目的関数、イベント、機体・有質量リンク・ball joint・モータ・制御器・静定・安全・時間制限は保持した。右の512×720個別拡大図は距離1.25→0.90m。左俯瞰は全10地点と高度範囲を収める固定画角。旧reader/設定/成果物とマージ済みspecは保持した。

MuJoCo3.15.0、float64、RK4、内部16substeps、100Hz、Scene N=1、`C-local-lqr-v1`。飛行NNは学習・使用していない。routingのみMasked PPOを有限追加学習した。指定された新Linuxサーバを使い、GPUは使用していない。

## 有限routing比較と選定

Bのpolicy/value重みから独立初期化し、seed42/43/44ごとにfresh Adam・RNG・counterを使った。各50176 static＋50176 mixture、合計301056 transitions。architecture/schema/normalization/objectiveは変更せず、terminal checkpointだけを比較した。固定300 validation（各100 static/priority/cost）に全4モデルを共通入力で評価し、事前1%改善・各stratum5%劣化防止・絶対品質gateによりF43を選定した。

| モデル | validation平均cost | static平均cost/exact | 全件完了・不正action |
| --- | ---: | ---: | --- |
| B | 357.041243388 | 1.073809828 | 300/300・0 |
| F42 | 345.884968175 | 1.038816945 | 300/300・0 |
| F43（採用） | 344.406874427 | 1.031046592 | 300/300・0 |
| F44 | 345.914677656 | 1.035168516 | 300/300・0 |

F43のvalidation平均はBより約3.54%低い。static平均cost344.545195988、greedy384.345094788、fixed606.553728233、static最悪exact比1.117697066。正のexact gapがあり最適経路とは呼ばない。独立したrouting held-outでも平均cost改善を測定した（下記）。各caseでの一律改善や最適性は主張しない。変化対象が訪問済みの場合と到達不能な有向辺の場合を区別した補足イベント監査を、元validation結果を書き換えず保存した。

B SHA-256: `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef`。採用F43: `ee966a2f761b513074c6df2e7de177b3d7bc76ac9d5ec3602d273e5210e84652`。実行環境差は共通validation1200推論で全order・cost一致を確認した。選定をやり直すための比較ではない。

## 新高度の物理開発と画角

選定F43による所定の開発15/15、12条件・120異座標移動の高度検証12/12がrunner内の独立mission/native/RK再構成に成功した。制御器・referenceの調整候補は0。joint freeze前に別コマンドで全27件のrawを再監査し、選定開発15件の全100Hz姿勢で両パネルのframingと拡大率を検証した。新154633区間・9896512RK内部段のnative/stage再生最大差はともに0。最大signed-work残差2.105592584e-11J、拘束残差2.609024108e-15m、最小package床距離0.609570355m、drone0.977332816m。最大held-reference高度誤差0.037476700m、leg端点範囲を越えるovershoot0.035356046m。各camera中央値は旧比1.41500664～1.41500838。詳細は [前提監査](evidence/elevated/joint-prerequisites.json) を参照。

先行B動画は開発・暫定版として保存し、F43必須15件や正式127分母には加えない。normal46.22物理秒、4623frame、全10地点成功。295808RK段の同一入力再生差0、最大signed-work残差9.306e-12J、最小package床距離0.631373749m、drone0.984075685m。右拡大率中央値1.4150（旧画角比+41.50%）、最小余白は左右上下すべて89px超。E→D実上昇2.10142786m、H→J実下降1.24558225m。

暫定動画: `outputs/suspended_mujoco_elevated_20261025T141122Z/provisional-normal-media-v1/hud.mp4`、13939338bytes、SHA `da16ac3cc7a62dc20b6869ee420dbf341423b25a617ae4c21630403f8ca547f4`。全可逆master・再合成・PTS・字幕proofも保存した。新たな成功数を増やすcamera-only再描画とは区別する。

## 監査・失敗・残作業

元最終3608files/8641550668bytes、330秒資格7519files/5137376485bytes、元数値資格2669files/7723572007bytes、C70診断705files/141473545bytesをローカル再hashした。Linuxの実engine/library bindingと物理/controller/reference依存は旧資格と一致。新高度の成功を旧127件から流用しない。

新実行は `outputs/suspended_mujoco_elevated_20261025T141122Z/`。転送・環境構築・preflightの失敗は同namespaceのログ/receiptに保存する。原asset名の誤指定によるfreeze準備中断、controller JSON整形差を値の変更と誤認した監査失敗は、held-out生成前の準備失敗。物理条件の変更や有利なcaseへの置換ではない。レビューでsingle-pass、全127証拠への結合、RAMリンク欠落・nested宣言欠落の検出を修正した。

全127件を一度だけ実行し、正式評価後に物理条件・モデル・source・camera・受入閾値を変更していない。最終画角の補足解析は凍結対象外の証拠scriptによるpostprocessingで、再飛行や成功数の追加ではない。push・PR作成・merge・公開は実施しない。

最終source-v6のCPU全体回帰は1226合格・20skip・1GPU専用除外、282.28秒、終了コード0。対象37テストを含む。joint freezeファイルSHA `f2fbe39ebf90d55507627d4a4d2d9178b1ef8800784cd08d5f6ecf7b4bc3e6f6`、canonical digest `7e66add5b62826064d176d15d2d33af4bc3dff3a245a6fce8bcf3948864d8a56`。

## 未使用routing held-out300件

joint freeze後に初めて生成した100 static/100 priority/100 costで、全4モデルが300/300完了・不正action0。選定F43の平均cost345.061332580、static平均exact比1.033253913、static平均cost340.711148938（greedy384.999561200、fixed584.669278391）で絶対gate合格。B平均355.530542240に対し差−10.469209659、約2.9447%低い。F42平均345.521869636、F44平均345.582798727。モデル再選定なし。

同一caseでF43がBより低cost170件、高cost71件、同等59件。priority平均差−10.336910353、cost-event平均差−11.337168739、static平均差−9.733549886。実際のevent境界でroute適用可能175、適用不能25。適用可能中でbeneficial37、harmful19、no-change119を保存した。dynamicの実pre/post境界400件中310に1e-8超の正のexact gap、最大80.564282553。**最適経路や全event改善とは呼ばない。** 全caseのorder/cost成分・固定/greedy・静的exact・動的境界exact・counterfactual suffixはrun内に保存し、[paired要約](evidence/elevated/routing-heldout.json) に記載する。routing300件は物理127件の分母に加えない。

観測表現は既存float32のunbounded Box、座標/2・3D距離/4等の正規化と相対座標networkを維持した。高さは旧学習のencoded0.5から約1–2.5へ分布が移るが、clipや追加rescaleはない。全10000 training/300 validation/300 held-out初期観測のfinite/rangeを [表現監査](evidence/elevated/observation-representation.json) に記録した。追加学習の挙動は実測し、表現可能性だけから性能を推定しない。

## 正式127件の実測と独立監査

新root10262501をjoint freeze後に初めて生成し、全caseを単発実行した。失敗置換・再試行は0。runnerとは別コマンドで、全raw/control/native/RK・10地点・同時静定・高度変化・oriented floor・continuity・mission/wall/flush clockを再構成した。[集計証拠](evidence/elevated/physical-formal.json) の分母にrouting300件は含めない。

| 集団 | 成功 | Wilson 95%区間 |
| --- | ---: | --- |
| basic | 9/9 | 70.0855–100% |
| 端点 | 18/18 | 82.4121–100% |
| 未使用物理held-out | 100/100 | 96.3007–100% |

scenario別はnormal43/43、emergency42/42、cost_shift42/42。627317制御区間・40148288RK段、native/stage再生最大差0、最大signed-work残差1.948189121e-11J、拘束残差2.775557562e-15m。最小package床距離0.608760225m（endpoint-emergency-2-1、0.1496875s、F、RK_STAGE）、drone0.977117021m（endpoint-cost_shift-1-1、0.2928125s、F、RK_STAGE）。物理時間45.10–57.11秒、中央値49.96秒。flushまでのwall最大167.018743秒で600秒以内。

最大振れ角12.282442度、**過渡的な箱全角速度338.476397度/秒**、held-reference高度誤差0.037489021m、leg端点高度範囲overshoot0.035356863m。箱速度の受入は各地点の連続1秒静定で判定し、過渡全区間が30度/秒以下とは主張しない。継承したより厳しいstrict shadowは実trace1270stop中0合格。これは動画優先基準と別の診断であり、strict方式でmission全体を再実行した結果ではない。厳しい静止性能の達成と読み替えない。

planning座標と実測poseの各1270判断境界中385に正のexact gapが残る。最大はそれぞれ13.858428553、14.269182749。実動的event84境界では再計画post-event gapは全て1e-8以下、同状態反実仮想比較で悪化0だが、ランダムrouting held-outではharmful19件がある。固定demoの結果から任意入力での最適性や常時改善を主張しない。全境界証拠はrun内 `formal-exact-boundaries.json`、全127高度系列は `formal-v1-derived/*-z.csv.gz` に保存した。

## 最終3動画と画角

録画caseは事前指定basic seed1だけで、F43を使用した。全10地点完了、100Hzの初期・終端を含む14526frame。撮影による追加物理成功数0。末尾frameは通常1表示期間だけ保持し、余分な物理stepはない。各MP4尺は物理時間より0.01秒長い。

| scenario | 物理秒／frame | 実測上昇・下降の例 | HUD MP4 bytes／SHA-256 |
| --- | --- | --- | --- |
| normal | 45.27／4528 | E→D +2.101427m、H→J −1.245583m | 13617251／`115f3a34b5959dd3704a1906fec9bfd1c4194338f5b6b6125dc223168bd7273f` |
| emergency | 50.51／5052 | E→D +2.101421m、G→I −2.745135m | 14830109／`25be856dc76fe3396569667312de5f633d0ef87af0c935120967cbd15cd2dbe2` |
| cost_shift | 49.45／4946 | C→H +1.712503m、G→I −2.745133m | 14150941／`bd87c3636c65d328c98ed94d10515881a89639841fb7c1c009cfa3896292a367` |

ローカルのrun直下 `final-{scenario}-media-v1/hud.mp4` が人手確認対象。raw/HUDそれぞれ可逆FFV1 masterとviewing MP4を保持し、全frameの独立再合成、decode、PTS、字幕領域、source/asset/raw hashを検証した。最終capture前後に全127入力と独立監査への結合も再検証した。右detailの幅中央値は旧比1.415006876–1.415008097（+41.50%）、assembly全体の最小余白は左91.871px・上133.594px・右91.433px・下144.612px。身体の計算にはrotor、箱角、rod、衝突形状を包む実geom_rboundを用いた。

**目標マーカー境界の補足:** 凍結済みcheckerはdecorative sphere/cylinder/boxに共通半径rを用いており、円柱・箱を過小評価していた。元receipt/sourceは保存し、変更していない。全形状を包むsqrt(3)rと実sceneのfloat変換誤差による独立補足を追加した。凍結済み15開発＋最終3本の全native frame数・入力hashに結合し、固定overview cameraと実modelから再投影した。左235.726px・上84.530px・右292.776px・下211.147pxで、元の40px基準を満たす。これは同じ実画角の強化検査であり、画角の調整ではない。文字の読みやすさ・重なりは別の人手確認で合格した。一般化したcheckerの修正や画角変更を次に行うなら、新たな宣言と未使用cohortを要する。

撮影時HUDとproofの人手確認待ち表記は保持する。利用者は上記3SHAについて「3本すべて全編1倍速で確認し、各項目に問題なし」と返信した。[人手確認receipt](evidence/elevated/human-review.json) に結び付け、動画bytesは変更していない。

## 受渡し状態

新runの絶対パスは `/Users/takanori/repo/physical_ai_adv1/drone_adaptive_routing/outputs/suspended_mujoco_elevated_20261025T141122Z`。必要raw、RK、B/F42/F43/F44とschema、全train/validation/held-out入力、sampling/optimizer/曲線、選定・freeze・資格再利用、実engine/source/config/asset/font、動画masters/proofs、ログを大容量Git対象外として回収した。正式127全体は39104files・25899565860bytesを回収・照合済み。指定サーバの全59788files・36851003632bytesについて全size/SHA-256照合が一致した。全体回収と人手確認の証拠は [handoff](elevated-payload-handoff-ja.md) と [回収receipt](evidence/elevated/recovery.json) に記録する。

高度系列の例: [normal](evidence/elevated/basic-normal-1-z.png)、[emergency](evidence/elevated/basic-emergency-1-z.png)、[cost_shift](evidence/elevated/basic-cost_shift-1-z.png)。実測drone/packageとheld referenceを別線で示す。荷物の目標は機体目標と同じ高さではなく、元の取付・リンク・箱による差を保つ。

最終ローカル受渡し目録は **60127files・36921663846bytes**（約36.92GB）。全remote entriesのsize/SHAと一致し、ローカルderived資料・人手receiptも含む。目録はrunと同じ階層の `suspended_mujoco_elevated_20261025T141122Z.delivery-inventory.json`。旧成果物とmerged specは保持し、公開操作は行っていない。
