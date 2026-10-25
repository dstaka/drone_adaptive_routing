# 吊り荷solver recoveryの実行報告

2026年10月23日 / 旧方針の実行終了 / 完全数値資格は通過、学習飛行・受入・動画は未達

## 1. 現在の結果と適用仕様

specs/2026-10-22-suspended-payload-solver-recovery/spec.mdに基づいて、資料準備から実装・実験へ進んだ。一般承認を再度待たずに実行し、旧成果物と失敗を保存している。今回のbranchはfeature/suspended-payload-solver-recovery、起点はe39cb098a9edc47007c5d427ae0d220939075ddaである。

MuJoCo3.15.0、float64、RK4、control0.01秒、16substepsのN1経路で、受動32・powered18・refinement12の全62trace、motor15、実reset30、初期姿勢reset30を新規実行し、独立raw再構成を通過した。receiptのstatusだけではoptimizerを解除せず、実行consumerが完全なraw・現在source・実solver buildを再検証する。

完全資格後のrandomized development pilotは20,000、200,000、200,000、300,000、300,000の計1,020,000有効遷移を完了した。共有settling完了はいずれも0であり、学習飛行の合格ではない。位置二乗costを追加したP5でも20k/100k/300kの全checkpointがhover30秒timeoutだった。実native照合は通過した。

両方式の非pilot共通ladderを事前宣言し、randomized・fixedとも同じ初期値・設定から各300万遷移を完了した。pilotを合わせた実学習は計702万遷移、共有完了は両方式とも0だった。30万・100万・300万時点はすべて両方式0/6で、全36飛行が最初のhover30秒timeoutだった。全36件のnative再検査と全raw・checkpointのローカル配送、全memberのsize/SHA・永続化を確認した。学習の経験条件は不適格であり、fresh最終学習・正式validation12条件/120実target遷移・paired受入254件・合格方策の全編raw/HUD動画3本は未実施である。2026-10-23のユーザー指示で旧方針の追加学習を終了した。新しいハイブリッド制御・Genesis優先specはレビューしたが、今回は実装を開始していない。

GPU累積時間上限と使用量trackingは撤廃された。旧台帳は履歴として残し、今回更新していない。同時1job、通常missionのwall600秒、deadlock停止は維持する。新規レンタル・契約変更・追加サーバ・サーバ破棄は行わない。

<!-- pagebreak -->

## 2. 数値修理と保持した実物理

Genesisの対応設定と積分経路を小診断で比較し、独立平面RK4参照の収束を確認した。大角度の長時間軌道差だけで正誤を決めず、短時間収束、energy/work、拘束、3刻み幅と実sourceに基づいてMuJoCo候補を選んだ。旧V10の530.703% energy増加、旧170度候補の約13%損失は過去の失敗であり、今回の合格へ読み替えない。

0.30mの圧縮も伝えるmassiveな剛体rod、両端の無制限受動ball joint、機体への反力・反モーメント、元motor権限を維持した。機体下面attachmentは(0,0,-0.020)m、荷物上面attachmentは各実寸法の半辺である。cubeは一辺0.030/0.045/0.060mを等確率、質量比0.10-0.30を独立一様分布とし、実mass/inertia/collisionをreset時に反映する。

最初のMuJoCo候補では、角減衰の基準massをregularized FREE massにした差が物性許容1e-5を超えたため、候補とreceiptを不採用にした。元URDFのauthored mass0.027kgを減衰基準へ戻して完全資格を取り直した。荷物massの基準は実測FREE mass0.02700047753751278kgのままである。

現在の全powered検算で、最大signed energy-work残差は5.202029460888436e-11J、最低床距離0.437935m、最大roll/pitch25.969度だった。同一RPMのbase16/half32比較6組では、全body位置差最大5.44985e-7m、姿勢角差最大4.51775e-6度だった。受動では未説明energy収支の絶対5%条件を独立に検査した。

実solver libraryはlibmujoco.so.3.15.0、SHA-256はd3a1476dd52e220031aca40b2b00f69d1febce2ee85c57d5015f68b0cc72afa9。版番号付きlibraryをhash一覧から漏らした旧receiptは使用せず、試験前の全installed inventoryとreference/flight両環境を一致させたschema-v2へ更新した。

<!-- pagebreak -->

## 3. 共通runtime・実学習・失敗の意味

actorは29観測・4motorのcurrent-action-v2である。元model4296の0:13列を移植し、追加13:29列は0、他の互換tensorは正しくcopyする。mass、size、bank ID、seedはactorへ渡さない。実actionは直ちに元のclip/RPM mappingで適用し、次観測historyには現在の適用actionを入れる。rollout境界ではbody・荷物・historyをresetしない。

初回20,000遷移では200rollout更新、4,000optimizer更新、無効0、安全失敗0、共有完了0だった。独立prefix監査で生成/有効/消費、native getter、物性、時刻、action/historyを照合した。過去の全stochastic actorやRNGを最終mean checkpointから再現したとは主張しない。

lambda0.99の200,000遷移では52episode中38安全失敗、共有完了0だった。bounded-spin/lambda0.95の200,000遷移では17episode中安全失敗0、共有完了0だった。各20k/100k/最終checkpointの同一条件mean-action評価も、hover30秒timeoutまたは安全失敗だった。bounded-spin20kの後半500stepは位置誤差0.35411m、荷物spin43.98度/秒であり、spinの低下だけを訪問成功へ読み替えない。

初回20kの独立raw調査では、少なくとも1motorがclipされるstep68.93%、pitch角速度lag1相関-0.783、action/history照合誤差0だった。同じ訪問観測で初期/20k actorを固定再評価してもmean出力自体の範囲超過は約69%で、noiseだけの問題ではない。ただしclip率はPPOの学習不能率を表さない。

角速度costを一次normから二乗normへ強めた試験では、最後20kのclip step率が0.04%、機体角速度norm平均0.391rad/sへ下がった。一方、同条件mean20kの位置誤差0.233m・荷物spin30.7度/秒、mean300kの位置誤差0.637m・spin56.3度/秒が残った。物理減衰、actionの時刻・権限・履歴、観測、荷物分布、settling閾値は変えていない。この結果からraw action超過costは優先せず、目標から遠い安定平衡にも戻りを促す位置二乗cost-2||実位置-目標||²を1項追加した。P5最終300kのmean誤差0.27755m・spin38.32度/秒が残り、sharedは未達だった。その後、報酬v5を保ち、gamma.999/lambda.99でcreditを延ばす共通設定を宣言し、両方式各3Mで実行した。100Hzの幾何学的discount尺度は約10秒、gamma-lambda trace尺度は約0.91秒であり、実際の100step rollout窓とは異なる。curriculum_stepsは1.8M、各初期stage約300k、stage5は残り約1.5Mとする。これらを合わせたrecipe試験であり、単独要因の因果効果は主張しない。

後半の荷物角速度は機体z軸まわりのtwistではなく、約4.207Hzの横振動だった。独立した固定支点・微小角のrod/cubeモデルの高速固有振動数4.20746Hzと整合した。これは失敗の診断であり、完全数値資格や制御可能性の証明ではない。下図の位置と角速度を含む全predicateを同時に100完全区間満たす必要があり、P4/P5はいずれも未達だった。

![同一条件の30万遷移checkpoint比較](docs/evidence/suspended-solver/pilot-matched-300k.png)

<!-- pagebreak -->

## 3.1. 共通学習の実記録からの診断

続く共通randomized3Mのraw調査では、stage0末尾100秒の1episodeで位置誤差平均0.15380m、位置・速度・link角度の5条件が100%通過していた。一方、荷物角速度平均95.833度/秒、角速度条件の通過率0%、shared dwell0だった。横方向の約4.22Hz modeが残り、直接spin costは99.36%のsampleで上限に張り付いて振幅差を区別しなかった。同じ記録軌道でcapを外した算術比較は、再学習後の挙動予測ではない。stage4から抽出した107完全episodeはすべて最初のcornerで支持推力が不足し、40～49stepで荷物の床条件により終了した。全分布の成功・一貫したanti-damping・忘却はこれだけでは判定できず、両方式の全rung評価でも共有完了0だったため、既存recipeのまま学習量だけを増やす方針は採用しなかった。

![共通randomizedの実学習episode末尾10秒](docs/evidence/suspended-solver/long-credit-stage0-spin.png)

<!-- pagebreak -->

## 4. final学習と受入のgate

両方式の共通development ladderは300,000、1,000,000、3,000,000を出発点とし、必要なら根拠をもつ共通次rungを宣言する。randomizedのdevelopment6条件すべてで実10target sequenceを通過し、両方式の6stage経験と連続10地点経験を独立監査してからSを選ぶ。今回の小pilotはstage0だけであり、このgateを満たさない。

final計画は両方式のS、N/T、seed7、architecture、optimizer、reward、curriculum、初期tensor、layout、選択規則を同じにfreezeする。両最終学習は元model4296とfresh optimizerから開始し、development checkpointを初期値にしない。未採用freezerのレビューで、保存済み成功数だけを信用せず元rawをconsumerでも再構成する必要を確認した。consumerがindex・parentから全prefix/実評価/初期tensor/artifact集合を再構成し、最終modelにも実run rootの全raw・resetのmethod/seed・fresh初期値を必須照合するよう修正し、否定テストと再レビューを通した。実final計画の陽性発行は学習gate達成後に検証する。

validationはseeds11/12、3荷物条件、2方式で12条件/120実target遷移である。その後、両model・source・cases・routing・camera・checkerをfreezeし、元の127条件を各方式で新規実行する。randomizedは基本9/9、端点18/18、held-out95/100以上が必要で、固定比較も全127を残す。

全10地点では機体と荷物の全predicateを同時に1秒、100完全区間満たして訪問完了とする。床>0.10m、roll/pitch60度、leg30秒・mission120秒・wall600秒を維持する。配置delta_z=0、重要度seed40029、A81/B42/C43/D64/E23/F84/G6/H98/I22/J56、alpha1/beta0.05/除数100、有向倍率1-5を保つ。最初の実共有完了後、緊急は未完C43→100、cost変化はF→Aだけ1→5を適用する。

実行coordinatorと独立checkerは失敗を分母から外さず、全254件、size/mass/scenario subgroup、Wilson95%区間を出す。元caseのSHAをkeyにしたhost-local共通claimで、同じstreamを別freezeやpathで開き直すことを拒否する。正当な成果物の別hostでの読取再検査には、保存した開封証拠を用いる。held-outはまだ開いていない。

<!-- pagebreak -->

## 5. 動画・人手確認・保存

software EGLの明示deviceとllvmpipeを使い、実native qpos/qvelをstepせずに2視点へ描画する経路を確認した。overviewと同時刻の機体・荷物detailを横に並べ、下に日本語HUDを追加する。静止画の視認性を確認したが、合格方策の動画ではない。

100Hzの全control instant0..Nを各1frameに対応させ、event用重複frame、cut、時間外の追加物理stepを入れない。最後の実sampleには通常の1表示期間を設ける。FFV1/bgr0のlossless raw/HUD masterは、全decoded RGB、全PTS、元viewport画素の一致を検査する。MP4視聴用はcrf0/yuv420pで別保存する。29箇所の字幕・数値領域と24pixel foreground tileのluminance MAE4を全frameで検査する。crf17は正当な日本語glyphも誤差超過したため不採用とし、閾値を維持してencoderを強めた。日本語HUD3frame診断では最大領域誤差1.920、foreground誤差2.686で通過した。これは合格動画ではない。

合格randomizedの基本seed1・nominal荷物で通常/緊急/cost変化の3本を全10shared完了まで作る。実native記録から各frameのpose・字幕・telemetryを再構成して照合し、最終hashとevent前から次shared settlingまでの具体的区間を示して、新しい連続1x人手確認を依頼する。現在は未撮影・pendingである。

不採用物理候補は6,736,281,600B、修正済み完全資格と初回pilotは7,800,832,000Bのarchiveへ保存し、ローカルでsize/SHA-256一致と展開を確認した。後続pilot・失敗評価も全raw/checkpointを別archiveへ保存している。実行時source archiveと最終CPU sourceを区別し、不足artifactを捏造しない。

実行初期に、受動barrier前の短いpowered回帰が入った。この試行は資格から除外し、受動32後に全powered/motor/resetを新規実行した。early-flight-v1の例外、減衰候補の不採用、renderer初期化の失敗も履歴として残す。

<!-- pagebreak -->

## 6. 検証と残作業

localの広いCPU回帰は1087 passed、18 skipped、1 GPU deselectedだった。新PPO設定の関連24件、実P1 prefixの厳格照合・改変8件、nativeとderivedの同時改変2件も通過した。remoteでは新source v19の94件が通過した。skipは実backend/実runまたはLinuxのseal機能が必要な検査であり、CPU doubleを実物理資格へ読み替えない。remoteに未配置の資料indexへ依存していた否定テスト1件を自己完結fixtureへ修正し、失敗した実行記録を保存した。

CPUレビューでは、versioned engine libraryのhash漏れ、final計画のraw再結合不足、失敗prefixと証拠不正の混同、routing/camera参照のsnapshot対応、CPU fixtureによるphysical証拠の回避、EGL設定順序、freeze copyによる再試行の問題を指摘した。採用前に修正・再検証し、進行中のsourceは無断変更しない。撮影APIの方策との未結合、HUDイベントの目標・時刻・実pose費用の未検証も独立レビューで修正した。全254件のfresh再検査と、元のtrial artifact集合へのrender前後の照合を公開撮影APIにも必須化した。実サーバの関連48件が通過し、学習・物理runtimeのsourceは変更していない。その後、10個目shared完了とepisode期限が同じstepの場合のPPO bootstrapだけlocalの次sourceで修正し、関連40件・全回帰1087件と独立レビューを通過した。完全rawから終了理由を再構成し、保存timeoutフラグは期限到達の事実を維持する。この境界は共有完了0の今回randomizedでは発火しておらず、現在のv19比較sourceへはまだ配置していない。

終了時の追加コードレビューで未修正P2を1件確認した。旧exposure監査は宣言command RNG・target順序を独立再構成しておらず、rawを変えずにpilotをnonpilotへ再ラベルした一時コピーを受理した。物理資格偽造や実P6の宣言違反を示す再現ではないが、完全なcurriculum証明とは扱えない。全経験gateは不適格、final/adoptionは未発行であり、新hybridの独立監査では修正が必要である。

旧方針で未達のまま引き継ぐのは、学習hover/実往復、共通development選択、fresh最終学習、validation、freeze、全254受入、合格方策の動画と新規人手確認である。これらを完了扱いにせず、ユーザー指示による方針変更として区切る。数値資格・独立checker・全失敗・原raw/source・実学習量・未使用held-outを保存した。

新specの方向性は妥当と判断した。制御器単独で先に制振と実移動を確認し、NN補正・教師学習と比較する。29観測固定の解除やGenesisの実GPU並列は新しい契約・全数値資格を必要とし、旧N1のPASSを流用できない。controller-only最終採用と最終学習必須の関係、N1 fallbackと並列完了必須の関係は、次の最終plan前に明確化する。制振可能性、実センサー、全分布の成功やGPUの高速化はまだ証明していない。

旧実行指示は履歴として保持する。次の方針はspecs/2026-10-23-suspended-payload-hybrid-genesis/spec.md、レビューはdocs/suspended-payload-hybrid-genesis-review_ja.md、旧成果・全gate・source/raw/hashはdocs/suspended-payload-solver-recovery-handoff.mdを参照する。push、PR作成・merge、動画公開、コース提出は行わない。
