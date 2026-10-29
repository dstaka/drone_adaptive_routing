# 吊り荷付き2機NN協調：一時停止報告と再開手順

2026-10-29、ユーザーが「時間がかかりすぎること」「現在の問題設定で目標達成できるか不明瞭」を理由に開発の一時停止を指示した。目標はpausedで、達成・受入合格・学習収束を主張しない。今回のshutdownと回収・文書化だけを完了し、自動再開しない。

- ブランチ: `feature/dual-drone-learned-coordination`。local commitのみ。push／PR作成／merge／公開なし。
- 承認spec: [2026-10-27 spec](../specs/2026-10-27-dual-drone-learned-coordination/spec.md)。マージ済みの過去spec・成果物は保持。
- 同一run: `outputs/dual_payload_coordination_20261027T144754Z`。
- 実験停止の確認: **2026-10-29T13:33:55.401784+00:00**。対象PID38003（学習）、34138（学習親）、64922（後続評価）、43940（監視）、34074（履歴保守）と記録済み子processは停止。ローカル同期PID48044も停止。共有サービス／サーバ契約・インスタンス自体は停止・破棄していない。

## 残った成果と未達

|工程|停止時点の状態|
|---|---|
|実装|同一MuJoCo世界の2機・各吊り荷、joint NN/PPO、担当変更・WAIT・優先機、安全監視、比較／アブレーション／freeze／media／回収のコードを実装|
|物理資格|現在のnative実行profileで81/81条件と補足10check合格。これは最終NN物理127条件の代替ではない|
|ソフトウェア検証|最後の全体回帰1741合格・48skip・Genesis1除外。後続のCPU管理114合格、監視／progress／回収67合格、評価並列化／監視19合格。停止作業では学習コードを変更せず、全体回帰を再実行していない|
|CPU/GPU|CPU1/2/4/7を測定し7workerを採用。GPU NN更新は速かったが全体効果が小さく、GPU物理の有効な優位性は未確立。ユーザー指示でCPUに固定|
|有限学習|seed42のみ107/256更新、438,272確定遷移。stage2。保存partial rollout2814を含む収集441,086。seed43／44未開始|
|validation／選定|未実施。選定モデルなし|
|joint freeze／held-out|未実施・未生成。未使用データによる再選定なし|
|最終NN動画3本／人手確認|未実施。既存のdevelopment B-static動画は成果の参考で、NN受入・最終3本の代替ではない|
|受入結果|**未達・未検証**。2機NNが巡回時間を短縮した証拠は得ていない|

## 学習が十分とは言えなかった根拠

105更新時点の同じstage2で、更新65–80→96–105の平均価値損失1390.18→666.35、方策損失0.00785→0.00263、masked entropy0.06157→0.02226。固定4096遷移当たり報酬は−22527→−22691で改善せず、損失低下だけを収束・成功とみなせない。

13:19Zのtraining観測では各worker直近20terminalずつ140件中成功2件、leg timeout104件、参照経路生成失敗34件。13:26Zの別snapshotでは成功3件、timeout104件、参照生成失敗33件。窓が違うので同一140件として混ぜない。後者ではtimeout104件中68件に目標提案変更、27件は期限時点でWAIT。期限超過区間のWAIT提案25.81%、全140件の安全保持drone-seconds／2機の総時間46.33%。これは5Hz介入interval率ではなく、受入の10% gateと直接比較しない。

これらは確率的trainingの保存ログで、greedy性能や正式validationの測定ではない。予算を続ければ改善するとも、期限だけ延ばせば解決するとも断定できない。実行可能な参照曲線の探索失敗・目標変更・安全保持・報酬設計を、次の大規模学習より先に小さい開発条件で切り分ける価値がある。今回はその追加実験を行わず停止した。

1seed化、総更新160／128への削減、epoch削減、区間期限30→60秒は相談・見積りのみで、**採用・適用していない**。3seed、256更新、5epoch、30秒区間／300秒team、安全条件、受入基準は元のまま。

## 保存した再開状態

- checkpoint: `outputs/dual_payload_coordination_20261027T144754Z/recovered/delivery-paused-campaign-001/root/run/production-training-001/seed-42/current-recovery.pt`
- サイズ: **16,458,209 bytes**
- SHA-256: `ceae64f79756bf91e60ae8deeb7f29a4194e87c546e534327655ba28954e175e`
- `selectable=false`。optimizer、Python／NumPy／Torch RNG、7workerの物理・mission状態、sampler、partial rollout、前stage履歴を含む。
- 元Linux環境の厳密loaderで受理。Torch2.14.1+cu132、NumPy1.26.4、Python3.12、MuJoCo3.15.0。停止後の物理再実行、NN forward、勾配更新はしていない。
- SIGINT時にworker0–6の最終flush acknowledgement未確認が記録された。元failure JSONを保持。最後のdurable checkpointが再開点であり、未保存のメモリ状態を復元できるとは主張しない。

今回の停止packet42files／43,755,252bytesは独立した送信元・回収先inventoryの全サイズ／SHA一致。以前の固定入力packet185files／51,921,409bytesも検証済み。元サーバのコピーは削除していない。

|保存先（run配下）|内容|
|---|---|
|`recovered/delivery-paused-campaign-001/`|学習全状態・圧縮episode・停止／失敗・loader検証・lock／benchmark・runtime依存版／MuJoCo library・remote timing最終snapshot|
|`recovered/delivery-frozen-inputs-001/root/`|元の176個のboundファイルと再現supportを含む固定source／assets／controller／学習・validation宣言／資格・replay参照|
|`recovered/development-video-003/`|以前作成したdevelopment動画。NN受入には使わない|
|`analysis/live-convergence-001/`|105更新時点の損失・報酬図PNG／PDF|
|`analysis/training-timeouts-001-ja.md`|直近training timeout診断と限界|
|`operational-tools/`|実行・計測・停止・回収・診断の今回用スクリプト|
|`receipts/recovery-batch-delivery-paused-campaign-001.json`|停止packetの独立回収照合結果|
|`timing/paused-report-001/`|全試行のUTC／monotonic別表JSON・CSV・日本語表。元の失敗・再実行・待機を保持|

これらの大きな成果物はGit管理外のローカルrunに置く。Git commitだけではモデルや動画は復元できないため、このrun自体もバックアップ対象とする。認証情報・秘密鍵・全環境変数dumpは含めていない。

## 再開する場合の手順（今回は実行しない）

1. 新たなユーザー指示で、元計画を続けるか問題設定から見直すかを決める。今回のSSHサーバが将来使えるとは仮定しない。
2. Linuxで元の絶対root `/workspace/dual_payload_coordination_20261027T144754Z` を再現し、先に固定入力packetの`root/`、次に停止packetの`root/`を同じ相対構造で復元する。元LearningLockのroot・内容を編集しない。
3. 停止packetの`runtime/dependencies.json`の依存版を再現する。`runtime/libmujoco.so.3.15.0`の元配置先も記録されている。CPU数7以上の実効quotaと空き容量を確認する。元native `.so`はLinux向けでmacOSでは動かない。
4. 独立サイズ／SHAを照合し、元source018でLearningLockとrecovery loaderを再確認する。完全性検査と実際の物理再現は別物なので、再開前に小さい保存状態の物理再現確認を行う。
5. 元条件を続けると決めた場合のみ、次の明示resumeを使う。`--resume`なしの新seed開始、途中checkpointのterminal扱い、既存結果上書きをしない。

```bash
cd /workspace/dual_payload_coordination_20261027T144754Z/source-production-018
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ../venv/bin/python -m adaptive_drone_patrol.dual_coordination_train --learning-lock ../run/cpu-production-pipeline-004/learning-lock.json --run --resume --original inputs/cf2x/cf2x.urdf --design inputs/controller.json --output /workspace/dual_payload_coordination_20261027T144754Z/run/production-training-001 --workers 7 --journal-root /workspace/dual_payload_coordination_20261027T144754Z/run
```

以前のpost-training-pipeline003は停止済みで、そのまま再起動すると既存出力directory作成で失敗する。自動後続処理を再開する場合は、当時の待機／失敗状態を保持した新しい管理attemptを用意する。元の順序は3terminal→validation1500と一度の選定→診断600→joint freeze→新規routing1500＋physical127→既存手順の最終3動画→新しい全編1倍速人手確認。適格seedなしなら元仕様どおり未達で停止する。

## 時間と振り返り

開始2026-10-27T14:47:54Z→開発process停止2026-10-29T13:33:55.401784+00:00は**46時間46分01秒**。停止後の回収・文書化を含むhandoff時点の経過はtiming表の生成時刻つき値を参照。全体の実行／待機／blocked内訳は`not recorded`。並列workerや親子processの時間を合算して全体時間と呼ばない。

|実測対象|所要時間と限界|
|---|---|
|最終native資格81条件|7048.010秒、約1時間57分。これ以前の失敗・再資格・補足は別試行として全表に保持|
|CPU1／2／4／7の各warm-up＋3反復|9101.446／5032.898／2754.644／1632.887秒。逐次測定の合計約5時間08分42秒|
|本番学習開始→ユーザー停止|42298.445秒、約11時間44分58秒。seed42未完了|
|最後の全体pytest|1741passed等、pytest849.63秒。wrapper monotonic851.437秒とUTC4503.728秒に差があり、原因・内訳はnot recorded|
|今回のprocess停止|4.052秒。回収・文書化時間は別試行|
|実装・GPU検討・各テスト・待機・失敗・転送|全工程／試行表に元記録を保持。不明な開始・終了・内訳はnot recorded|

時間がかかった主要因は、停止予測horizonの不備による物理再資格、GPU互換性・学習経路の検討と失敗、CPU worker候補の有限測定、native物理・参照計算・安全監査を各遷移で行う高コスト、全体回帰の反復、同期・承認・環境是正の待機だった。NNのPPO更新自体はCPU7worker一巡407.591秒中16.932秒（4.15%）に留まり、epochやGPU NN更新だけでは全体を大きく縮めにくかった。

次回は、長い資格・全比較を本番規模で積み上げる前に、小規模で完遂率・保持時間・target churn・参照生成失敗を調べ、問題設定と報酬から学習が成立しそうかを確認する。損失低下を成功と誤認しない。backend調査には終了条件を設け、最初の全体実測で主なコストを特定する。変更に関係しない全体テスト・同一証拠の再監査を繰り返さず、必要な安全検査と比較の意味を保って独立評価を並列化する。これは将来への提案で、今回の条件変更ではない。

## 進捗の読取り

ローカル`outputs/dual_payload_coordination_20261027T144754Z/progress-ja.md`と`outputs/dual_payload_coordination_20261027T144754Z/progress.json`は一時停止の最終snapshot。定期同期・heartbeatは終了している。将来の読取りでは、保存時点の記録と現在の稼働状態を区別する。

sidechat確認文: 「`outputs/dual_payload_coordination_20261027T144754Z` のprogress-ja.md／progress.json、timing/paused-report-001と一時停止handoffを読み、保存時刻、停止理由、学習の確定更新数とpartial rollout、未達項目、再開条件を教えて。現在動いているとは推測せず、not recordedはそのまま示して。」
