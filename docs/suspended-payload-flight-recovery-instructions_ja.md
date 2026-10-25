# 新Codexセッションへ貼り付ける実行指示

この公開ファイルに接続情報・ローカル絶対pathは含めない。新セッションは対象リポジトリのルートで開き、現在有効なGPU接続コマンドを別の非公開メッセージで渡す。以下の本文を人が新セッションへ送ると実装開始指示になる。資料を作成・レビューしただけのセッションでは実装やGPU作業を開始しない。

---

このリポジトリのルートで `specs/2026-10-22-suspended-payload-flight-recovery/spec.md` に基づき、前回達成できなかった吊り荷飛行の物理修理・学習・シミュレーション・比較評価・動画撮影まで実装してください。最初に実装計画を提示してください。その後、この指示と新spec内の通常判断について計画承認を改めて待つ必要はありません。計画提示だけで終了しないでください。

私が別途提示する現在有効なGPU接続先へのSSH、必要な転送、新しい隔離環境の構築、新spec範囲の物理probe・batch検証・計測・学習・評価・撮影を許可します。GPUが必要になるまで接続しないでください。接続先が未提示・無効なら必要な情報を尋ね、独立CPU作業を続けてください。追加レンタル、契約変更、サーバ破棄、8時間枠の追加は許可しません。接続情報やマシン固有絶対pathを公開文書・Gitへ含めないでください。

## 最初に読むものと保護

- `AGENTS.md`、新spec全文、`docs/suspended-payload-flight-recovery-handoff.md`、2026-10-21 retry spec/handoff、元payload spec/handoffを読んでください。先行5 spec/handoffと分散配置の採用・失敗・互換性・受入、両Linux lock、実cfgとpinned sourceも確認してください。
- 想定起点は `e3c145a096e9223804684b1e3510bba30f0f6676`。HEAD・branch・履歴・stage済み差分・未追跡を確認し、進んでいてもresetせず保護してください。2d0e003やf3e3374へ戻さないでください。
- 新spec・この公開指示・新handoff・spec indexは未commitの場合があります。保護し、今回の実装と同じ新commitへ含めてください。実装先はルートで、歴史コピー `adaptive_drone_patrol_20261013/` や個人ファイル・`.DS_Store` を混ぜないでください。
- 旧spec・旧handoff・旧証拠・failed profile・失敗checkpointを履歴として残してください。旧 `model_2000.pt` は学習初期値や再開に使わないでください。

## まず物理を修理する

- 前回V10の実source・raw・archive・完全receiptのhashを確認し、失敗後のCPU修正版と区別してください。元ファイルが不足していれば記録し、捏造しないでください。
- 大角度14/16の受動energy増加、床8件、dt pair0/6をそれぞれ調べてください。native ballが動いたこと、9anchor・15motor・30reset・547CPU成功を運用域合格へ読み替えないでください。
- 0.30 m固定リンク、実荷物、実mass/inertia/collision、受動関節、機体への反力・反モーメントを保ち、pinned Genesis1.4.3内でsourceに根拠のある修理候補を調べてください。原機体の実mass・rotor位置/spin/KF/KM/RPM/root意味を再検証してください。
- probe前にsource/settings/ケース/初期状態/閾値/timeoutを宣言し、まず小さい原因診断、次に受動全caseの独立checkを行ってください。受動失敗のままpowered全suiteや学習へ進めないでください。
- 新specの62trace、motor15、30回以上の実partial reset、幾何・反作用・仕事・quaternion・有限だが異常なstateを独立再構成してください。rawはclip/reset前に保存し、異常前のwindowも残してください。
- integrator/精度/batch/sourceを変更したら物性と仕事上界の前提を再検証してください。新receiptとactual training configurationが一致して全gateが成立するまでoptimizerを許可しないでください。status手編集・checker弱化・旧PASS差替えで解除しないでください。

## 高度の提案は条件付きで扱う

- 既定は元の全精度配置、delta_z=0です。受動energyや制御不良を隠すためには上げないでください。
- 修理後のdevelopment診断で幾何/過渡余裕不足が実証された場合だけ、新specに従いSと全10地点を同じdelta_zだけ上げることを許可します。既定候補0/0.25/0.50/0.75/1.00 mを事前宣言し、根拠のある最小値を両学習より前に固定してください。この範囲内の通常判断について追加承認は不要です。
- x/y・相対z・重要度・元乱数drawは保ち、初期状態・curriculum・validation・254case・HUD/camera・routing入力/品質を整合させてください。旧filesを上書きせず、base hashとtransformを持つ新layout/wrapperを作ってください。
- 受入失敗後に高度やseedを変更して救済しないでください。1 mを超える変更、地点別配置変更、床/settling/timeouts緩和などは根拠と具体的な修正specを示し、その変更だけ明示確認してください。

## 真の並列化と公平な学習

- 同時1 GPU jobを保ったまま、内部のN環境を増やす方式を調べてください。現sourceはnum_envs=1、単一bank/target/history/RNG/traceを前提とするので、数字だけ変えないでください。
- 物理資格を得たbackendでN=1/8/32/128を有限予算で計測し、実memory・初期化・simulation/inference/reset/logging・PPOを含めて選んでください。22%から速度倍率を推定せず、実際の有効遷移/sを使ってください。loggerを切った値だけで本番予算を見積もらないでください。
- environmentごとの実荷物geometry/mass/inertia、全stageの独立episode抽選、state/history/dwell/timer/RNG、partial reset、無効slot隔離を実測してください。batch内の異常を黙ってresetしてPPOへ混ぜないでください。
- S=N*T*Kで総環境遷移数を数え、vector step/optimizer iterationと区別してください。defaultは両方策各300000総遷移。端数や予算変更は新specの共通事前固定手順で扱い、同じ初期値・seed7・N/T・architecture・optimizer・reward・curriculumで比較してください。
- 30万遷移のままN128にすると各環境の平均経験は約23.4秒です。速さだけでNを選ばず、長距離・連続目標・各stageの実経験を確保してください。rollout境界でepisodeや荷物をresetせず、不足するならNを下げるか両方の共通予算を事前変更してください。正しさ・速度・経験の条件を満たす並列数がなければN1を維持してください。
- 並列化によるthroughput改善とpolicy品質改善を区別してください。実測余裕でsampleを増やすなら、両最終学習より前に同じ予算と全254評価・撮影の余裕を固定してください。pilotから最終学習を再開せず、両方とも正しいmodel4296移植とfresh optimizerから始めてください。
- actorへmass/size/inertia/bank ID/seedを入れず、29観測の意味・scale/clip・実action timingとreset/retargetの履歴を再確認してください。意味変更は新contractと明示移植が必要です。

## 実装から評価・動画まで

- 未完のnative共通runtime、validation、acceptance、capture/HUD統合も実装してください。学習済みcheckpointを旧Euler runtimeへ流用しないでください。独立CPU作業はGPU中も進めてください。
- training/development/validation/held-outを分離し、validation12条件/120実遷移の後でモデル・設定を凍結してください。両方策それぞれ基本9・端点18・held-out100、合計254を新規実行し、randomized9/9・18/18・95/100以上を独立判定してください。
- 失敗・分母・subgroup・Wilson95%区間を保存し、固定比較の方が良ければそのまま報告してください。全10地点で機体と荷物の全条件を同時1秒満たして訪問完了、床>0.10 m・傾斜60度・leg30秒/mission120秒/wall600秒を維持してください。
- 重要度seed40029、A81/B42/C43/D64/E23/F84/G6/H98/I22/J56、alpha1/beta.05、除数100、有向倍率1〜5を維持してください。最初の新実完了後のC43→100、F→Aだけ1→5を維持し、新しい実位置・残タスクからQ*/J*/suffix/反実仮想/成分コストを再計算してください。
- 合格した採用randomized方策、基本seed1・nominal荷物で通常/緊急/コスト変化の全編rawと日本語HUD3本を撮影してください。全10地点のshared settlingまで省略せず実測1x、機体・荷物・リンク・全ID/重要度/状態/目標/両高度/荷物条件/揺れ/進捗を読めるようにしてください。
- 全frame decode、時刻/元画素/字幕/telemetryを照合し、最終hashと具体的イベント前〜次shared settling区間を示して新しい1倍速連続視聴を私へ依頼してください。未確認はpendingです。旧承認は使わず、待機中もCPU検証と文書を進めてください。

## GPU、成果物、commit

- 現累計10824.12788819254秒、残り17975.87211180746秒。開始時に両台帳と新しい正当な使用量を照合し、8時間をゼロに戻さないでください。新specのphase配分から始め、実測に基づき全後段が入るか確認してください。初期化・失敗・再試行・pilotも全計上してください。
- 新workspace・実験名・一意な `outputs/suspended_recovery_<date>_<unique>/` とPython3.12隔離flight/CPU routing環境を使い、既存lock版を照合してください。古い環境/絶対pathを流用せず、外部成果物をローカルへ転送してサイズ・SHA-256を照合してください。
- 物理未修復、予算不足、サーバ不可なら依存する学習・受入・撮影は止め、独立CPU成果・全失敗・具体的次案・文書・ローカルcommitを保存してください。成功を装わず、評価件数を減らして合格にしないでください。
- コード/comment/docstringは英語、説明・字幕・報告・再実行案内は日本語。README、report、影響PDFを更新し、全PDFページを描画確認してください。新handoffとcompact evidenceに全gate・未達・hash・全試行・比較・GPU・動画・人手確認を記録してください。
- 新spec・この公開指示・新handoff・索引・コード・テスト・必要文書・compact evidenceを1件の新ローカルcommitへまとめてください。Author/Committerは `dstaka <22057509+dstaka@users.noreply.github.com>`、global設定変更なし、amendなし。stage内容を直前確認し、モデル/動画/大量log/個人ファイル/接続情報/絶対pathを混ぜないでください。
- push、PR作成/マージ、動画公開、コース提出、統合demo_fullや提出用再編集は行わないでください。最後は日本語で全gate・未達・GPU累計/残量・commit IDと、新specリンク付き英文PR descriptionを報告してください。
