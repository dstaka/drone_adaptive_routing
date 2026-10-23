# 新規Codexセッションへ貼り付ける実行指示

対象リポジトリのルートで新しいセッションを開き、下の「実行本文」を送ってください。現在有効なGPU接続コマンドは別の非公開メッセージで渡してください。この公開ファイルへ接続情報や機械固有の絶対pathを書かないでください。

このファイルを作成・レビューするだけのセッションは実装・SSH・GPU作業を開始しません。人が実行本文を新セッションへ送信すると、実装開始指示になります。

---

## 実行本文

このリポジトリのルートで、`specs/2026-10-22-suspended-payload-solver-recovery/spec.md` に基づき、吊り荷飛行の数値修理・実装・学習・テスト・254件の比較評価・全編動画撮影まで進めてください。

最初に新spec・handoff・計画を読み、現状に合わせた実装計画を提示してください。その後は実行を続けてください。新spec内の通常判断について計画承認を改めて待つ必要はありません。計画提示だけで終了しないでください。実行方法はこのセッション内での実装を基本とし、独立したCPU調査・レビューを並行させる場合も、GPU実行は同時1jobに保ってください。

### 最優先の変更：GPU時間上限とtrackingの撤廃

今後のGPU使用に時間上限はありません。必要なだけ使用してください。累積使用時間・残量・phase配分のtrackingは全く不要です。

旧spec・旧handoff・旧指示・実行helperに残る8時間枠、残量照合、GPU台帳更新、後段が時間枠へ収まるかによる停止条件は、新specとこの指示で撤廃します。旧ledgerは履歴として残し、更新も新しいledgerの作成もしないでください。古いBudgetForecastやledger枯渇で新経路が止まらないよう実装・テストしてください。GPU使用時間を最終報告する必要もありません。

実験の設定・source・失敗・raw・学習sampleの記録は必要です。数値物理のsimulation時刻、missionのwall600秒、実測throughput、1x動画の時刻はGPU使用時間trackingとは別です。同時1GPU jobと暴走/deadlock停止は維持してください。新規レンタル・契約変更・追加サーバ・サーバ破棄は許可していません。

### 起点・読込・保護

- 想定HEADは `e39cb098a9edc47007c5d427ae0d220939075dda`。HEAD/branch/history/staged/unstaged/untrackedを確認し、進んでいてもresetしないでください。
- `AGENTS.md`、新spec全文、`docs/suspended-payload-solver-recovery-handoff.md`、`docs/superpowers/plans/2026-10-22-suspended-payload-solver-recovery.md` を読み、前回recovery・retry・元payloadと先行5spec/handoff、分散配置の採用/失敗/互換性/受入、実cfg/locks/sourceも確認してください。
- 新spec・計画・この指示・新handoff・spec indexは未commitの場合があります。保護して今回の実装と同じ新規commitへ含めてください。歴史コピー、個人PNG、`.DS_Store`等を混ぜないでください。
- 旧証拠・failed profile・失敗checkpointは残してください。旧model2000は初期値・再開に使わないでください。実source archiveと最終CPU sourceを区別し、不足artifactを捏造しないでください。

### 接続と新環境

SSH、必要な転送、新しい隔離環境の構築、新spec範囲のprobe・検証・学習・評価・撮影を許可します。現在有効な接続コマンドは別の非公開メッセージで渡します。GPU/remoteが必要になるまで接続しないでください。未提示・無効なら接続方法を尋ね、独立CPU作業を続けてください。

新workspaceと一意な `outputs/suspended_solver_<date>_<unique>/`、Python3.12の隔離flight/reference/CPU routing環境を用意してください。初期は既存lockを照合し、採用するpatch/更新/backendには新しいlockと実source/patch/build hashを保存してください。古い環境を流用せず、外部成果物をローカルへ転送してsize/SHA-256を照合してください。

### 数値修理と早い実飛行確認

- 同じ実機体、0.30m固定リンク、実荷物のmass/inertia/collision、受動関節、機体への反力・反モーメント、元motor権限を保ってください。
- まず旧代表失敗と独立平面診断を確認し、source/settings/初期状態/入力/閾値/timeoutを宣言した小診断で候補を選んでください。最初から全suiteを盲目的に繰り返さないでください。
- 対応する積分器・精度・dtの比較に加え、根拠のある対象限定engine patch、必要な依存版更新、MuJoCo等の参照計算または別backendへの移行を許可します。新spec内で同じ実物理を保つこれらの通常判断について、改めて承認を求める必要はありません。
- 参照計算自体の収束を確認し、短時間の状態/加速度、energy/work、拘束と3刻み幅を比較してください。長時間の大角度軌道差だけを万能な正誤判定にせず、約13%の未説明energy損失も無視しないでください。新specの受動energy収支5%条件を独立検査してください。
- 受動32全caseを独立に通してからpoweredへ進んでください。その後、診断controllerで実backendのhover・短往復・retarget連続性を先に確認してください。
- 全62trace、motor15、30回以上の実reset、物性・仕事・raw/sourceの完全資格が実装と一致するまでoptimizerは禁止です。status手編集・checker弱化・旧PASS差替えでは解除しないでください。
- 完全資格後、N1で小さい学習hover/往復を実測してください。並列化の完成を待ち続けず、実際の飛行成立を早く確認してください。rawはclip/reset前に保存し、異常前windowと例外時の実getterも保持してください。

### 並列化と学習量

- N1/8/32/128は実geometry/物性/episode RNG/state/history/target/dwell/reset隔離を測定して選んでください。数字だけ変更せず、inactive/異常slotをPPOへ混ぜないでください。選択batchは完全資格を取り直してください。
- 速度は記録込みの実有効transition/sで比較し、22%稼働率から倍率を推定しないでください。正しいbatchが得られなければN1で続けてください。
- 30万遷移は学習の上限ではありません。developmentで両方式の30万→100万→300万を出発点に学習曲線と実hover/往復/連続目標を評価し、必要なら事前宣言した次rungへ共通に増やしてください。GPU残量による増量制限はありません。
- S=N*T*Kで個別環境の総有効遷移を数えてください。vector step/optimizer更新/episode経験を区別し、rollout境界でbody/荷物/historyをresetしないでください。
- development内の共通reward/curriculum/PPO調整は許可します。両最終学習前にS、N/T、seed7、architecture、optimizer、reward、curriculum、初期tensor、layout、checkpoint選択規則を同一freezeしてください。
- 両最終方策は正しいmodel4296移植とfresh optimizerから開始し、pilotや失敗model2000を初期値にしないでください。29観測・current-action-v2・4motor actionの意味を保ち、actorにmass/size/inertia/bank ID/seedを入れないでください。sample計画の再変更が必要なら両最終方策を同じ新計画でやり直してください。

### 受入と動画は維持

既定高度はdelta_z=0。物理修理後のdevelopmentで余裕不足が実証された場合だけ、Sと全10地点を共通に0/.25/.50/.75/1mから最小根拠値へ移動して両最終学習前に固定してください。元draw・x/y・相対z・重要度を保ち、初期状態・curriculum・validation・254case・routing・HUD/cameraまで整合させてください。

validationは12条件/120実target遷移。両model/source/設定/casesをfreezeした後、両方策それぞれ基本9・端点18・held-out100、合計254を新規実行してください。randomized9/9・18/18・95/100以上を独立判定し、固定比較が良ければそのまま報告してください。失敗・分母・subgroup・Wilson95%区間を保存してください。

全10地点で機体と荷物の全条件を同時に1秒満たして訪問完了としてください。床>.10m、roll/pitch60度、leg30秒/mission120秒/wall600秒を維持してください。重要度seed40029、A81/B42/C43/D64/E23/F84/G6/H98/I22/J56、alpha1/beta.05/除数100、有向倍率1～5と、最初の新完了後のC43→100/F→Aだけ1→5を維持し、実位置・残タスクから説明を再計算してください。

合格したrandomized方策、基本seed1・nominal荷物で通常/緊急/コスト変化の全編rawと日本語HUD3本を撮影してください。全10地点のshared settlingまで実測1xで記録し、全frame decode・元画素・字幕・telemetry・時刻を照合してください。最終hashとイベント前～次shared settlingの具体的区間を示し、私へ新しい連続1x視聴を依頼してください。旧承認は使わず、未確認はpendingのまま、待機中もCPU検証と文書を進めてください。

### 範囲外の判断と保存

制御器＋学習補正への変更、観測契約の変更、角度範囲/荷物分布の縮小、物理モデルやmotor権限変更、床/settling/timeoutsの緩和、1m超/地点別配置変更、評価件数削減は今回の通常判断に含みません。必要な場合は根拠と具体的な修正specを示し、その変更だけ明示確認してください。

物理失敗は依存する学習・評価・撮影を止めますが、根拠のある次の設定・patch・backend候補を調べてください。時間がかかることやN1であることだけで終了しないでください。試した選択肢と具体的な未解決条件から、許可範囲で進められないと分かった場合は、全失敗・CPU成果・次案を正直に保存してください。

README・日本語report・影響PDFを更新し、全PDFページを描画確認してください。新handoffとcompact evidenceに全gate・未達・hash・全候補/試行・学習量・比較・動画・人手確認を記録してください。コード/comment/docstringは英語、説明・字幕・再実行案内は日本語としてください。

新spec・計画・この指示・新handoff・索引・コード・テスト・必要文書・compact evidenceを1件の新規local commitへまとめてください。Author/Committerは `dstaka <22057509+dstaka@users.noreply.github.com>`、global設定変更なし、amendなし。stageを直前確認し、モデル/動画/大量log/個人ファイル/接続情報/機械固有絶対pathを混ぜないでください。

push、PR作成/merge、動画公開、コース提出、demo_fullや提出用再編集は行わないでください。最後は日本語で全gate・未達・実学習量・commit IDと、新specリンク付き英文PR descriptionを報告してください。GPU使用時間・残量の報告は不要です。
