# PR単位の仕様書

各仕様書は、その変更を実装するコード・テストと同じPRに含めます。PR作成時に仕様書内へPR番号またはURLを追記し、PR本文から仕様書にリンクしてください。マージ済み仕様書は履歴として保持します。

**2026-10-27・2026-10-29にユーザー指示で一時停止（受入未達）：** [吊り荷付き2機の協調判断をNNで学習する](2026-10-27-dual-drone-learned-coordination/spec.md)。中央のjoint policyが目的地・動的な担当変更・待機・交差時の優先機を選び、既存飛行制御器と安全監視を組み合わせます。主成果は実際の全体巡回時間の短縮と、担当変更／譲り合いの学習効果です。3seedの有限学習、固定validation300、joint freeze後のrouting held-out300と別の物理127、GPU/CPUの実測選択、最終3動画の新規人手確認と成果物回収を規定しました。数値・構成は今回のユーザー指示で採用済みです。学習・高速化・受入成功はまだ主張していません。以前のマージ済みspecと成果物は保持します。

停止時点は[一時停止報告・再開手順](../docs/dual-coordination-pause-ja.md)を参照してください。

同spec §11.1–11.2に、次回セッションの工程・試行別wall-clock計測、実行／待機／停止の区別、並列時間の重複排除、最終振り返り、およびsidechatから読めるローカル進捗記録を追加しました。未計測時間は `not recorded` と明記し、過去の所要時間を推測で補完しません。

**実装・評価・動画確認・回収照合済み：** [目的地の高度引き上げ・高低差拡大と右側詳細カメラのズーム](2026-10-25-elevated-payload-destinations-detail-zoom/spec.md)（Revision2）。新高度2.13〜4.87m、右detail距離0.90m、既存Masked PPOと3D追加学習3seedの有限比較を実装しました。固定300validationでF43を選定し、joint freeze後の未使用routing300でB比平均cost2.9447%改善・絶対gate合格。別の物理127件は全件独立監査合格、最終3動画14526frameの機械検証が完了しています。exact hashに結合した最終3本の新規全編1倍速人手確認も合格しました。必要成果物全体をローカルへ回収し、送信元59788files／36851003632bytesの全サイズ・SHA一致を確認しました。[日本語結果](../docs/elevated-payload-result-ja.md)、[handoff](../docs/elevated-payload-handoff-ja.md)、[英文PR description](../docs/elevated-payload-pr-description.md)を参照してください。旧成果物・マージ済みspecは保持し、今回もlocal commitのみです。

**完了済みの基盤（PR #7、mainへマージ済み）：** [MuJoCo制御器単独で完成させる吊り荷輸送](2026-10-23-suspended-payload-hybrid-genesis/spec.md)（Revision3）。3シナリオの開発10地点完走を根拠にMuJoCoを最終主系とし、Genesisの追加実験とNN学習・C/H比較を必須成果から外しました。Genesisの中断・限定資格・失敗・未確認事項は[保存記録](../docs/suspended-payload-genesis-interruption.md)に残し、不可能性や完了とは読み替えません。以下はこの完了済み基盤の要件・履歴です。

荷物速度0.15m/s・振れ角速度15度/s・箱全角速度30度/sと、位置・傾き・連続1秒の動画優先基準を維持します。合計120秒／12000step制限はなく、各leg30秒・wall600秒を維持。source一致による既存MuJoCo資格とC0/C1/C2を再利用し、開発頑健性15条件、公称＋6端点330秒、validation12条件／120指示leg、正式127条件（basic9/9・端点18/18・held-out95/100以上）、改善画角の最終3動画とfresh人手確認を完了条件とします。継承validation列の120legは96異座標変更と24保持を含み、120回の異座標移動とは呼びません。

[実行記録](../docs/suspended-payload-hybrid-progress.md)、[結果報告](../docs/suspended-payload-mujoco-result-ja.md)、[現行計画](../docs/superpowers/plans/2026-10-25-suspended-mujoco-completion.md)を参照してください。開発動画と正式受入・人手確認の完了は区別します。旧直接motor学習は共有settling未達のまま保存済みです。以下のGenesis優先・旧GPU予算・静定条件・mission制限などは過去の記録です。

## 2026-10-14: 適応型Drone巡回デモの改善案

以下は2026-10-14時点の分割計画です。最新の実装記録は末尾に記載します。

目的は、**環境変化に応じて同じAI Policyが巡回順序を変更し、Droneが全地点まで安定して飛行すること**です。当日の成果物は仕様書で、各実装は後続セッションで実施しました。

参照元は [既存の仕様書](../adaptive_drone_patrol_20261013/spec.md) と、その配下の現行実装です。ルートの [初期仕様書](../spec.md) および参照元は変更しません。新仕様は英語、利用者向けの説明・動画字幕は日本語を維持します。

| 実装順 | 仕様書（1ファイル＝1 PR） | 主な成果 | 依存 |
| --- | --- | --- | --- |
| 1 | [10地点以上の経路最適化と3シナリオ](2026-10-14-scalable-adaptive-routing/spec.md) | 地点数依存のスキーマ、再学習、C地点の緊急点検、移動コスト変化、厳密解による訪問順序の比較 | なし |
| 2 | [安定飛行と全目的地の常時表示](2026-10-14-visible-multigoal-flight/spec.md) | A〜Jの実飛行、全地点表示、色と数値による優先度表示、到達・hovering判定 | PR 1 |
| 3 | [1倍速動画と統合デモ成果物](2026-10-14-realtime-demo-video/spec.md) | 1倍速動画、10地点対応HUD、距離と優先度の判断を説明する総コスト比較、成果物検証 | PR 1・2 |

### 分割の理由

1 PRにまとめる案では、モデル・物理実行・動画編集の不具合を切り分けにくくなります。2 PR（経路判断／飛行と動画）でも実装できますが、後半にGPU検証と動画編集の検証が集中します。上記の3 PRを推奨します。仕様の準備は今回まとめて行い、各仕様は対応する実装PRで追加してください。仕様書だけを先にすべてマージする運用は想定しません。

### 仕様で採用した前提と判断

- 必須デモは10地点（A〜J）。データ構造は地点数から導出し、12地点でもCPUで検証します。地点数が異なるモデルの再学習は許容します。
- 再計画は既存の「到達・点検完了後」の境界で実施します。飛行途中の即時割込みは今回の対象外です。
- BはC地点の優先度上昇、Cは有向移動コスト行列の更新です。Scenario Cの「C」と地点Cは別の意味です。
- 移動コストは論理的なペナルティです。風・障害物や物理速度の変更までは含めません。
- 「最適」は目的関数と厳密解との差で評価します。学習Policyが必ず大域最適になるとは仮定しません。数値ゲートは仕様内に提案値として定義しました。
- 各地点の優先度は固定の薄い赤〜濃い赤の色スケールと数値で表示します。選択中は輪郭、訪問済みはチェック印で区別し、優先度の色を上書きしません。
- 「近い低優先度地点を先に回る理由」は、同じ判断時点から残り全地点を巡回する総コストで説明します。「近い地点を先にした条件付き最適解」「遠い高優先度地点を先にした条件付き最適解」「AIの予測」を比較し、最適な次の一手と巡回全体の最適性を区別します。
- 1倍速はシミュレーション時間に対する速度です。全編動画は尺を固定せず、従来の60〜180秒の提出動画は必要なら省略を明記した1倍速の抜粋で構成します。
- 「安定飛行」は各地点の位置・速度・連続滞在条件と、事前に固定した3シード×3シナリオの完走で検証します。

最終完成は3 PRすべての受入条件を満たした時点です。PR 1のCPU結果だけでは実飛行の成功を、PR 2の録画だけでは完成動画の1倍速を主張しません。実装時の接続情報・残りGPU予算は別途確認し、以前の実行許可や8時間枠を新たな予算として読み替えません。

## 2026-10-15 implementation session

The approved [scalable adaptive routing specification](2026-10-14-scalable-adaptive-routing/spec.md) is implemented in the repository root. See the [CPU workflow](../docs/scalable-routing.md), [implementation plan](../docs/superpowers/plans/2026-10-15-scalable-adaptive-routing.md), and [measured handoff](../docs/scalable-routing-handoff.md). Physical integration and video remain separate follow-up specifications. The operator requested one local commit and a prepared PR description without push.

## 2026-10-15–16 visible flight implementation session

[安定飛行と全目的地の常時表示](2026-10-14-visible-multigoal-flight/spec.md)は承認済み仕様として、このPRでルートの物理実行へ統合します。[実装計画](../docs/superpowers/plans/2026-10-15-visible-multigoal-flight.md)と[実測・再実行の引き継ぎ](../docs/visible-flight-handoff.md)を参照してください。CPUロジック検証、開発飛行、固定9試行の受入結果を分けて記録し、最終編集・リタイミングは後続PRに残します。

## 2026-10-16 realtime video implementation session

[1倍速動画と統合デモ成果物](2026-10-14-realtime-demo-video/spec.md)を承認済み仕様としてルートへ実装しました。[実装計画](../docs/superpowers/plans/2026-10-16-realtime-demo-video.md)、[入力・出力・再実行・検証記録](../docs/realtime-video-handoff.md)、[英文PR description](../docs/realtime-video-pr-description.md)を参照してください。全3シナリオの自然尺動画、96.30秒の提出用、元撮影からの独立PTS・画素照合と日本語PDFを生成しました。2026年10月17日のユーザーによる目視OKを反映し、残っていた人手確認を完了、仕様受入をPASSとして記録しました。改善案は別spec／PRで扱います。

## 2026-10-20 integer priorities implementation session

[1〜100の整数重要度](2026-10-17-integer-priorities/spec.md)を独立した変更として実装しました。[実装計画](../docs/superpowers/plans/2026-10-20-integer-priorities.md)と[整数重要度の引き継ぎ](../docs/integer-priorities-handoff.md)を参照してください。既存配置を維持し、新profileのモデル・データ・9物理試行・1倍速動画を別出力先へ保存しました。全7受入条件を満たし、人手確認・PDF確認も完了しています。分散配置と吊り荷は今回のcommitの対象外です。

## 2026-10-20 dispersed destination implementation session

[分散3次元目的地配置](2026-10-17-dispersed-destination-layout/spec.md)を整数重要度PRのマージ後から実装しました。[実装計画](../docs/superpowers/plans/2026-10-20-dispersed-destination-layout.md)と[新規測定・失敗・再実行の引き継ぎ](../docs/dispersed-destination-layout-handoff.md)を参照してください。配置承認、追加学習、9物理試行90点検、全5動画11,361フレームの独立照合、最終動画の人手確認とPDF確認を完了し、全7受入条件を満たしました。他の未追跡specは今回のcommitへ含めません。

## 2026-10-17: 吊り荷を運ぶ強化学習制御

[吊り荷付き10地点巡回と荷重・寸法ランダム化](2026-10-17-suspended-payload-flight/spec.md)を、動画改善PRとは独立した仕様として作成しました。既存の小型機で荷物を吊り下げ、各地点で荷物も安定するまで待ちます。初版は固定長リンクによる近似から始め、3種類の箱サイズと連続的な質量をepisodeごとに抽選します。方策は荷物の動きを観測しますが、正確な質量・寸法は入力しません。投下とロープの弛みは対象外です。

確認済みの受入は、機体・荷物の同時安定1秒、基本9/9成功、端点条件18/18成功、未学習ランダム100試行中95件以上成功です。固定荷物で学習した比較方策にも同じ127条件を使用します。動画成果物は1倍速rawと日本語HUD付き全編3本までとし、提出用再編集は別PRへ分離します。作成時点では最終レビュー待ち・実装未開始でした。2026-10-21の明示指示により実装・物理probe・一部学習を行い、数値破綻のため停止しました。固定比較・254受入・動画は未達です。[実装と失敗の引き継ぎ](../docs/suspended-payload-flight-handoff.md)を参照してください。

## 2026-10-21: 吊り荷飛行の再挑戦

[長時間物理qualificationを先行する吊り荷再挑戦](2026-10-21-suspended-payload-flight-retry/spec.md)を作成しました。起点は前回の失敗を保存した `2d0e003`。旧PASSを無効のまま保持し、固定リンクモデル内で別の受動関節表現を調査して、120秒の固定anchor・連続モーター駆動、dt比較、reset隔離を学習前に検証します。荷物分布・同時安定1秒・127条件ずつの比較・全編3本の完成条件は維持します。[別Codexセッション向けの実行指示](../docs/suspended-payload-flight-retry-instructions_ja.md)を用意しました。作成時点では文書のみでした。2026-10-22の明示指示後、native ball版・独立checker・新観測/学習guardを実装し、新GPU環境で検証しました。powered診断で荷物床余裕違反が生じ、学習・validation・254受入・動画は開始していません。旧成功の流用や閾値緩和は行わず、失敗と残り診断を保持します。[再挑戦の引き継ぎ](../docs/suspended-payload-flight-retry-handoff.md)を参照してください。新spec・索引・公開指示は今回のコード・テストと同じローカルcommitへ含めます。

## 2026-10-22: 物理修理・条件付き高度変更・並列学習

[次回の吊り荷回復spec](2026-10-22-suspended-payload-flight-recovery/spec.md)を準備しました。起点は `e3c145a`、残GPUは17975.87211180746秒を基準に再照合します。大角度の受動energyをpowered前の独立barrierで検査し、実batch化と総sample/PPO更新の公平性を規定しました。高度変更は物理修理後に余裕不足が実証された場合だけ、Sと全地点の共通変換として両学習前に固定します。未完のruntime・validation・254評価・全編3本・新規人手確認まで対象です。[提案の評価と引き継ぎ](../docs/suspended-payload-flight-recovery-handoff.md)、[新セッションへの実行指示](../docs/suspended-payload-flight-recovery-instructions_ja.md)を参照してください。人による実行指示後に実装・新環境・6GPU診断を行いました。旧失敗rawを完全再現し、積分離散化の独立診断と128/256-substep受動3traceを取得しましたが、完全資格・学習・254受入・動画は未達です。数値品質未確定と実測による予算不足で依存GPU作業を停止し、CPU実装・全失敗・証拠を保存しました。

## 2026-10-22–23: 数値解法の修理と実測に基づく学習

[新spec](2026-10-22-suspended-payload-solver-recovery/spec.md)、[引き継ぎ](../docs/suspended-payload-solver-recovery-handoff.md)、[実装計画](../docs/superpowers/plans/2026-10-22-suspended-payload-solver-recovery.md)、[貼り付け用実行指示](../docs/suspended-payload-solver-recovery-instructions_ja.md)を用意しました。想定起点は `e39cb098a9edc47007c5d427ae0d220939075dda`。準備後の実行指示により数値修理・実装・新規資格・7.02M遷移の学習と全3rung比較を実施しました。数値資格は通過しましたが共有settling0・全rung両方式0/6のため、2026-10-23のユーザー指示で未達として区切ります。

ユーザーの新指示により、今後はGPU時間上限・使用時間tracking・旧残量による停止を撤廃します。同じ実物理を保つ設定/積分器/精度、対象限定engine patch、必要な版更新、別backendの検証を許容し、developmentの実測で共通の最終学習量を決めます。実N1での小さい飛行成立を早く確認し、並列化を待ち続けない計画です。物理・同時安定・254受入・3本の実測動画は維持し、制御方式や角度範囲・mission時間等の追加緩和は別承認にします。準備ファイル群を今回の実装・テスト・未達証拠と同じ新規local commitへ含めます。次のハイブリッドspecの実装は未開始です。
