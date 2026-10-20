# PR単位の仕様書

各仕様書は、その変更を実装するコード・テストと同じPRに含めます。PR作成時に仕様書内へPR番号またはURLを追記し、PR本文から仕様書にリンクしてください。マージ済み仕様書は履歴として保持します。

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
