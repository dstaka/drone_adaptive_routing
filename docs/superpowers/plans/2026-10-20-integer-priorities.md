# Integer Priorities Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, as explicitly requested by the operator. One final local commit; no push or publication.

**Goal:** 承認済み整数重要度specを実装し、新規学習・評価・9物理試行・1倍速動画の検証結果を引き継ぐ。

**Architecture:** scenario v3と明示profileを追加し、v2・旧データの解釈を保つ。生成・学習・推論・厳密計算・物理証拠・動画を同じ契約へ通す。検証済み飛行モデルと長方形配置を維持し、routingモデルを新規学習する。

**Tech Stack:** Python、NumPy、CPU MaskablePPO、Genesis、FFmpeg、Pillow、ReportLab。

**Spec:** `specs/2026-10-17-integer-priorities/spec.md`

## Global Constraints

- 重要度は厳密な整数token1〜100。地点ごとに独立・一様・復元抽選し、形状と独立したPCG64乱数列、固定seedを使用。3シナリオ・実行seedで初期値を共有する。
- 重要度観測は100で正規化。alpha=1、beta=0.05、速度1m/s、サービス0.50秒、帰還なし。有向倍率は1〜5。
- 最初の実点検直後に未点検Cのみ100へ更新し、初期Cは100未満。L/H差>=10、既存の厳密解・実pose距離条件を維持する。
- 形状・履歴成果物・他の未追跡spec・個人ファイルを保護。起点 `7da4d23721770613d3ca910a20718ed6fd0e9371`。
- 一意な出力 `outputs/integer_20261020_v1/`。モデル・動画・大量ログはGit管理外。GPU累計28,800秒枠、開始2,803.57274048205秒を引き継ぎ、ジョブ前に台帳を照合する。
- コード・コメント・docstring・PR説明は英語、利用説明・字幕・再実行案内は日本語。公開パスは相対参照。

## Review Focus

- 整数相当float・bool・文字列・非有限値を変換前に拒否する（Task1）。
- 同shapeの旧checkpointをv3に対し推論前に拒否し、IPCでも契約を照合する（Tasks1–2）。
- コストイベントの倍率は1〜5を維持し、学習時の重要度イベントは整数1〜100とする（Task1）。
- 同値最大はID順に扱い、L/H差1〜9では説明条件を満たさない（Task3）。
- 全自動照合が通っても連続視聴が未確認なら人手ゲートをPASSにしない（Task4）。

### Task1: 数値・schema契約

対象: `scenario.py`、`objective.py`、`routing_env.py`、`tests/test_integer_priorities.py`。
提供: `generate_scenario(..., priority_profile=...)`、`Scenario.priority_profile`、profile別`schema_for`。

- [x] 厳密入力、境界、イベント、独立乱数、重複、旧解釈、正規化、成分計算の失敗するテストを先に実行する。
- [x] v3 `integer-100-v1` を実装し、v2既定値を保持。生と重み付き成分を別保存する。
- [x] 関連テストとCPU全体を実行する。

### Task2: 新しい互換モデル

対象: `routing_agent.py`、`routing_teacher.py`、`benchmark.py`、学習/manifestスクリプト、`configs/routing_integer.yaml`。
使用: Task1のprofile。提供: seed42/43/44の新モデルとvalidation凍結manifest。

- [x] profile不一致・教師provenanceの失敗テストを実行する。
- [x] 教師・静的/イベント混合学習・validation・held-outへprofileを伝搬し、入力・ラベル・実ステップを保存する。
- [x] CPU学習、validationだけで凍結、各seedの静的100・動的50例評価を順に行い、既存品質条件を維持する。

### Task3: デモ選択・厳密証明・表示

対象: `analytical.py`、`physical_evidence.py`、`scene.py`、`annotations.py`、`video_render.py`、`video_check.py`、選択スクリプト、整数デモ設定。
使用: 凍結したTask2モデル。提供: 共通抽選fixture、全試行seed/却下理由、独立成分照合、色manifest。

- [x] 差10以上・同値ID順・固定色・整数ラベルの失敗テストを実行する。
- [x] 長方形を維持し、demo40000〜40099から凍結したイベント/最適性条件で検索。すべての抽選と却下を保持する。
- [x] 全ラベル・字幕と独立checkerへprofileを伝搬し、0.05、生と重み付き成分を表示する。
- [x] 完全な方策suffixと独立成分を再生する。

### Task4: 新規物理スイートと1倍速動画

対象: 新環境用実行設定の生成、既存物理/動画スクリプト、コンパクトな証拠。
使用: サイズ/SHA-256照合済み採用飛行モデルとTasks2–3の凍結条件。

- [x] 今回指定されたサーバで新しい作業先と別々の固定CPU/飛行環境を構築し、失敗も残す。
- [x] 選別入力を転送・照合し、その環境に合う実行パスを生成する。
- [x] GPUスモーク後、seed1/2/3×3シナリオを同時1ジョブ・残予算内timeoutで実行。全9試行と元動画を保持する。
- [x] 転送manifestを照合し、物理判断、厳密成分、安定性/連続滞在、イベント条件を再検証する。
- [x] 全編・統合動画を生成。全フレームchecker、実フレーム目視、イベントから次の点検までの1倍速連続視聴を実施し、人手未確認をPASSにしない。

### Task5: 文書・PDF・最終レビューと1commit

対象: README、日本語レポート、互換性文書、`docs/integer-priorities-handoff.md`、英文PR説明、対象specと索引の今回分。

- [x] 実測、失敗、hash、再実行、モデル、GPU累計、未達を記録し、PDFを生成・描画・表示確認する。
- [x] 全CPU試験、lint、独立checker、成果物/差分/プライバシー確認とレビューを実施する。
- [x] 他spec・個人作業の保護を確認し、明示したファイルだけをstage。指定Author/Committerで新規1commitを作成する。

## 実行記録

計画自己確認: 受入7項目をTasks1–5へ対応付けた。今回の明示指示は計画後に同セッションで実装を継続するものであり、計画の再承認は求めない。既存のルートcheckoutは `feature/integer-priorities`。作業先を追加せず、タスクごとのcommitを最後の1件へまとめる。詳細な進行・RED/GREEN・実行判断はこの計画のignored実行台帳へ記録する。

最終結果: 全7受入PASS。ローカル/固定Linux CPU各177件、GPU1件、9物理試行/90点検、独立媒体7545+2741フレーム、人手連続視聴、4ページPDF表示確認を完了。新規1commitの対象は明示allowlistで管理する。
