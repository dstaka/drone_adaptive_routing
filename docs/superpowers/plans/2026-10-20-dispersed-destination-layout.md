# 分散3次元目的地配置 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans in this session. Steps use checkbox syntax. The operator explicitly authorized implementation; no additional plan-approval pause. One final local commit replaces per-task commits.

**Goal:** 整数重要度を維持して実目的地を分散3D配置へ変更し、新しい物理・動画証拠を作る。

**Architecture:** 配置専用の決定的乱数列と独立checkerで候補を生成し、採用済みpolicyの説明条件で検索する。配置manifestをrouting・Genesis・表示の共通入力とし、人手確認、遷移評価、受入凍結、9物理試行を順に行う。過去の証拠は変更しない。

**Tech Stack:** Python / NumPy / MaskablePPO CPU / Genesis GPU / rsl_rl / Pillow / ffmpeg / ReportLab.

**Spec:** [承認済み仕様](../../../specs/2026-10-17-dispersed-destination-layout/spec.md)

## Global Constraints

- 起点92a2a4d。作業先はルート。未追跡対象specを保護し最終commitに含める。別spec・個人ファイルを除外。
- S=(0,0,1), A–J。x=[-2.7,2.7], y=[-1.8,1.8], z=[1,3] m。実span>=4.86/3.24/1.80 m。
- x/yのソート後連続3点のspan>0.10 m。高さ3帯に各1点以上、内部境界は上側、z=3は最上帯。
- 3D間隔>=0.50 m、S距離>0.20 m。非有限値拒否。投影間隔・遮蔽を別検査。
- 重要度seed40029、81/42/43/64/23/84/6/98/22/56。正規化100、alpha=1、beta=.05、有向倍率1–5、Cのみ100へ更新。
- 到達半径.20 m、速度<=.30 m/s、保持.20 s、点検.50 s、安全制限を維持。
- GPU予算28800秒、開始累計3198.9943080281373秒。単一GPUジョブ、timeout・全失敗保存。
- 配置確認前に受入凍結しない。固定execution seed1/2/3×3シナリオ。未確認をPASSにしない。
- コードは英語、説明は日本語。公開参照は相対パス。大規模成果物はignoredの一意な出力先。

## Review Focus

- 固定bin境界を跨ぐ3点帯・高さ境界の誤分類をTask1で検査。
- 3D分離しても投影が重なる座標をTask1で拒否。
- 旧飛行設定の範囲拡大だけで互換と扱う誤りをTask3で防止。
- 新配置に旧解析数値・物理動画を転用する誤りをTask2/4/5でsnapshot/hash照合。
- 人手確認未了・失敗seed・時間切れの隠蔽をTask4/5で独立した未達記録として保持。

### Task 1: 決定的配置とプレビュー
**Files:** Create `src/adaptive_drone_patrol/dispersed_layout.py`, `scripts/select_dispersed_layout.py`, `tests/test_dispersed_layout.py`, `configs/dispersed_layout.yaml`.
**Interfaces:** `layout_checks(coords) -> dict`, `layout_candidate(seed, attempt) -> list`, manifestはSを含む11×3実座標とgenerator/seed/attempt/checks/hash。
- [x] 制約・再現・非有限・境界・投影衝突の失敗テストを作りRED確認。
- [x] 独立PCG64(SeedSequence([seed,300,attempt]))、連続一様3D抽選、固定規則による棄却、候補ログを実装。
- [x] CPU試験を実行しGREEN確認。実数値は切り丸めず保存。
- [x] 旧配置とのXY/XZ/YZ・高さ比較を生成し、実座標とIDを提示。人手確認を依頼し、回答待ちもTask2/3を進める。

### Task 2: Routing解析と互換性
**Files:** Create dispersed scenario YAMLs and `scripts/evaluate_dispersed_routing.py`; reuse analytical/benchmark/independent verifier.
**Interfaces:** 共通layout manifestから3 scenarioを作成。解析replay・全判断explanationsは既存schema2に適合。
- [x] モデル・設定のサイズ/SHA-256と埋込契約を確認。
- [x] 重要度固定のまま候補座標でC応答・有向倍率1→5・L/H差>=10の厳密説明を検索。全候補棄却記録を保存。
- [x] 新配置分布でvalidation100、static100、dynamic50の固定seedを分離して評価。必要なら新モデルを学習し全先行品質ゲートを再実行。
- [x] 全suffix・Q*/J*・成分・policy経路を標準ライブラリcheckerで再計算。

### Task 3: 飛行範囲拡張と遷移評価
**Files:** Modify flight training configuration/hooks; create dispersed training and transition scripts/configs/tests.
**Interfaces:** 新飛行checkpoint/config/hash・world/relative observation contract metadata。既存の17観測4行動を維持する場合も実遷移で互換性確認。
- [x] 旧cfgの範囲・観測飽和を記録。範囲だけの変更を採用しない。
- [x] 許可サーバの利用可否を確認し、一意workspace・分離venv・両lock照合・転送manifestを作る。
- [x] 採用model_1300を起点に広域と高度1–3の学習を行い、長距離・上昇・下降・同時変更・反転・hover・地面余裕・飽和を評価。
- [x] 開発/validation/held-outを区別し、根拠ある有限timeoutを受入前に記録。失敗全保存。

### Task 4: 凍結と固定物理受入
**Files:** Generated frozen suite, physical evidence and compact acceptance records.
**Interfaces:** 配置承認・モデル対・閾値・カメラ・イベント辺を凍結したmanifest。
- [x] 人手配置確認を受領後にのみ凍結。全10ラベルと高低差を実frameで確認。
- [x] seed1/2/3×normal/emergency/cost_shiftを新規実行。90点検と先行安全条件を独立checkerで検証。
- [x] 全90実判断snapshotからQ*/J*・成分・イベント・L/Hを再計算。失敗を差し替えない。

### Task 5: 動画・文書・完了commit
**Files:** Modify README/report builder; create `docs/dispersed-destination-layout-handoff.md`, compact `docs/evidence/dispersed/`, PR description; update only target spec index.
**Interfaces:** 新raw/full3本/demo_full/submissionとtimeline/hash、日本語PDF、独立検証、人手視聴記録。
- [x] 新3全編1倍速・統合・60–180秒提出用を作成。必要なカットを明記しイベント前後〜次点検を保持。
- [x] 全frame decode・PTS/step/画素・成分を独立照合。主要frameとPDFを表示確認。
- [x] 動画リンク・具体区間を提示し人手確認を依頼。未回答は未確認のまま保持。
- [x] 全CPU試験・lint・diff・秘密情報/絶対パス監査・manifest照合・両台帳を確認。
- [x] 今回だけを明示stageし指定author/committerの新規commitを1件作成。push/PR/公開/提出しない。
