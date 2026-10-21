# 吊り荷飛行 実装計画

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. The operator explicitly authorized execution in this session without another plan approval. Produce one final local commit.

**Goal:** 固定長0.30 m受動リンクで実荷物を吊るし、両方策を同一127条件で比較し、独立検証した全編3本を保存する。

**Architecture:** 無荷物経路を維持し、吊り荷の物理adapter・観測・共有settlingを別profileへ分離する。物理probeを学習の必須前提とし、成立しない場合は失敗を保存してCPU実装・文書を継続する。ルーティングは既存CPU processと整数契約を継承する。

**Tech Stack:** Python、Genesis 1.4.3、PyTorch/RSL PPO、既存CPU routing lock、pytest、FFmpeg、ReportLab。

**Spec:** [2026-10-21更新版](../../../specs/2026-10-17-suspended-payload-flight/spec.md)

## Global Constraints

- 起点f3e33747ebd62d5e5e0466cb671a1b39069d182f。ルートで作業し、既存stage・未追跡の個人ファイル・歴史資料を保護する。
- 配置seed51000/candidate3353、重要度seed40029、routing seed44、alpha1/beta.05、有向F→Aのみ倍率1→5を維持する。
- 3寸法.030/.045/.060 m等確率、質量比.10〜.30独立一様。固定比較は.20/.045。同一episode内では不変。
- 機体位置.20 m/速度.30 m/s、荷物位置.20 m/速度.10 m/s、揺れ5度/10度毎秒、荷物角速度10度毎秒を同時1秒。機体傾斜60度、双方floor.10 m。
- leg30/mission120/wall600秒。GPU同時1job、累計28800秒、開始5779.474017477245秒。閾値・分母・失敗seedは変更しない。
- コード・コメントは英語、説明・文書は日本語。大規模成果物は一意なignored出力へ。push/PR/公開/提出は禁止。

## Review Focus

- 同じshapeで異なる観測意味を受理しない。実cfgのscale/clip/history/モデルsignatureを照合する。
- turning point、非有限値、sample欠落、target差替え、ground contactで共有settlingを通さない。
- 部分resetで他envの荷物・乱数列を変更しない。mass変更とinertia/geometry変更を区別する。
- stale receipt・モデルhash・telemetry差替え・静止hoverのラベルだけで受入を通さない。
- 環境の作成失敗と物理的実現不能を混同しない。未実行はFAIL実測ともPASSとも報告しない。

### Task 1: 保護・前提・予算・依存環境

**Files:** 対象spec、`docs/evidence/suspended/inputs.json`、ignored実行環境/出力。

- [x] Git履歴・stageを確認し、入力specの原本とstage差分をignored出力へ保存する。
- [x] 採用3artifactの実サイズ/hashと両台帳を照合する。
- [x] サーバguide全文を読み、新規workspaceとflight/CPU各venvをlockで構築・照合する。失敗は全logを保持する。
- [x] pinned実装のDrone rotor番号、force/torque適用先、root意味、関節とper-env property更新を実ソースで調査する。

予算配分（上限秒）: feasibility1800、randomized training4200、fixed training4200、validation1800、254評価7000、撮影2500、GPU検証1000、予備520.525982522755。環境構築はGPUを初期化しないCPU作業。すべてのGPU初期化以降を既存supervisorで計上する。

### Task 2: 独立CPU契約と固定case

**Files:** `src/adaptive_drone_patrol/payload_contract.py`、`payload_cases.py`、`tests/test_payload_contract.py`、`tests/test_payload_cases.py`、`scripts/prepare_payload_cases.py`。
**Interfaces:** `PayloadSampler.reset(env_ids)`、`payload_metrics(sample, target, payload)`、`SharedSettling.update(time_s, valid)`、`build_cases()`。

- [x] 不正質量/寸法、独立stream・部分reset、turning point・spin・回転cube床余裕、連続時間・target固定を失敗試験で確認する。
- [x] 観測に質量/寸法等を含めない版管理contractを実装し、cfg不一致を拒否する。
- [x] 基本9・端点18・held-out100（各size/scenario11＋small normal1）のpaired manifestを新規生成・hash固定する。
- [x] CPU試験を実行し、物理成功とは明確に区別する。

### Task 3: 物理実装と事前probe（Task 1のpinned環境が前提）

**Files:** 分離したpayload Genesis adapter、URDF生成器、probe runnerとtests。

- [x] nonzero mass/inertiaの固定リンクとpassive二方向関節、cube collisionを実装する。
- [ ] rotor/root、固定anchor振り子、coupled反力、全6端点、dt/substep収束、部分reset隔離を実測する。
- [ ] 全目標と初期擾乱の地面余裕、数値有限性、拘束残差、force観測の可否と圧縮近似を保存する。
- [ ] 全probeの実測合格後に限りTask 4へ進む。成立しなければ修正案を保存し、独立CPU作業へ進む。

### Task 4: 学習・validation・固定評価（Task 3合格が前提）

- [ ] 同一初期値・architecture・optimizer・step数の両方策。旧17入力を新入力へ移植し追加列を0初期化、optimizerは新規。
- [ ] hover/短長距離/反転/上下/同時変更/連続target curriculum。全stageで荷物分布を維持する。
- [ ] training/validation/held-out独立stream。clip前後、飽和時間、荷物最下点、失敗前rewardを記録する。
- [ ] validationのみで両モデルを選択し、実source/config/checkpoint/事前遷移全telemetry/camera/閾値をhash凍結する。
- [ ] 両方策の127件ずつ全試行を実行し、独立checkerで10地点共有settlingを再構成する。9/9・18/18・95/100を別判定しCI・subgroupを報告する。
- [ ] 新しい実判断時点からQ*/J*/policy suffix/成分を再計算する。

### Task 5: 動画・文書・検証・1 commit

**Files:** `README.md`、`docs/report_ja.md`、`docs/suspended-payload-flight-handoff.md`、PR description、証拠、spec index。

- [ ] 採用方策の基本seed1で完全raw＋日本語HUD全編3本。実フレームの機体/荷物/link/ID/高さと圧縮字幕を確認する。
- [ ] 全decode/時刻/元画素/telemetry対応を独立検証し、最終hashとイベント前〜次settlingの区間を提示して新規人手確認を依頼する。未確認はpending。
- [x] 日本語レポート・影響PDFを新出力へ生成し全ページを描画確認する。
- [x] CPU全試験/lint/compile/diff、必要な先行gate、転送size/hash、台帳を確認する。
- [ ] stageのscope・個人情報・秘密情報を点検し、指定identityで1件の新しいローカルcommit。amend/pushなし。


## 実行結果（2026-10-21）

Task 1・2の保護、lock構築、pinned調査、CPU契約、case宣言を完了。Task 3の短時間13 probeと30 resetは限定範囲で通過したが、Task 4のランダム化学習で数値破綻し、2/8/16 substepでも再現した。Task 3の運用域ゲートは最終的に不合格で、短時間PASSの再利用を禁止した。Task 4は途中停止、固定比較・validation・254受入・判断分析は未実行。Task 5の動画と人手確認は未達、独立CPU検査・証拠回収・文書/PDF・1件のローカルcommitを実施する。チェック未完了は合格へ読み替えない。[詳細](../../suspended-payload-flight-handoff.md)。
