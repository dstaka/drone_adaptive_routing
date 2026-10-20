# 分散3次元目的地配置の引き継ぎ

[承認済み仕様](../specs/2026-10-17-dispersed-destination-layout/spec.md)を起点 `92a2a4d87cb9e3f6768e698f71e98a33e18c45db` からリポジトリルートへ実装した。[実装計画](superpowers/plans/2026-10-20-dispersed-destination-layout.md)と[コンパクトな証拠](evidence/dispersed/layout.json)を参照する。今回の成果物は `outputs/dispersed_20261020_v1/`。歴史資料・先行spec・採用済み旧モデル/動画・他spec・個人ファイルは保持した。

## 配置と重要度

生成器は `numpy-PCG64-SeedSequence(seed,300,attempt)-uniform-xyz-v1`。配置seed51000、候補3353。各候補で10地点の3次元座標を連続一様抽選し、座標を手修正せず制約で棄却する。重要度用乱数列は独立した既存 `SeedSequence([40029,100])`。初期重要度 A81/B42/C43/D64/E23/F84/G6/H98/I22/J56 を3シナリオと全実行seedで共有する。

| ID | x [m] | y [m] | 地面からの絶対高度z [m] |
| --- | ---: | ---: | ---: |
| S | 0 | 0 | 1 |
| A | -2.257485 | 1.733525 | 1.101699 |
| B | -2.683144 | 0.044095 | 1.269621 |
| C | 1.724141 | 0.623950 | 1.758623 |
| D | -1.720731 | -1.677043 | 2.792510 |
| E | -2.596933 | -1.209621 | 1.394858 |
| F | -0.328243 | 1.188113 | 1.148212 |
| G | -0.303235 | -1.626957 | 2.911967 |
| H | 1.170309 | 1.667050 | 2.895835 |
| I | 2.377433 | 1.052894 | 1.084423 |
| J | 0.918157 | 0.565451 | 2.067850 |

表だけを丸めて表示し、実装・シナリオ・manifestは完全精度の値を保持する。座標SHA-256は `21dd7cf41bf12c592ccdb4bf9c8817352feeafb9597103a3ba626e51a6d910e0`。実spanは5.060577/3.410568/1.827545 m、高さ3帯は5/2/3点、最小3D間隔0.865248 m、Sとの最小距離1.241500 m。x/yのソート後連続3点で閉区間0.10 m帯を検査する。内部高度境界は上側へ割り当てる。独立した標準ライブラリcheckerでも合格した。

公称水平領域5.4×3.6 mは旧領域の9倍。実際の水平凸包面積は12.274250 m²であり、9倍を凸包面積比・実飛行距離比として使わない。

最初の固定ラベル方式による50,000候補検索は全候補棄却。幾何合格484、両イベント条件合格50だが投影上のラベルが収まらなかった。失敗記録は `demo/failed_selection.json` に保存した。幾何条件・最小投影間隔60px・ラベル140×85pxを維持し、引出線で文字位置を分離する方式に再構成して検索し直した。`demo_v2/layout.json` に全3,354候補のhash・理由、生成入力、選択条件を保存した。独立の再実行 `demo_reproduction/` で座標・シナリオhash・全棄却記録が一致した。これは説明条件を満たすよう選択した固定デモであり、任意配置の頑健性を示さない。

2026年10月20日、`layout-preview-v2.png` と座標/高さ一覧を提示した後、ユーザーから「この配置を承認する」を受領した。これは配置の承認である。新しい実飛行frame・動画の確認を別に記録する。

## Routing契約と新しい評価

採用seed44は `outputs/integer_20261020_v1/training/seed_44.zip`、297837 bytes、SHA-256 `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef`。埋込契約とsidecarを照合して使用した。観測 `routing-v3-integer100-n10-324` の座標単位・固定除数2、距離除数4、整数重要度除数100は変更しない。座標観測はBox(-inf,inf)であり、新しい座標を旧範囲へclipしない。意味の変更はなく、旧shapeだけに依存せず実データで評価した。

`dispersed-constrained3d-v1` 分布では、先行のsplit/seed/重要度/倍率を保ち、座標だけを上記幾何制約を満たす各seed最初の候補へ置換する。validation100例と静的held-out100例（20000–20099）、動的50例（30000–30049）を分離し、3seed42/43/44を共通入力で評価した。選択済み44を結果で変更しない。

| model seed | 平均AI/厳密解比 | greedyとの差 | fixedとの差 | 最悪比 | 動的 適用/予定 | 選択変更 | 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.057082 | -31.117771 | -250.183828 | 1.248482 | 49/50 | 8 | 2 |
| 43 | 1.073484 | -26.079101 | -245.145159 | 1.319687 | 49/50 | 6 | 3 |
| 44（継続採用） | 1.078192 | -24.744215 | -243.810273 | 1.397014 | 49/50 | 6 | 4 |

全450 learned episodes完走、無効行動0。全モデルが平均比<=1.10、greedy以下、fixed未満を満たし、routing再学習は不要と判断した。採用seedの最悪絶対gapは115.783557で、常時最適・全イベントで改善とは主張しない。

新配置の解析30判断・225完全順序について、Q*・J*・policy suffix・生/重み付き成分を再計算した。標準ライブラリによる独立照合の最大成分差は1.8189894035458565e-12。新しいコスト変化辺はF→Aのみ、倍率1→5と宣言し、受入前に凍結した。逆方向は変更しない。C緊急は43→100を維持する。物理境界の解析は新規受入ログから再生成し、上のCPU解析を物理実績へ転用しない。

## 飛行互換性・学習・環境

起点の採用飛行モデルは `outputs/visible_20261015/remote/outputs/visible/finetune_v1/model_1300.pt`、475573 bytes、SHA-256 `4a6f3b333735f3c10ed4e51f2ddd3466308cf90b7dfb3bec289b519a084a539f`。同ディレクトリ `cfgs.pkl` は1592 bytes、SHA-256 `25720a5b59b1edd769ed36ecc29be16febde3ad6088dcc6bdd878f71d2f8fff1`。3件ともhandoff/manifestに一致した。

旧設定はx/y±1 m、z=1 m。17観測の相対位置は1/3倍、±1でclipする。ターゲット相対距離による終了制約もx/y±3 mで、新配置の全域遷移には適合しない。validatorだけを広げず、採用済みmotor方策を初期値として3D範囲の追加学習を実施した。観測単位・正規化・4モータ行動・推論physicsを保持し、metadataを `hover17-relative-world-v1` として保存する。新しい最長遷移に必要な相対距離境界をx±6/y±4/z±3へ拡張し、roll/pitch60度・地面0.10 m条件を維持する。新checkpointは新cfgと実遷移検証を一体で扱う。

許可済みサーバへ接続でき、他GPUジョブなしを確認した。新しい作業ディレクトリ名は `dispersed_20261020_v1_01`。絶対パスと接続情報はignoredの `private/` にのみ保存する。新しい `.venv-flight` と `.venv-cpu` を各Linux lockから構築し、飛行169版・CPU33版をすべて照合した。転送payloadとmanifestを保持する。

初回学習はseed7・256環境、採用済みmodel1300から1501 iteration、独立GPUジョブのtimeout3000秒で実行したが、後述の通りvalidationで失敗した。validationはseed11/12、held-outは31/32/33として先に宣言した。長い水平移動、上昇、下降、水平/高度同時変更、反転、地面余裕、観測飽和、反復hoverを実sampleで検査する。到達半径.20 m、速度<=.30 m/s、到達保持.20秒、点検滞留.50秒、leg30秒・mission120秒を継続し、失敗後に緩めない。

## 再実行の入口

成果物の出力先は毎回一意にする。採用モデルと各hash、配置承認、遷移評価が揃ってから受入凍結する。GPUは常に既存台帳の累計を継承し、同時1ジョブ、残予算内timeoutとする。

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=outputs/dispersed_rerun_new/mpl
.venv-routing311/bin/python scripts/select_dispersed_layout.py \
  --config configs/dispersed_layout.yaml --out outputs/dispersed_rerun_new/demo
.venv-routing311/bin/python scripts/preview_dispersed_layout.py \
  --layout outputs/dispersed_rerun_new/demo/layout.json \
  --out outputs/dispersed_rerun_new/layout-preview.png
.venv-routing311/bin/python scripts/verify_dispersed_layout.py \
  --layout outputs/dispersed_rerun_new/demo/layout.json \
  --out outputs/dispersed_rerun_new/geometry-independent.json
.venv-routing311/bin/python scripts/evaluate_dispersed_routing.py \
  --models outputs/dispersed_20261020_v1/models-frozen.json \
  --out outputs/dispersed_rerun_new/routing-evaluation
```

実行環境ではcwdから新しい設定を生成し、起点モデル/新モデル/ソースのhashを照合する。古いサーバ絶対パスを再利用しない。採用した再学習は `finetune_visible_flight.py --domain dispersed --curriculum --iterations 3000 --num-envs 256`。起点設定には採用済みmodel1300/cfgを明示し、`probe_dispersed_flight.py` を別venvのGPU job supervisor経由で実行する。

## 最終化の状態

2026年10月21日に最終化した。配置承認、新規9物理試行90点検、画質修正後の全3本と両イベント区間の人手確認、最終5動画11,361フレームの独立照合が完了した。全7受入条件を満たし、未達項目はない。初回学習・初回encodeの失敗は保持する。


## 追加学習・失敗・採用モデルの確定

広域一括の初回model2800はvalidation seed11の最初の遷移で失敗した。step57、実高度0.091373533 mで地面0.10 mの終了条件へ達した。傾斜30.0773度、相対位置観測最大0.739081で、観測飽和や傾斜上限が原因ではなかった。失敗したmodel/cfg/全57sample/初期実frameを `remote-development-v1/` と最終アーカイブへ保持した。seed12・held-out・固定受入はこのモデルで実行していない。

採用済みmodel1300を再び初期値にし、x/y/zの範囲を4段階 `[1,1,1.5]` → `[1.5,1.2,2]` → `[2.1,1.5,2.5]` → `[2.7,1.8,3]` へ拡張した。各段階750、計3000更新、256環境、seed7。固定学習率3e-5、entropy係数0、gradient上限.3、8秒retarget、20%端点学習を用いた。学習rewardはtarget2、crash−200、hover.5、velocity−.03、angular−.01、yaw.1、altitude−.5、upright−1。ルーティング目的関数のalpha/betaは変更しない。training_manifestとcompatibilityに全設定を保存した。RSLの複数learn呼出しによるiteration表記に従い、最終ファイル名はmodel4296である。

| 用途 | 保存先 | bytes | SHA-256 |
| --- | --- | ---: | --- |
| 新規採用飛行モデル | `outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/finetune_v2/model_4296.pt` | 475573 | `26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431` |
| 対応飛行設定 | 同ディレクトリの `cfgs.pkl` | 1628 | `35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55` |
| 継続採用routing | `outputs/integer_20261020_v1/training/seed_44.zip` | 297837 | `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef` |

validation11/12で全9遷移と、通常・緊急・コスト変化の開発seed11各10点検が成功した。マーカー有無の同seed11は全3,691sampleのbytesが一致した。その後 `model-selection-v2.json` にモデル選択を記録し、held-out31/32/33を初めて開いた。全3seedとも9/9遷移が成功した。全5probeは計45遷移・18,455sample、相対位置観測の飽和sampleは計2,445。最長7.83秒、最低高度0.228257 m、最大傾斜56.108433度。位置/速度/四元数/実指令/相対観測/滞留を独立標準ライブラリで再生し、宣言ターゲット・全sample・実行ソース・summaryのhashを結び付けた。固定初期条件の決定論的結果であり、seed変更で擾乱試験を行ったとは主張しない。

初回レビューで、自己申告ターゲットだけを見て静止hover9回を合格させる経路、古いPASS receiptの受入、旧小数優先度3D表示の色契約不一致を発見した。失敗回帰試験を確認後に修正し、source/summary/sample/suiteのhashを凍結時にも再計算する。実pickleの観測除数・初期位置・指令範囲・地面・傾斜条件も検査する。旧契約の表示互換性を保持した。詳細は `docs/evidence/dispersed/review.json`。

## 凍結した新規固定9物理試行

配置確認と遷移評価後、モデル対・座標・カメラ・指令/観測設定・イベント辺・到達/安全判定・有限timeoutを凍結した。`remote-final/outputs/dispersed/frozen-suite-v1.json` のSHA-256は `4070109a98075ad51659a60552ddfe8ad761593e1a625bc6ff6e47333ba8ffe7`。物理実行ソースhashは `52c2c2e8b410db1fa83878acba0c13d19bb228825551fb4130db73cf53af3794`。カメラ/マーカーmanifest hashは `f26648052c9f6ca1de01a7be5ffee94461d667c1f7919cb26064188d8b2b7fce`。

execution seed1/2/3×通常/緊急/コスト変化を各1回、新規に実行し、9/9ミッション・90/90点検が独立物理checkerでPASS。全14,673元フレームをdecodeし、元PTSとstepを照合した。新しいメタデータ専用checkerも全9runを上の凍結モデル/座標/ソース/カメラ/閾値へ結び付けた。物理sourceを凍結した後の追加は、ローカルの独立binding checkerとその単体試験、文書・証拠、および後述のCPU動画encode品質修正で、受入実行先へ追加しなかった。

| シナリオ | 成功/予定 | 実点検 | 実時間 [s] | 実軌跡 [m] |
| --- | ---: | ---: | ---: | ---: |
| 通常 | 3/3 | 30/30 | 32.64 | 19.325204 |
| 緊急 | 3/3 | 30/30 | 33.45 | 21.256661 |
| コスト変化 | 3/3 | 30/30 | 30.74 | 21.636046 |

各seedの決定論的軌跡は一致する。最大傾斜49.613568度、最低実高度0.835222 m、最高実高度3.298682 m、最長leg6.05秒。目的地z=1〜3 mは目標座標の範囲であり、過渡飛行の実高度範囲とは区別する。滞留中最大残差0.199986335 m、最大速度0.299398902 m/s。reset・終了・安全違反なし。半径.20 m、速度<=.30 m/s、保持.20秒、点検.50秒、leg30秒、mission120秒を変更していない。

最初のF点検完了4.72秒後に更新。緊急はC43→100で次選択が更新なしAからJへ変わり、Cは残り順序の8番目から2番目へ早まる。コスト変化はF→Aのみ1→5で、次選択がAからHへ変わる。更新後同一条件で完全suffixを再計算した改善は57.542980と25.501141で、物理時間/電力の削減とは呼ばない。次の実点検は緊急J7.79秒、コスト変化H7.91秒。

全90判断・675完全順序の物理snapshotからQ*/J*・policy suffix・成分を新規再計算し、実pose条件を含むL/H判断は27件。例はcost_shift_seed_1/decision-3、10.30秒。L=C43、H=A81、差38。計画距離.865248/3.518883 m、実pose距離.858543/3.489623 m。Q*(C)=J*=227.444495、Q*(A)=233.699205、AIの完全suffix229.095741、gap1.651246。C先の条件付き最適suffixはC→A→B→E→D→G→I、A先はA→B→E→D→G→C→I、AIはC→I→A→B→E→D→G。最適な次の一手と、最適ではないAIの残り全巡回を区別して表示する。

C先の移動15.409638、生の重要度時刻4240.697143、重み付き212.034857。A先は移動13.943632、生4395.111467、重み付き219.755573。AIは移動12.733198、生4327.250863、重み付き216.362543。近い地点を先にした条件付き最適巡回の総移動が常に小さいとも主張しない。

## 転送・GPU累計・試験

新しい開発成果物はv1=34ファイル、v2=104ファイル、最終成果物252ファイルをそれぞれサイズ/SHA-256で照合した。最終archive43,333,464 bytes、SHA-256 `e3fe0fb253c9c3d40f4eda27e629bdfc4b4870fc94e53bfcf0feb0ab71f2073c`。全モデル・失敗ログ・3開発飛行・固定9試行・probe・設定・台帳を `remote-final/` に保持した。

| GPU job | 結果 | wall秒 |
| --- | --- | ---: |
| train_v1 | 学習終了、モデルは不採用 | 699.104446 |
| validation_v1 | 地面条件で失敗 | 37.998906 |
| train_v2 | 段階学習終了 | 1186.071411 |
| validation_v2 | 2probe＋通常開発飛行成功 | 92.735334 |
| validation_extra_v2 | マーカー比較＋残2開発飛行成功 | 98.501478 |
| heldout_v2 | 3固定probe成功 | 136.466318 |
| acceptance_v1 | GPU契約試験＋固定9試行成功 | 329.601816 |

開始3198.9943080281373秒、今回増分2580.479709449108秒、累計5779.474017477245秒、残り23020.525982522755秒。両台帳を一致させ、active jobなし。GPU契約試験1件は8.82秒でPASS（受入job時間に含む）。最新CPU全体はローカル・固定Linux環境とも196 passed、GPU1件は別実行。lint/compile/空白検査を実施した。

## 新しい動画の人手確認・再実行

今回の全編3本と、緊急2.72〜10.42秒、コスト変化2.72〜12.41秒を提示した。ユーザーから「全3本と両区間を1倍速で確認し、問題なし」を受領した。各区間はイベント4.72秒前から、次点検およびC点検まで連続する。確認は `docs/evidence/dispersed/manual-review.json` の3ファイルhashへ結び付け、前回整数動画への確認を流用しない。

主要実frameの開始・全target変更・イベント直前/直後・最終点検を5枚のcontact sheetで確認し、全8説明カードを実解像度で確認した。全ID/整数重要度/点検状態/現在目標/実高度、XZ高低差、同一frameの実機体cropが読める。比較カードは最適firstと非最適policy suffixを区別する。4ページPDFを全ページ描画し、文字・図・改ページ・欠落のないことを確認した。

以下は新しいLinux workspaceへ入力と依存関係を準備した後の再実行入口。実行設定は固定のサーバパスを再利用せず、現在のcwdから生成する。`finetune_v2` のmodel/cfgと採用routingを同じ相対配置へ転送し、上記hashを照合する。再実行は必ず新しい出力先を使い、GPU supervisorで残予算内timeoutを付ける。

```python
from pathlib import Path
import yaml
root = Path.cwd()
cfg = dict(
    budget_ledger='outputs/gpu_ledger.json', gpu_budget_s=28800,
    flight_checkpoint=str(root/'outputs/dispersed/finetune_v2/model_4296.pt'),
    flight_checkpoint_hash='26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431',
    flight_cfg=str(root/'outputs/dispersed/finetune_v2/cfgs.pkl'),
    flight_cfg_hash='35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55',
    routing_model=str(root/'outputs/integer_20261020_v1/training/seed_44.zip'),
    routing_model_hash='a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef',
    hover_source=str(root/'third_party/genesis_drone/hover_env.py'),
    hover_source_hash='c2230130b9748b69416a18edad8491579e68c67882c8a6b6ab3662c5bbf76703',
    planner_python=str(root/'.venv-cpu/bin/python'),
    flight_eval_config=str(root/'configs/visible_flight_eval.yaml'),
    scenario_files=[str(root/f'configs/demo_scenarios_dispersed/{s}.yaml')
                    for s in ['normal','emergency','cost_shift']],
)
path = root/'configs/execution_dispersed_rerun_new.yaml'
with path.open('x') as stream:
    yaml.safe_dump(cfg, stream)
```

```bash
PYTHONPATH=src .venv-flight/bin/python scripts/probe_dispersed_flight.py \
  --config configs/execution_dispersed_rerun_new.yaml --seed 11 \
  --out outputs/dispersed_rerun_new/probes/seed11 --camera
PYTHONPATH=src .venv-cpu/bin/python scripts/verify_dispersed_transitions.py \
  --root outputs/dispersed_rerun_new/probes/seed11 \
  --out outputs/dispersed_rerun_new/probes/seed11/independent-check.json
```

validation12、モデル選択記録、その後held-out31/32/33の順を守る。採用済み重みから学習を再現する場合は、上の設定を別ファイルへ生成し、checkpoint/cfgと各hashを起点model1300へ置換する。`scripts/finetune_visible_flight.py --config <起点設定> --domain dispersed --curriculum --iterations 3000 --num-envs 256 --out <新学習先>` を使う。検証seedや受入seedを学習モデル選択へ混ぜない。

ローカルで物理証拠を再検証し動画を再生成する入口は以下。`--explanations` はこの新規9試行から再計算したものを使う。

```bash
export PYTHONPATH=src
.venv-routing311/bin/python scripts/build_video.py --config configs/video_dispersed.yaml \
  --input-root outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/acceptance_v1/genesis \
  --explanations outputs/dispersed_20261020_v1/physical-verification/decision_explanations.json \
  --mode full --out outputs/dispersed_media_rerun_new/demo_full.mp4
.venv-routing311/bin/python scripts/build_video.py --config configs/video_dispersed.yaml \
  --input-root outputs/dispersed_20261020_v1/remote-final/outputs/dispersed/acceptance_v1/genesis \
  --explanations outputs/dispersed_20261020_v1/physical-verification/decision_explanations.json \
  --mode submission --duration 180 --out outputs/dispersed_media_rerun_new/submission.mp4
.venv-routing311/bin/python scripts/check_realtime_media.py \
  --root outputs/dispersed_media_rerun_new --ffprobe ffprobe \
  --font outputs/visible_20261015/remote/NotoSansJP.ttf
```

新しい物理受入を再実行する場合、配置確認・全5probeのfresh receiptを揃えてから `freeze_dispersed_suite.py --layout <承認済みlayout> --execution <新設定> --probe-root <5probeの親> --out <新freeze.json>` を実行する。凍結後に `run_demo_scenarios.py --config <新設定> --mode genesis --record --trials 3 --out <新試行先>` を単一GPU jobで実行する。`verify_visible_flight.py --root <新試行先>/genesis --model <routing> --model-hash <上記routing hash> --out <新検証先>` と `verify_dispersed_binding.py --root <新試行先>/genesis --freeze <新freeze.json> --freeze-sha256 <そのhash> --layout <承認済みlayout> --out <新binding.json>` を別CPU環境で実行する。新しいコード構成のsource hashを保存し直し、今回の受入結果を新試行の結果として転用しない。


## 動画の初回不合格と画質修正

初回 `delivery/` の独立checkerはdemo_fullのframe210で高度文字z=1.148mの輝度MAE4.194を検出し、上限4に対してFAILとした。元PNGに同じ文字checkerを適用すると0であり、表示文字/座標ではなくH.264 CRF18圧縮が原因だった。CRF18/0 × YUV420/444の比較を保存し、互換性を保つYUV420・CRF0では通常/緊急とも最大3.469302で合格した。さらに一般的なH.264 High profileを維持するCRF1を検証し、通常3.566167/緊急3.567394で合格したため、最終採用をCRF1とした。表示内容・物理入力・カメラ・時刻・判定閾値を変えず、CPU最終encodeだけを変更し、`delivery_v2/` へ再生成した。受入物理sourceは変更せず、再飛行も行わない。旧動画・失敗log・全診断を保存する。

CRF0は診断用比較として保持し、最終encodeは一般的なHigh profileを維持できるCRF1/YUV420とした。CRF8は高度文字差4.038/4.031で不合格、CRF6は3.733/3.721、CRF1は3.566/3.567で合格した。これらは画質選定の診断であり、受入分母や物理判定の調整ではない。最終全frame検証は選定後の新出力に対して改めて実施し、全5本PASSとなった。

CPU-only encode修正後、ローカル全体196 passed（119.78秒）、動画関連22 passed（0.91秒）を再確認した。固定Linuxの196件は物理/独立checker凍結ソース時点の結果であり、以後の変更はローカルencode品質のみ。GPU物理試行・モデル選択をやり直したとは記録しない。

固定受入をGPU台帳へ記録する具体的な入口（開始前に残予算が5400秒以上あることを確認する）。job supervisorが残予算と単一実行を検査する。

```bash
PYTHONPATH=src .venv-flight/bin/python scripts/freeze_dispersed_suite.py \
  --layout outputs/dispersed_20261020_v1/layout-approved.json \
  --execution configs/execution_dispersed_rerun_new.yaml \
  --probe-root outputs/dispersed_rerun_new/probes \
  --out outputs/dispersed_rerun_new/frozen.json
PYTHONPATH=src .venv-flight/bin/python -m adaptive_drone_patrol.jobs \
  --out outputs/dispersed_rerun_new/job --ledger outputs/gpu_ledger.json \
  --budget-s 28800 --timeout 5400 -- \
  .venv-flight/bin/python -u scripts/final_gpu_validation.py \
  --config configs/execution_dispersed_rerun_new.yaml \
  --out outputs/dispersed_rerun_new/acceptance
```

probe・学習についても同じjobs wrapperを使い、学習3600秒・probe1200秒等の有限timeoutを残予算以下で設定する。旧累計をゼロへ戻さない。validation前にheld-outを開かず、全5probeが揃う前にfreezeを実行しない。

画質修正済み最終3本の新しいリンクと同じ連続区間を再提示し、ユーザーから「最終3本と両区間を1倍速で確認し、問題なし」を受領した。`docs/evidence/dispersed/manual-review-final.json` に最終CRF1の3ファイルサイズ/SHA-256・区間・回答を保存した。旧hashに対する回答から最終版の承認を推定せず、両方の回答記録を保持する。


## 最終動画・PDF・全受入の確定（2026年10月21日）

`outputs/dispersed_20261020_v1/delivery_v2/` を最終出力とする。通常980 frame/32.666667秒、緊急1004/33.466667秒、コスト変化923/30.766667秒、統合4227/140.9秒、提出用4227/140.9秒。全11,361 frameのdecode・時刻・元step・表示・高度文字・元画素・実機体cropを独立checkerで照合し、5/5 PASS。元映像PTS最大誤差7.105427357601002e-15秒、HUD文字の最大輝度MAE3.525731421744325、説明カード0.6804767770876466、機体crop3.5492444444444446。閾値を緩めず合格した。全飛行96.9秒、説明カード44秒、実飛行比率68.7721788502484%、カットなし。提出用は統合版と同じ映像bytesを使い、別のsubmission timeline・mapping・manifestを持つ。

最終PDFは `delivery_v2/report/final-report.pdf`。初回PDFと中間draftは上書きせず保持した。最終実frame・PDFの表示確認、各動画/モデル/config/PDFのサイズ・SHA-256、元映像を含む最終manifest照合を行った。受入証拠は `docs/evidence/dispersed/acceptance.json`、動画照合は `docs/evidence/dispersed/media-verification.json`、成果物は `docs/evidence/dispersed/artifacts.json`。大規模成果物と完全manifestはignoredの出力先に保持する。

一時的に手動引き継ぎを案内したが、実行中checkerは全5本を完了した。照合を省略した事実はなく、未完了扱いのまま合格にしたものでもない。最終人手確認はCRF1の新hashへ結び付けて別に受領済み。全7条件の未達なし。ただし擾乱・障害物・実機安全性の一般保証や全policy suffix最適性は主張しない。指定identityの新規ローカルcommitに今回分だけを収め、push/PR作成/公開/提出は行わない。

最終commit直前の現在ソースでもCPU全体を再実行し、196 passed・GPU1件deselected、53.79秒で合格した。`docs/evidence/dispersed/tests.json` にPython source/test treeのhashとコマンドを記録した。GPU契約1件は前述の受入job内で別に合格済み。最終lint/compile/空白・対象範囲・機密情報/絶対パス監査・93ファイルmanifest照合・両GPU台帳照合もPASS。
