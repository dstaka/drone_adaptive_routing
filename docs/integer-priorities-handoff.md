# 整数重要度の引き継ぎ

[承認済み仕様](../specs/2026-10-17-integer-priorities/spec.md)を、realtime-demo-videoを含む `7da4d23721770613d3ca910a20718ed6fd0e9371` からリポジトリルートに実装した。新成果物は `outputs/integer_20261020_v1/`、過去の素材・モデル・動画・ログは保持する。分散配置・吊り荷・個人ファイルは今回のcommit対象外である。納品revisionはこの文書を含む新規ローカルcommitで、確定後のIDを同出力先の `final-commit.json` に記録する。

## 確定契約と生成規則

scenario v3 / `integer-100-v1`、観測 `routing-v3-integer100-n10-324`。実値・表示とも1〜100整数で、7.0、真偽値、文字列、小数、非有限値、範囲外を拒否する。観測は100で正規化し、alpha=1、beta=0.05、名目速度1 m/s、サービス0.50秒、帰還なし。有向倍率は1〜5のまま。生の重要度時刻成分と0.05適用後の寄与を別保存する。v2と旧データは元の小数1〜5の解釈を維持し、新schemaと配列サイズだけで混用しない。

生成規則は `numpy-PCG64-SeedSequence(seed,100)-integers-1-101-v1`。`Generator(PCG64(SeedSequence([seed,100]))).integers(1,101,N)` で各地点独立に復元抽選する。形状の乱数列とは分離する。1〜100すべて、重複、固定seed再現をCPU試験で確認した。開始後は明示イベント以外で再抽選しない。

デモseed40029の初期値は A=81、B=42、C=43、D=64、E=23、F=84、G=6、H=98、I=22、J=56。`configs/demo_scenarios_integer/` の3シナリオと実行seed1/2/3が共有する。既存の長方形配置・S・0.60m間隔を維持し、分散配置は実装しない。最初の点検後に未点検Cのみ43→100へ上げる。色は1〜100の固定写像をscene manifestへ全値記録する。Hは条件を満たす最大重要度地点をID順に選び、同値の最大地点もすべて記録する。L/H差10以上と既存の厳密コスト・計画距離・実pose距離の条件を併用する。

## 学習・選択・一般評価

新しいcandidate MaskablePPOを3seed42/43/44で学習した。学習専用12,000状態・30epoch初期化後、各50,176静的＋50,176イベント混合PPOステップ（計100,352）を実行した。推論時に厳密解・教師・heuristicへのfallbackはない。設定は `configs/routing_integer.yaml`。教師データは `outputs/integer_20261020_v1/training/teacher/`、npz SHA-256 `f812cb66367cba279638111124357d84cb5d43e12bcb49ba10bcac76e9ef9aee`、入力・厳密ラベルmanifest SHA-256 `1a27bb82069f43fae7f7723023486ca3b63775ee71767fe983da77fdd9c975a3`。

validation100例の平均コストはseed42=143.639609、43=144.362925、44=143.058966。held-outを開く前にseed44を採用・凍結した。静的held-outでseed42が良くても選び直さない。全学習監督の実時間538.932336秒、教師生成118.835759秒（監督時間に含む）。各seedの学習実時間は約379〜382秒で並行実行しているため単純合算を全体時間とは呼ばない。

| seed | 静的平均コスト | 平均AI/厳密解比 | greedyとの差 | fixedとの差 | 動的 適用/予定 | 選択変更 | 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 140.630838 | 1.033168 | -9.946941 | -111.361167 | 50/50 | 10 | 8 |
| 43 | 141.712510 | 1.041228 | -8.865270 | -110.279495 | 50/50 | 7 | 3 |
| 44（採用） | 141.470677 | 1.040070 | -9.107103 | -110.521328 | 49/50 | 7 | 6 |

静的100・動的50を各seedに共通使用し、450 learned episodesが完走、無効行動0。静的の平均比<=1.10、greedy以下、fixedより小さい、の既存条件を全seedで満たした。採用seedの最悪比は1.156867、最悪絶対gap24.205075で、一般に常時最適とは主張しない。動的49適用のうち改善は6例であり、すべての環境変化に有益とは限らない。完全なデータ・順序・失敗/skip・成分は `static/` と `dynamic/` に保持し、Gitにはsummaryのみ置く。

## 採用モデルと互換性

| 用途 | 保存先 | bytes | SHA-256 |
| --- | --- | ---: | --- |
| 新routing採用seed44 | `outputs/integer_20261020_v1/training/seed_44.zip` | 297837 | `a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef` |
| 採用飛行モデル | `outputs/visible_20261015/remote/outputs/visible/finetune_v1/model_1300.pt` | 475573 | `4a6f3b333735f3c10ed4e51f2ddd3466308cf90b7dfb3bec289b519a084a539f` |
| 飛行設定 | 同ディレクトリの `cfgs.pkl` | 1592 | `25720a5b59b1edd769ed36ecc29be16febde3ad6088dcc6bdd878f71d2f8fff1` |

旧routing採用checkpointを実際に読み込み、v3シナリオを推論前に拒否した。新checkpoint内部のschemaとsidecarを照合し、shape互換の旧教師データも拒否する。学習はローカルPython3.11.14/Torch2.2.2 CPUで実施し、新サーバPython3.12.3/Torch2.6.0+cpuで読込み・シナリオ適合を確認した。新サーバの飛行環境は別venvのTorch2.14.0+cu132/Genesis1.4.3。飛行169版、CPU33版が各Linux lockと一致した。飛行モデルは17観測・4モータ・同じ指令範囲を保持し、失敗した元モデルへ戻していない。

## デモ選択と独立計算

demo40000〜40099の範囲を順に調べ、30候補目40029を採用した。全候補の初期抽選値、却下ゲートと比較結果を `demo/selection.json` に保存する。これは条件付き選択の説明用デモで、無作為benchmarkではない。学習・validation・held-out集合とは分離する。イベントなしFへの予測に対し、C更新後はC、H→Fの倍率1→5後はJを次に選ぶ。更新後の同一条件で全suffixを再評価した改善は0.631889と14.734742である。別々の予測と実飛行を混同しない。

全30 analytical判断・225完全順序を標準ライブラリだけの独立計算で再生し、最大成分差9.094947017729282e-13。新profileのN=3〜7、prefix有無、全forced-firstを完全列挙とも照合した。これらは解析/単体試験であり物理測定ではない。

## 物理・動画・受入記録

新サーバで長距離・反転・同地点操作を確認した。マーカー有無の1,508サンプルは物理状態・動作が一致する。旧素材465ファイルと、新規スモーク転送28ファイルをサイズ・SHA-256で照合した。固定9試行は新モデルと新重要度を使い、実行seed1/2/3×通常/緊急/コスト変化として一度ずつ走らせる。確定結果と動画・GPU累計は末尾へ追記する。

## 再実行

出力先は毎回新しくする。既存モデル・詳細ログ・動画を上書きしない。Linux環境は `docs/visible-flight-handoff.md` の別venv構築手順に従い、今回も両lockを保持する。最新の両GPU台帳の履歴を確認し、使用累計をゼロへ戻さない。

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=outputs/integer_rerun_new/mpl
.venv-routing311/bin/python scripts/save_scenario_manifests.py --num-checkpoints 10 \
  --priority-profile integer-100-v1 --out outputs/integer_rerun_new/manifests
.venv-routing311/bin/python scripts/train_routing.py --config configs/routing_integer.yaml \
  --seeds 42 43 44 --out outputs/integer_rerun_new/training
.venv-routing311/bin/python scripts/freeze_routing_models.py \
  --models outputs/integer_rerun_new/training/models.json --out outputs/integer_rerun_new/frozen
.venv-routing311/bin/python scripts/eval_routing.py --models outputs/integer_rerun_new/frozen/models.json \
  --num-checkpoints 10 --split static --episodes 100 --out outputs/integer_rerun_new/static
.venv-routing311/bin/python scripts/eval_routing.py --models outputs/integer_rerun_new/frozen/models.json \
  --num-checkpoints 10 --split dynamic --episodes 50 --out outputs/integer_rerun_new/dynamic
.venv-routing311/bin/python scripts/select_demo_scenarios.py --models outputs/integer_rerun_new/frozen/models.json \
  --out outputs/integer_rerun_new/demo
```

新サーバへ採用モデル・sidecar・飛行モデル/設定・ソース・3シナリオを転送し、manifestと照合する。実行設定のパスはその作業環境で生成する。元サーバの絶対パスは使わない。接続情報・実行先依存の設定はignored成果物だけに保存する。次の生成例は、リポジトリルートに同じ相対配置で転送済みの場合に使う。

```bash
.venv-cpu/bin/python - <<'PY'
from pathlib import Path
import hashlib
import yaml
root = Path.cwd()
flight = root / 'outputs/visible_20261015/remote/outputs/visible/finetune_v1'
model = root / 'outputs/integer_20261020_v1/training/seed_44.zip'
cfg = {
    'flight_checkpoint': str(flight / 'model_1300.pt'),
    'flight_cfg': str(flight / 'cfgs.pkl'),
    'routing_model': str(model),
    'hover_source': str(root / 'third_party/genesis_drone/hover_env.py'),
    'planner_python': str(root / '.venv-cpu/bin/python'),
    'scenario_files': [str(root / f'configs/demo_scenarios_integer/{s}.yaml')
                       for s in ['normal', 'emergency', 'cost_shift']],
    'flight_eval_config': str(root / 'configs/visible_flight_eval.yaml'),
    'budget_ledger': 'outputs/gpu_ledger.json', 'gpu_budget_s': 28800,
}
for key in ['flight_checkpoint', 'flight_cfg', 'routing_model', 'hover_source']:
    hash_key = 'flight_cfg_hash' if key == 'flight_cfg' else key + '_hash'
    cfg[hash_key] = hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()
assert cfg['routing_model_hash'] == 'a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef'
with (root / 'configs/execution_integer_rerun.yaml').open('x') as stream:
    yaml.safe_dump(cfg, stream)
PY
.venv-flight/bin/python -m adaptive_drone_patrol.jobs --out outputs/integer_rerun_new/job \
  --ledger outputs/gpu_ledger.json --budget-s 28800 --timeout 5400 -- \
  .venv-flight/bin/python -u scripts/final_gpu_validation.py \
  --config configs/execution_integer_rerun.yaml --out outputs/integer_rerun_new/acceptance
```

固定受入を撮り直す場合は必ず9件すべてを再実行し、過去の成功で置換しない。新しい分散配置セッションでは、この確定モデル/生成規則/設定を起点に、その配置specの受入・飛行範囲・比較条件を改めて評価する。今回の結果を新配置の成功として転用しない。

## 確定した新規物理結果とGPU記録

固定9試行は9/9成功・90/90点検。全runの8322元フレームの数・PTS・物理stepを照合した。通常17.61秒、緊急18.07秒、コスト変化18.76秒。滞在中最大残差0.1996167203m、最大速度0.2998611199m/s、最大傾斜37.68631880度、最低高度0.8058773279m。reset・終了・安全異常0件。全90判断・675完全順序を再計算し、重要度差10以上かつ実pose条件を満たすL/H判断は33件である。

比較カードはcost_shift_seed_1 / decision-3（7.53秒）。L=D（64）、H=A（81）、差17。計画距離0.600/1.897367m、実pose距離0.643139/1.902376m。Q*(L)=J*=J_policy=111.025026525162、Q*(H)=130.165238515481。Lの移動4.800000、生の重要度時刻2124.500530503232、重み付き106.225026525162。Hは移動6.980648169101、生2463.691806927611、重み付き123.184590346381。両者はそれぞれ完全な条件付き最適suffixで、単なるforced-first方策の比較ではない。

新サーバの一意な実験ディレクトリ名はinteger_20261020_v1_01。実際の接続先・絶対パスは `outputs/integer_20261020_v1/private/remote-workdir.txt` とignored実行設定/ログに保存し、公開文面へ含めない。受入ソースpayloadは `private/acceptance-source-v1.tar.gz`、SHA-256 `4421ff081058d6f138797f6241c1b9531b6394fc2ca91825f479728fcf809c6e`。実物理ソースhashは `a553140392ba13837d6fcca1fa29ef47f10b82d067b1777040f5ae8db564cfdd`。受入後のソース差は `scripts/build_report.py` のPDF日付、`src/adaptive_drone_patrol/video_delivery.py`、`scripts/finalize_realtime_delivery.py` の動画最終化metadataと、`tests/test_integer_priorities.py`、`tests/test_video_delivery.py` の追加CPU試験。物理条件・モデル・元記録は変更しない。最終Linux CPU検証は別ソースディレクトリで実施した。

GPUは初期累計2803.57274048205秒から、カメラprobe99.80334640096407秒、描画なしprobe24.137300082016736秒、GPU契約テスト＋固定9試行271.4809210631065秒を追加。今回増分395.4215675460873秒、累計3198.9943080281373秒、残り25601.005691971863秒。両台帳は同値で、未精算active jobなし。新サーバで飛行モデルの再学習は不要だった。契約変更・追加レンタル・停止・破棄は行っていない。

新規受入転送は132ファイル、アーカイブ6629586bytes、SHA-256 `29c47aac3514cd904db73eba1467e5537b7364c788de505df69769abb59aacf7`。すべてのファイルのサイズ/SHA-256が一致した。詳細manifestは `remote-acceptance/acceptance-delivery-manifest.json`、照合は `transfer-receipt.json`。スモークの28ファイルも照合し、archive SHA-256は `eeee839da5d1f242338f0d0448ad8b1439efded7d9c45a4fc8d178088f19a5aa`。旧465ファイルの照合とは別の新規物理証拠である。

初回の環境構築スクリプトは背景起動時のstdin扱いにより空になり未起動だった。別名で転送完了を確認して再実行した記録をprivateログに残す。独立レビューでは新しい重み付き成分の未照合を検出し、改ざんを拒否するRED→GREEN試験と独立標準ライブラリ/動画checkerを修正した。再レビューに重大な未解決指摘なし。

最終CPU試験はローカル・固定Linux環境とも177 passed、GPU1件は別実行でpassed。Ruff・差分空白検査も通過した。Linuxログ1963bytes、SHA-256 `925861f1ec00a66096feaa47dece663b8093cc9c80ecd07fbcc8f907b794f564`。実行環境・コンパクトな測定は `docs/evidence/integer/`、詳細ログはignored成果物に保存する。

2026年10月20日、ユーザーから「全3本と両区間を1倍速で確認し、問題なし」と回答を受領した。対象は新規3全編と、緊急/コスト変化の1.43〜6.00秒。3.43秒の更新前後から次の実点検（緊急5.74秒、コスト5.55秒）までを含む。エージェントによる静止frame・contact sheet・全8カードの目視と分けて記録する。ブラウザで1倍速再生と次の点検後までの進行も確認したが、人手確認の根拠はこのユーザー回答である。

## 最終メディアと受入

編集動画5本の全7545フレーム、1倍速raw3本の全2741フレームが独立checkerでPASS。元撮影も新しいscratchへ全デコードし、物理step・実PTS・HUD・機体拡大・画素と成分を照合した。最大イベント同期誤差0.030000000000001137秒。統合/提出版は98.50秒・2955フレーム、実飛行54.50秒（55.329949%）、全区間保持・省略なし。正規化rawの末尾0.01秒はcodec coverで追加飛行ではない。

最新版PDFは `delivery/report/report.pdf`、4ページ。すべて描画して確認した。初版は7節見出しの孤立を見つけ、見出しと次の段落を同ページに保つ修正後に再生成した。初版・その描画は `report-draft-v1/` に保存し、最終版の1/2ページは画像bytesも同一、3/4ページを改めて表示確認した。データ監査ではNPZ全配列を各状態で繰返し解凍する初回実行を中断し、配列を一度だけ読み込んで再実行した。データ・モデル・受入条件は変更せず、30,000候補と12,000教師状態をすべて照合した。

`delivery/acceptance.json` と `docs/evidence/integer/acceptance.json` は全受入PASS、未達項目なし。人手確認も受領済み。最終media manifestは90ファイル、全サイズ/SHA-256が一致した。動画・モデル・PDF・設定の主要hashは `docs/evidence/integer/artifacts.json`、詳細は `delivery/manifest.json` と新規転送manifestを参照する。公開文書/commitには接続先・ユーザ名・マシン固有絶対パスを含めない。

再検証・最終化:

```bash
export PYTHONPATH=src
export PATH="$PWD/outputs/integer_20261020_v1/tools:$PATH"
.venv-routing311/bin/python scripts/verify_visible_flight.py \
  --root outputs/integer_20261020_v1/remote-acceptance/outputs/integer/acceptance_v1/genesis \
  --model outputs/integer_20261020_v1/training/seed_44.zip \
  --model-hash a28ac49e696c7439b69ce90979be5a919df489e00a12f077c100cb735b1debef \
  --out outputs/integer_physical_recheck_new
.venv-routing311/bin/python scripts/finalize_realtime_delivery.py --root outputs/integer_video_new \
  --visual-notes outputs/integer_video_new/visual-review.json \
  --execution-context outputs/integer_video_new/execution-context.json
```

最終化は独立checker・実フレーム・PDF確認の後に実行する。`execution-context.json` に、その再実行の開始時GPU累計 `gpu_baseline_used_s`、新実験ディレクトリの相対名 `new_server_directories`、実際に照合した転送記録 `transfer` を記録し、増分0や旧転送だけだったという誤ったmetadataを継承しない。人手未確認なら `continuous_playback_review` は未完了のまま保存する。次の配置PRは、この文書を含むcommit・確定seed44モデル・整数生成規則をそのまま起点に利用できる。新配置の学習/飛行/説明/動画条件は次specに従い新規検証する。

PDF配置修正後にもローカルと固定Linux CPU全体を再実行し、双方177 passedを再確認した。最新Linuxログは `linux-cpu-commit-tests.log`、1963bytes、SHA-256 `77551fc791c0db847ab6d43320a06af74d447b88f1e7f375982abd39c9e3efc9`。旧CPUログも保持する。一時レビュータブとloopbackサーバは人手確認後に終了した。
