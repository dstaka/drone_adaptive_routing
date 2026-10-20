# 1倍速デモ動画の引き継ぎ

[承認済み仕様](../specs/2026-10-14-realtime-demo-video/spec.md)を、両先行PRがマージされた `974274d` から実装しました。`ff346a0` と指定の `91cf9858d12540052ac2ad46572b1c54f5ec02bf` はともに祖先です。実装先はルートで、歴史資料・個人の未追跡ファイル・過去成果物は変更していません。今回のspecは未追跡から保護し、コードと同じcommitに含めます。

動画生成、独立した時刻・画素検証、静止フレームとPDFの目視を実施しました。2026年10月17日にユーザーから動画の目視OKを受領し、残っていた人手確認を完了として記録しました。`acceptance.json` は仕様受入PASS、未達項目なしです。改善案は別spec／PRで扱います。

## 入力と保護

- 元素材：`outputs/visible_20261015/remote/outputs/visible/acceptance_v1/genesis/`。
- 元の物理判断説明：`outputs/visible_20261015/remote/outputs/visible/verification_v2/decision_explanations.json`。
- 過去の注釈版・目視資料・失敗ログも元の場所に保持。旧media-checkを今回の受入には流用しません。
- 転送アーカイブ：`outputs/visible_20261015/private/visible-delivery.tar.gz`、SHA-256 `d6cba7b79cfeb60cdb9c29b7f94889fb69d6c650f974508dba4317a332d9a88f`。
- 今回も転送manifestの465ファイルすべてのサイズとSHA-256を再照合し、一致しました。前回のreceiptは変更しません。[今回の入力照合](evidence/realtime/inputs.json)も保存します。

採用モデルは次のままです。再学習・再撮影はありません。

| 入力 | 保存先 | SHA-256 |
| --- | --- | --- |
| 飛行モデル | `outputs/visible_20261015/remote/outputs/visible/finetune_v1/model_1300.pt` | `4a6f3b333735f3c10ed4e51f2ddd3466308cf90b7dfb3bec289b519a084a539f` |
| 飛行設定 | 同ディレクトリの `cfgs.pkl` | `25720a5b59b1edd769ed36ecc29be16febde3ad6088dcc6bdd878f71d2f8fff1` |
| ルーティングモデル | `outputs/visible_20261015/remote/outputs/scalable_20261015/normalized-src/outputs/training/seed_42.zip` | `c4c2a937d26650c09a7810073382263e79ba58fcf238a752387fd41d45a2bc2a` |

## 出力と選択

一意な保存先は `outputs/realtime_20261016_v2/`（Git管理外）です。選択は第一候補のexecution seed 1。9試行すべて成功しているため、最低の共通seedを維持し、見栄えや結果で選び直していません。全9試行の詳細summaryを `all_trials.json` に保存し、9/9成功・90/90点検の分母を保持しました。

| 動画 | 内容 | 長さ／フレーム数 |
| --- | --- | --- |
| `full_normal.mp4` | 通常、全10点検・省略なし | 17.833333秒／535 |
| `full_emergency.mp4` | 地点Cの優先度上昇、全10点検・省略なし | 17.233333秒／517 |
| `full_cost_shift.mp4` | B→Aのコスト上昇、全10点検・省略なし | 17.233333秒／517 |
| `demo_full.mp4` | 全3飛行＋独立カード | 96.30秒／2,889 |
| `submission.mp4` | 同じ完全飛行が提出尺に収まるため、省略なし | 96.30秒／2,889 |

全編集動画は1920×1080、H.264、30fpsです。Genesisの元画面1280×720をそのまま主画面へ配置し、歪ませません。実飛行52.30秒、カード44秒、実飛行比率54.3094496366%です。固定120秒に飛行を引き伸ばさず、カットも不要でした。各出力のtimeline・selection・verification・manifestを隣接ファイルとして保存します。

`sources/*_seed_1/` は選択した元run全体の完全コピーです。元raw、frames.csv、telemetry.csv、events.jsonl、decisions.jsonl、scenario、scene、summaryを保持しています。新しい1倍速rawは `raw/*_seed_1/raw.mp4`、対応表は同じ場所の `frames.csv` と `timing.json`。元の物理ログは隣接selectionから `sources/` へ明示的に参照します。

## 時刻と末尾の扱い

実physics timestepは0.01秒、定期撮影は2stepごとの50fpsです。元recorderは定期撮影に加え、到達・選択・終了時に強制撮影しており、各runに11件の同step追加画像があります。元rawの最後のPTSは物理mission-endから通常0.30秒、緊急・コスト変化0.33秒遅れていました。元の動画・対応表のbytesは変更しません。

正規化では、同stepの最後の撮影（最新event_seq）を採用します。stepやevent_seqの逆行、不明な重複、対応表の欠落、cadenceより長い隙間は拒否します。破棄した追加撮影のindexもすべて記録します。端数stepで強制撮影された末尾もVFRの実PTSへ置き、通常896、緊急868、コスト変化868フレームになります。終端packetの0.01秒はコーデック上のcoverで、追加飛行を意味しません。

編集は物理区間 `[start,end)` を単位傾斜で30fpsへ標本化し、直前の実撮影を選びます。最後の1出力sampleのみ実mission-endの最終画像へ置き換え、10/10点検の最終状態を表示します。これは最大1出力frameの端点量子化です。架空の点検後飛行や時計進行を追加しません。通常の尺量子化は約0.013333秒、緊急・コスト変化は各約0.003333秒で、合計も0.02秒です。

## 判断説明と表示

A〜JのID、優先度の固定1〜5色と数値、点検済み✓、現在目標の水色輪、実位置・実軌跡・実距離を表示します。緊急はシナリオB、更新対象は地点Cと書き分けます。コスト変化はB→Aの有向辺と1→5の倍率を明示します。map boundsは全座標から計算します。

L/H例は `cost_shift_seed_1 / decision-1`、実時刻3.96秒です。L=地点C（優先度2.660865759308、計画距離0.600 m、pose距離0.611002767 m）、H=地点H（優先度2.693373135601、計画距離1.200 m、pose距離1.181392281 m）。Q*(L)=J*=J_policy=149.096595315295、Q*(H)=168.795996、差約19.699401です。完全な数値、条件付き最適suffix、成分と全policy rolloutは保持した説明JSONから表示します。

現在の条件を固定した残り巡回の比較であり、物理電力やAIの内部思考の説明ではありません。最適な次の一手と全policy suffixの最適性を別々に判定し、証明のないforced-first値からは最適というcaptionを出しません。ライブ映像のL/H表示は現在の実目標が変われば消すか、その判断IDと時刻を明示します。

緊急とコスト変化のカードは、同一境界の更新なし／更新後の予測と、更新後条件で再価格付けした総コストを示します。改善は25.219822682394と7.035441364304。保持した実飛行は予測とは別で、全点検完了まで収録しました。

## 独立検証と開発時の失敗

`video_check.py` はcompositorやその時刻helperをimportしません。元のphysics step・実timestep・元rawの実PTSと出力の実PTSから対応を再計算し、宣言されたplayback_rateだけでは合格しません。

元ファイルのSHA-256を確認後、checker自身が元rawを新しいscratchへ全デコードします。正規化rawも全フレームを独立デコードし、元撮影の対応indexと局所tile画素を照合します。編集版は全フレームをデコードし、実画素の参照strip、主カメラ画面、元画像由来の機体拡大、日本語の時計・点検数・目標・優先度・イベント字幕を照合します。stripだけに依存しません。カードは認証済みのsnapshot・候補値・順序・独立した成分計算と実表示を照合し、その後も静止していることを確認します。

色の強い文字のH.264 YUV420色差間引きは、輝度MAEで扱います。実字幕の輝度誤差1.90に対し、字幕を消すと36.76となり、上限4を大幅に超えて拒否されました。主映像の消去、候補数値の999への改変、PTS伸長、runの取り違え、欠落frame、未証明の最適caption、manifestの自己参照も回帰テストに含めます。合成fixtureはテスト専用で、物理測定として表示しません。

開発版 `outputs/realtime_20261016_v1/` にchecker失敗log（`checker.log`、`checker-v2.log`）と旧合成動画を保持しました。目標切替stepで距離だけが旧目標を参照する問題も実フレームで発見し、v2で実poseから現在目標へ再計算するよう修正しました。checkerも表示距離を独立計算して照合します。カード時刻の厳密比較は約4.44×10^-16秒の浮動小数点差で失敗したため、物理stepの一致を保持しつつ微小誤差を許容しました。飽和色字幕のRGB差も原因を実画素で切り分け、輝度比較に変更しました。開発版VFRを診断用rawvideo muxerへ出した際のDTS丸め警告は、元MP4のPTS逆行ではありません。v2のcheckerは診断出力のtimebaseも明示します。

CPU全体テストの負荷並行実行では、既存IPCテストの0.2秒timeoutが1件発火しました。該当IPC群を単独再実行して全件通過し、最終の修正版では全体試験142 passed、1 deselectedとなりました。テストのtimeout値や既存IPC実装を今回の都合で変更していません。

## 目視と受入

開始、全目標切替・点検、両イベント前後、最終点検と全カードを、実際の符号化動画から `visual_qa/` に抽出しました。個別の1920×1080フレームとcontact sheetを確認しました。日本語glyph、10地点、色・数値・選択・点検の区別と比較表は読めます。元のcamera/geometryを維持しているため、到達付近で機体と中立ランドマークが重なる場面はあり、同じ元画像の機体拡大で確認できます。レポートは `report/report.pdf`、最終3ページをPopplerで描画して確認しました。

[目視記録](evidence/realtime/visual-review.json)には、2026年10月16日にエージェントの連続再生確認が未完了だった経緯と、翌日のユーザーによる目視OKを記録しています。localhostの確認ページはローカルHTTP 200でしたが、アプリ内browserはtimeout、直接file URLはセキュリティポリシーに拒否されました。迂回はしていません。一時HTTPサーバは終了しました。

当初依頼した確認範囲は `full_emergency.mp4` と `full_cost_shift.mp4` の1.96〜6.00秒を1倍速で連続視聴するもので、イベント前後と地点Cの次の実点検（5.42秒）を含みます。2026年10月17日にユーザーから「目視して動画はOKでした。改善案を思いつきましたが、それは別のspec/PRで対応します。」と回答を受領し、残っていた人手確認を完了として受け入れました。ユーザーの回答が示していない再生ソフト・視聴区間・速度設定の詳細は推測せず、エージェント自身が連続再生したとも記録しません。

## 再実行

[READMEのCPUコマンド](../README.md)で、新しい出力先へfull、submission、独立checkerを順に実行してください。フォントは `outputs/assets/NotoSansJP.ttf`（SIL OFL 1.1）。既存ローカルPython 3.11.14の `.venv-routing311` を使用し、Linuxの飛行／CPUロックは変更していません。FFmpegは既存imageio-ffmpegの7.1、ffprobe 7.1.1は今回のignored `tools/` へ取得し、[入力記録](evidence/realtime/inputs.json)へhashと配布元を記録しました。

```bash
export PYTHONPATH=src
.venv-routing311/bin/python scripts/inspect_realtime_frames.py --root outputs/realtime_new
.venv-routing311/bin/python scripts/finalize_realtime_delivery.py --root outputs/realtime_new \
  --visual-notes outputs/realtime_new/visual-review.json
```

finalizerは自動検証・実フレーム・PDFの確認後にのみ実行します。連続視聴が未完了なら未達のままmanifestを作成します。Gitにはコンパクトな証拠だけを保存し、元ログの実行先依存の設定はignored成果物に残します。

GPUは両最新台帳で2,803.57274048205秒、今回の増分0秒、残り25,996.42725951795秒です。新サーバ接続・新サーバ作業ディレクトリ・新しい外部転送はありません。継承した転送は465/465一致。契約変更・停止・追加レンタルは行っていません。

最終revisionはこの記録を含む1件のローカルcommitです。commit後のID・identity・差分範囲は、Git管理外の `outputs/realtime_20261016_v2/final-commit.json` に保存します。push、GitHub PR作成・マージ、動画公開、コース提出は行いません。

## 最終検証コマンド

```bash
export PYTHONPATH=src
.venv-routing311/bin/python -m pytest -m 'not gpu' -q
.venv-routing311/bin/python -m ruff check src scripts tests
.venv-routing311/bin/python scripts/check_realtime_media.py --root outputs/realtime_20261016_v2 \
  --ffprobe outputs/realtime_20261016_v2/tools/ffprobe
.venv-routing311/bin/python scripts/inspect_realtime_frames.py --root outputs/realtime_20261016_v2
.venv-routing311/bin/python scripts/finalize_realtime_delivery.py --root outputs/realtime_20261016_v2 \
  --visual-notes outputs/realtime_20261016_v2/visual-review.json
git diff --check
```

CPU全体142件と新規19件はそれぞれ通過、GPU試験1件は今回は実行しません。独立レビューで指摘された元画像の信頼境界、画面・カードの未照合、候補値の結び付け、raw画素、manifest自己hashを修正して再確認しました。採用checkpointをCPUで実際に読み込み、選択した30物理boundaryのactionと完全policy suffixも再現しました（[記録](evidence/realtime/policy-replay.json)）。

## 確定した機械検証結果

v2の編集動画5本、計7,347フレームを全デコードしてPASS。正規化raw3本、計2,632フレームも、元撮影と実PTS・画素の照合でPASSです。元撮影の独立デコードも実施しました。イベント同期の最大絶対誤差は0.030000000000015348秒で、30fpsの1フレーム以内です。提出用の実飛行比率は54.30944963655244%。全8納品動画のtimeline・selection・verification・個別manifestを保存し、最終manifestの93ファイルすべてのサイズとSHA-256が一致しました。

最終CPU全体試験は142 passed、1 deselected（38.95秒）、Ruffと差分空白検査もPASSです。[機械検証](evidence/realtime/verification.json)、[チェック集計](evidence/realtime/checks.json)、[成果物ハッシュ](evidence/realtime/artifacts.json)、[受入状態](evidence/realtime/acceptance.json)、[CPU環境](evidence/realtime/environment.json)を保持します。2026年10月16日はユーザーの「未確認のまま記録する」という回答に従い人手確認を未完了としました。翌17日の目視OKにより、その未達は解消しています。以前の記録とPDFは `outputs/realtime_20261016_v1/acceptance_20261017_before/` に保存し、動画・物理ログは変更していません。更新後のPDFとmanifestは再生成・照合しました。

最終組立では内部commit対象listをmanifest対象外のscratch名へ移し、manifestを再計算しました。組立途中の旧manifestはv1開発記録へ保存し、確定93件を再照合しています。
