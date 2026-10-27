# Elevated payload 実行・回収 handoff

[仕様](../specs/2026-10-25-elevated-payload-destinations-detail-zoom/spec.md) と [結果](elevated-payload-result-ja.md) を併読する。新runは `outputs/suspended_mujoco_elevated_20261025T141122Z/`。旧runとマージ済み仕様を上書きしない。接続情報・鍵はGitへ記録しない。

## 固定した入力と実行順序

高度profile `payload-elevated-z-v1`、camera `configs/payload_mujoco_elevated_camera.json`、選定F43 SHA `ee966a2f761b513074c6df2e7de177b3d7bc76ac9d5ec3602d273e5210e84652`。controllerは既存 `C-local-lqr-v1`。routingとflight controlを混同しない。飛行NNなし。

LinuxはPython3.12/NumPy2.5.3/MuJoCo3.15.0の既存資格bindingと一致する環境。実engineの209ソース/libraryを `runtime-evidence-v1/` に回収し、全SHAをfreezeと照合済み。CPU実行、N1、RK4/substeps16/100Hzを保持する。routing学習はローカルPython3.11/Torch2.2.2で完了。サーバ推論の1200共通入力で全route/cost一致を確認。実versions/sourceはrun内receiptを正とする。

1. finite F42/F43/F44学習→300validation→事前規則によるF43選定。完了。
2. 選定F43の15開発、12高度条件/120異座標移動。runner内独立監査に成功。別の完全raw再監査・camera全姿勢検証も完了してjoint freezeを保存。
3. joint freezeを実ファイル・model・source・実engine・依存versionsから再検証後、routing held-out300を一度だけ生成し、全4モデルを報告。再選定しない。実測F43の絶対gateは合格。平均cost345.061332580対B355.530542240。以降モデルを切り替えない。絶対gate失敗なら作業未達として停止する。
4. 物理127のcampaign claimを排他的に作成してからcohortを生成。basic9/9、endpoint18/18、heldout≥95/100。routing300は分母外。同じfreezeで別outputを指定しても再実行不可。失敗caseの置換・再試行なし。
5. 127全caseをraw/control/native/RK/mission/flushから別途再監査。aggregateは各audit・全input bytesと結合し、最終動画の前後にも全127を再検証する。
6. accepted basic seed1のnormal/emergency/cost_shiftだけを100Hzで最後まで撮影・全frame再合成・PTS・字幕・全身体/全目標の画面内・旧距離比拡大率を検証。追加の物理成功数なし。
7. 必要全raw/RK/models/source/receipt/master/MP4/logをローカルへ回収。サイズ/SHA一致と3本のfresh連続1倍速人手確認を満たして初めて全体完了。

## 容量と回収

実測開発15件は4385files/3080390048bytes、高度12件は4035files/3122382940bytes。正式は保存先だけを事前固定し、27basic/endpointsと各9セルj0/1の18heldout、計45caseを `/dev/shm/<run-name>/formal-memory-v1/` に置く。残る82heldoutは通常disk。logical formal directory内のcaseリンクはそのRAM実体を指す。分布・実行順・合否には影響しない。

回収は `rsync -aL` 等でディレクトリsymlinkを参照先内容として転送する。RAM容量/保存状態を確認し、サーバ停止・再起動前に実体を回収する。既存trialに上書きしない。回収inventoryは全階層のcase宣言を検証し、欠落root/declared subtree/dangling link/読取失敗/循環を拒否する。完了判定は転送exit codeではなく、独立した全size/SHA照合で行う。

`inventory_elevated_artifacts.py --root ... --output ...` で送信元receipt、ローカルで `--verify <receipt>` を付け別名outputへ検証結果を保存する。証拠更新後に旧inventoryを再利用しない。正式rawの欠落があれば最終captureの受入も失敗させる。

## 失敗と再開の境界

旧資格rawとseven330sはbytes/source/engine依存が同一である根拠を保存して再利用する。新高度missionの成功を旧127件から流用しない。preflight/環境不足/reader整形差の失敗はrun内に保存した。PPOのpartial receiptは完全optimizer/RNG/env stateではなく、再開可能checkpointとは扱わない。有限学習は完了しているので追加学習しない。

held-out露出後のソース・controller・座標・camera・モデル変更で合格へ合わせない。必要なら旧結果を失敗として保持し、新規に宣言したcampaignと未使用held-outを要する。既に露出したcohortをfreshと呼ばない。

最終動画の撮影時HUDは「機械受入済／人手確認待ち」を保持する。利用者の後発返信は、3MP4の正確なSHAへ結び付けた別receiptに保存し、動画bytesを書き換えない。人手確認は高度上下、全assembly、ズームの見やすさ、吊り荷/箱の挙動、文字、連続1倍速を対象とする。

関連source/tests/config/spec/docs/evidenceのみ指定identityでlocal commitする。無関係な既存untracked/private/large出力は除外する。push・PR作成・merge・公開は禁止。

旧matched-previewのXMLが参照する絶対assetパスを解決するため、専用環境内の `/workspace/suspended_mujoco_final_20261025_r3a/inputs/cf2x` を今回復元した同一CF2Xフォルダへaliasした。元XML/asset bytesは変更しない。これは旧サーバの稼働を仮定したものではない。再構築時は保存したXMLのassetパスとハッシュを確認する。

freezeの広いファイル列挙には転送時Macメタデータ42件も含まれた。凍結後のsource削除・再整形は行わず、`frozen-metadata-v1/path-map.json` と元bytesを別回収した。実行source/config/tests/requirementsとmodel/schema376件はローカルとfreezeが全SHA一致。メタデータはGitへ追加しない。freezeの完全再現時だけ保存mapの元位置へ復元する。

## 今回の確定結果と成果物の入口

F43のrouting held-out300は全件完了・invalid0、平均345.061332580（B比−2.944672%）。これは物理分母とは別。物理は新root10262501で一回のみ実行し、別processの完全再監査でも9/9 basic・18/18端点・100/100held-out。正式全rawは39104files/25899565860bytesを回収・照合済み。物理集計は `formal-v1/independent-statistics.json`、各caseの完全input bindingは `formal-v1/independent-audit/`、数値と経路のderived要約は `formal-physical-summary.json` と `formal-exact-boundaries.json`。全127高度系列は `formal-v1-derived/`。

最終動画はrun直下 `final-normal-media-v1/hud.mp4`、`final-emergency-media-v1/hud.mp4`、`final-cost_shift-media-v1/hud.mp4`。各directoryにはnative/HUD両方のFFV1可逆masterとMP4、全frame/PTS/recomposition/caption/source/asset証明を保存する。全14526frame・右detail約41.50%拡大・全assembly最小91.43px。正確なbytes/SHAは [動画要約](evidence/elevated/final-media.json) と各 `video-proof.json`。

元目標boundのrはbox/cylinderには不足することが判明した。frozen checkerを変更せず、`verify-marker-bounds-v2.py` の補足解析を追加し、実scene/modelと全native frame数、開発のfrozen入力hash、最終proofのtrial/source/freezeへの結合を検証した。全18件でsqrt(3)r＋実scene変換誤差の球に包まれたmarker余白は最小84.53px（元基準40px）。`marker-geometry-final-supplement-v2.json` を正式補足とし、元の過小評価receiptとv1を保存する。この固定cameraでの実測証拠であり、汎用checkerが全shapeを正しく扱うとの主張ではない。source-onlyの独立レビューで2点のreceipt結合不足を直し、再実行と再レビューを完了した。

全3本のfresh全編連続1倍速について利用者が各項目問題なしと確認した。[人手receipt](evidence/elevated/human-review.json) はexact SHAへ結合済み。撮影時の「人手確認待ち」表記を保持するのは記録の整合性のためで、未確認という現在状態を表すものではない。

strict shadow0/1270、最大過渡箱spin338.476deg/s、正の経路exact gap385/1270を保存する。動画優先の連続1秒静定合格を、飛行中ずっと静止・strict mission成功・最適経路と読み替えない。random routingの71悪化caseと19 harmful eventも [結果](elevated-payload-result-ja.md) に報告した。

## 再検証と再構築

既露出cohortを再飛行してfresh評価と呼ばない。回収済み成果物の再hashやrawの再解析を実行できる。新しい物理実験・source/camera/model変更は別宣言と未使用cohortを準備する。

- `executed-source-v6.tar.gz` は正式実行source、`routing-executed-source/` は実際の学習source。各caseにも実行source snapshotがある。後付けdocs補足scriptとは区別する。
- `runtime-evidence-v1/` は実MuJoCo209source/library bytesとpath-map、`encoder-runtime-v1/` は実encoder binary、`system-dependencies-v1.json` はpip/OS/ffmpeg/shared library情報。OS全体の完全bundleとは主張しない。
- `server-setup-evidence-v1/` は今回使用したsource/input archivesとinstall logs。旧GPUサーバは不要であり、指定された新サーバで復元・CPU実行した。特定サーバの継続稼働を再現条件にしない。
- `joint-freeze-v1.json` SHAは `f2fbe39ebf90d55507627d4a4d2d9178b1ef8800784cd08d5f6ecf7b4bc3e6f6`。source/config/tests/requirements/model/schema376件をlocalと再照合済み。広いfreezeに含まれたMacメタデータも別保存済み。
- 全体回収は `remote-final-run.inventory.json` を送信元正本とし、ローカルで `inventory_elevated_artifacts.py --verify` を使用する。自分自身のinventoryはroot外へ保存し、再帰を避ける。追加のローカルderived資料・人手receiptを含む最終inventoryも別途作成する。
- connection resetでmaster転送が中断したため、転送のみ再開した。`postprocessing-transfer-interruptions-v1.json` と `recovery-resume-v2.log` に保存する。学習・物理caseの再試行や入力の置換は行っていない。

指定サーバnamespace全体の回収照合は **59788files・36851003632bytes、全サイズ/SHA-256一致**。`remote-final-run.inventory.json` と `remote-final-run.local-verified.json` を参照。最終3動画への人手確認も合格している。ローカル解析・人手receiptまで含めた受渡しinventoryはrunと同じ階層の `suspended_mujoco_elevated_20261025T141122Z.delivery-inventory.json` に保存する。

最終ローカル目録は **60127files・36921663846bytes**、全remote entryと一致、ローカルdirectory symlink0。目録自身のSHAは [回収receipt](evidence/elevated/recovery.json) に保存。これによりサーバのRAM45caseを含む必要実体はローカルの通常fileとして保持済み。サーバ停止後も回収物の検証・解析が可能。
