# MuJoCo 吊り荷巡回の結果（受入・映像・回収・人手確認済み）

利用者の2026-10-25指示と [Revision3](../specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md) に従い、MuJoCo＋制御器単独 `C-local-lqr-v1` を最終方式とした。**正式127/127条件が10地点完走・全地点の動画優先静定に成功し、全試行の独立証拠再構成にも合格した。** 最終3動画の全検証・全必要成果物のローカル回収とSHA照合、および利用者によるfresh連続1倍速確認も完了した。

## 実測の受入結果

| 対象 | 結果 | 根拠・制限 |
| --- | --- | --- |
| 旧資格 | source・実binary・モデル・設定・rawを再照合 | 元2669files／7,723,572,007bytesを全件ローカル再hash |
| C0/C1/C2 | 10/10、10/10、30/30 | C1実移動最小0.503364306m、C2最小0.501801838m。帰路も0.50m超 |
| paired診断 | feedback-off C0=0/10、refined32 C0=10/10 | 旧失敗は失敗のまま保存 |
| 開発頑健性 | 15/15、全10地点成功 | 公称3、独立内部荷重6、端点6。制御の再調整なし |
| 長時間powered | 公称＋6端点、7/7で330秒 | 各33000区間、一度だけreset、途中resetなし。巡回試験ではない |
| validation | 12/12条件、120指示leg | 継承座標列の96異座標変更＋24初期・重複保持。120異座標移動とは呼ばない |
| 正式basic | 9/9 | 各条件1試行 |
| 正式端点 | 18/18 | 各条件1試行 |
| 正式held-out | 100/100 | Wilson95%区間[0.963006502,1.0]。調整・教師データに未使用 |
| 全正式試行の証拠 | 127/127で完全 | native・制御・mission・最終flushを全件独立再構成 |

正式試験の完了時刻は43.31～51.24物理秒、中央値45.14秒。resetから最終flushまでのwallは最大20.054756065秒で、600秒を下回った。安全・leg30秒・wall600秒の失敗、欠損証拠、再試行は0件。シナリオ別はnormal43/43、emergency42/42、cost_shift42/42。サイズ・荷重帯別分母と信頼区間、全127結果を [最終要約](evidence/suspended-hybrid/mujoco-final127.json) に保存した。

330秒試験では全231000区間・14784000RK内部段を独立再生し、nativeとstageの最大差はともに0。最大signed-work残差3.971045128e-12J、拘束残差8.881784197e-16m。[前提監査](evidence/suspended-hybrid/mujoco-final-prerequisites.json) と [開発27条件](evidence/suspended-hybrid/mujoco-development27.json) を参照。

## 固定した方式と変更しない条件

MuJoCo3.15.0／float64／RK4／内部16substeps／外部100Hz／Scene N=1。元CF2X、有質量0.30m剛体リンク、両端ball joint、減衰、モータ上限・KF/KM、目標座標、既存routing NNと目的関数は変更しない。制御器は理想シミュレータセンサだけを入力し、真の荷重・寸法・慣性・seedを実行入力にしない。飛行NNは使用せず、NN学習は今回の完成範囲外。実機転移、雑音・遅延への頑健性は未証明である。

`payload-settling-video-v1` は荷物速度≤0.15m/s、振れ角速度≤15°/s、箱の全角速度≤30°/sと、既存の位置・速度・傾き条件を同時に連続1秒要求する。各leg30秒・resetから最終flushまでwall600秒。合計120秒／12000stepのmission上限は新経路に存在しない。旧defaultsと歴史的判定は維持する。

canonical freeze digestは `6e5e331d8a59d5ebefdafd264eb77ebde8891e5c5d26d4fc37b87aba4645c4a2`。freezeファイルSHAは `b464c322578519f9eb4367fcd70c65ffa6eb719994cb4f3212534ca5a1ec8d3b`。実行時60sourceのSHAと作業ツリーは全件一致した。正式全監査ファイルは2,965,334bytes、SHA `59f3dfd081bc13631534ba299a5c0a6d768ca99a2ae7021a81d0ff45835d827e`。

## 動画と画角

利用者が求めたのは**全10地点が収まる元の固定俯瞰主画面を拡大すること**である。俯瞰距離10m→5m、look-atを調整して空・床の余白を減らした。元の個別拡大図512×720、距離1.25mは変更しない。

正しい開発版は `outputs/suspended_mujoco_final_20261025_r3a/overview-normal-development-v1/hud.mp4`。既存成功raw45.30秒／4531framesの表示改善であり、新しい飛行成功数0。11,660,695bytes、SHA `fee710297af62c71abf81fcc9bb668bc50d16cdc03da8c7c4af481d41cc272eb`。

先に作ったclose-follow動画は意図の取り違えとして保存し、画角要件の達成には数えない。`close-normal-development-v1/hud.mp4`、15,603,477bytes、SHA `aa985433a23160564d2f7b959fb8c265c10edacca66cc67873f52a3b2ccda187`。

最終版は正式basic seed1のnormal43.94秒、emergency44.52秒、cost_shift44.95秒を同じ改善画角で再描画した。新しい飛行・成功数は増やさない。100Hzの全native姿勢とHUD、全フレームの身体・全10目標の画面内収まり、PTS、29字幕領域、元meshを含む全依存assetとsourceを検証した。raw/HUDの可逆masterと視聴用MP4の検証を区別する。最終映像には実controller・backend・profileと「機械受入済／人手連続1倍速レビュー待ち」を表示する。最終3動画の機械検証と視聴用MP4のローカルサイズ/SHA照合は完了した。可逆masterを含む全必要成果物を回収・照合し、利用者は「3本すべて連続1倍速で確認し、問題なし」と回答した。確認対象MP4の3つのSHAと結び付けた [人手確認記録](evidence/suspended-hybrid/mujoco-human-review.json) を保存した。撮影時の「人手確認待ち」HUD・機械proofは歴史的状態のまま変更しない。機械証拠は [最終3動画要約](evidence/suspended-hybrid/mujoco-final3-media.json) に保存した。

## 旧厳格条件と経路の限界

同じ正式raw上の旧厳格静定は0/1270地点。開発頑健性とvalidationも0/150、0/120である。動画優先条件で次目標へ進んだ実際のイベント列の診断であり、旧条件の反実仮想巡回を主張しない。旧失敗を成功へ書き換えない。

正式1270意思決定で、既存目的関数の厳密解に対するpolicy suffix差は0～31.112072064。数値差1e-8を超える正の差は679判断。正式normal初回は299.924038927対269.593841614、差30.330197313。**最適経路とは呼ばない。** 見越していない将来eventや実物理の時間・エネルギーに対する最適性も主張しない。旧normal開発例の差30.359386893は元記録のまま残す。

## 保存した失敗・中断と回収

旧paired診断のC0をC1と取り違えた最終readerを修正し、旧manifestの初期幾何再生成で発生した約5e-18mのNumPy丸め差は、元manifestのSHAとそこに保存された厳密caseを使って解消した。rawとcaseの厳密照合は緩めない。元v1の3readerを実SHAで限定した読取専用derivationを別receiptに保存した。制御・物理・routing等の差や旧bindingでの正式実行は認めない。v1 freeze失敗、v2のformal生成前の描画環境不足による安全中断、v3の実行source・設定・ログを保全した。

Genesisは利用者方針変更で停止し、G2の60秒partial、prepared ABA、未実行v5 source等を保存した。中断は数値合否ではない。旧run・checkpoint・source binding・保護RAM keeperは変更しない。[Genesis中断記録](suspended-payload-genesis-interruption.md) に既存成功・失敗・未確認を記載した。

回収先は `outputs/suspended_mujoco_final_20261025_r3a/`。開発27条件499files／829,569,113bytes、長時間7条件7519files／5,137,376,485bytesは全サイズ・SHA一致済み。元asset15件と全入力20件も一覧化した。正式127条件2796files／3,973,046,459bytesも全件回収・サイズ/SHA照合を完了した。遠隔manifest自身のSHAとも一致した。大容量転送の `Connection reset by peer / Broken pipe` は部分ファイルを保持してchecksum付きで再開し、転送終了コードだけで完了としない。

## ソフトウェア確認と完了範囲

ローカルCPU全体テスト `pytest -q -m 'not gpu'` は1189合格・20skip・1GPU除外、288.59秒、終了コード0。actual MuJoCoのremote物理監査と混同しない。現行final/duration/final-media/overviewと該当entrypointのRuffも合格。staged空白検査は凍結済みoverview renderer末尾の空行1件を検出した。撮影sourceのSHAを保つため原文を維持し、当該EOF空行以外の空白検査は合格した。[テスト記録](evidence/suspended-hybrid/mujoco-revision3-tests.json) を保存した。未完了Genesis/H専用テストはsourceごと保全して対象外とした。

今回の必須の受入・映像・回収・fresh人手確認はすべて完了した。新run3608files／8,641,550,668bytesと別管理の330秒raw7519files／5,137,376,485bytes、計11127files／13,778,927,153bytesをサイズ・SHAで全件照合した。[回収記録](evidence/suspended-hybrid/mujoco-local-recovery.json) を参照。日本語報告/handoff・英文PR description・関係コードとテストを、指定された `dstaka <22057509+dstaka@users.noreply.github.com>` のAuthor/Committerでlocal commitに記録する。[handoff](suspended-payload-mujoco-handoff.md)、[英文PR description](suspended-payload-mujoco-pr-description.md) を参照。push・PR作成・merge・公開はしない。

最終視聴用MP4のサイズ/SHA：

| 動画 | frames | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| basic-normal-1 | 4395 | 11245806 | `f52fd9f506e416719a51f081ce3640c88044848d0b6ab7f6685bca10f5576b9f` |
| basic-emergency-1 | 4453 | 11459540 | `4aae86a49d773d2828d0ba5a1852006d0c901fbf281afb27158b22f722784a99` |
| basic-cost_shift-1 | 4496 | 11194467 | `a80f15427d0d36263ff6051e39aa3d950cca8c6b7d94457a5ae615ad153b1f95` |

初期frame0から最終frameNまで100Hzで収録し、最後は通常の1表示期間0.01秒を持つ。表示秒数は物理最終時刻より0.01秒長く、追加の物理STEPや時間省略はない。撮影時の機械proofと後発の人手確認receiptを併せて読む。
