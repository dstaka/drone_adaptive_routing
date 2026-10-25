# MuJoCo 吊り荷巡回 handoff（完了）

現行仕様は [Revision3](../specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md)、結果は [日本語報告](suspended-payload-mujoco-result-ja.md)、提案文は [英文PR description](suspended-payload-mujoco-pr-description.md)。利用者がMuJoCo＋制御器単独を明示選択した。Genesis、新規NN、C/H比較、GPU利用実証へ戻らない。旧指示・自動helperを新しい実行開始命令として扱わない。

## 完了状態

- 元MuJoCo資格・C0/C1/C2をactual source/build/設定/モデルと再照合して再利用。C0 10/10、C1 10/10、C2 30/30。C0 feedback-off0/10の旧失敗を維持し、refined32は10/10。
- 新開発頑健性15/15、validation12/12。継承座標列の120指示legは96異座標変更＋24初期・重複保持であり、120異座標移動とは呼ばない。
- 公称＋6端点の330秒powered7/7。全231000区間・14784000内部段でnative/stage再生差0、途中resetなし。巡回成功には数えない。
- 正式127条件を各1試行で全件成功。basic9/9、端点18/18、held-out100/100、全127証拠の独立最終監査も合格。調整・教師へのheld-out使用なし。
- 最終normal/emergency/cost_shiftの全100Hz native1×動画を作成し、全フレーム姿勢/HUD、PTS、字幕、全身体/全10目標の収まり、外部assetとsourceを照合。主画面は全10地点が入る固定俯瞰の拡大、元個別拡大図は据え置き。
- 利用者は3本すべてを全行程連続1倍速で確認し「問題なし」と回答。確認した3つのMP4 SHAを別receiptに固定。撮影時の「人手確認待ち」HUDと機械proofは過去の状態として保存し、後から書き換えない。
- 必要raw/model/video/log/config/sourceをローカル回収し、全件サイズ/SHA照合済み。必須の未達・人手確認待ちは残っていない。

## 固定方式

MuJoCo3.15.0、float64、RK4、16substeps、外部100Hz、Scene N=1、`C-local-lqr-v1`。飛行NNなし、既存routing NNは使用。元CF2X、有質量0.30m剛体リンク、ball joint、箱、減衰、モータ、目標、目的関数を変更しない。真の荷重値をcontrollerへ渡さない。

`payload-settling-video-v1`。荷物速度0.15m/s、振れ角速度15°/s、箱の全角速度30°/sと既存位置・速度・傾き条件を同時に連続1秒。leg30秒、resetから最終flushまでwall600秒。合計120秒/12000stepのmission上限はない。旧defaults・旧判定を変更しない。

canonical freeze digestは `6e5e331d8a59d5ebefdafd264eb77ebde8891e5c5d26d4fc37b87aba4645c4a2`。正式mission時間43.31～51.24秒、中央値45.14秒、reset-through-final-flush wall最大20.054756065秒。全10地点を実際に飛び、1秒静定を各地点で満たす。旧厳格shadowは同じ正式rawで0/1270。厳密経路解との差は最大31.112072064であり、最適経路とは呼ばない。理想センサ条件で、実機・雑音・遅延の頑健性は未証明。

## ローカル回収先

ルートは `outputs/suspended_mujoco_final_20261025_r3a/`。大容量成果物はGit無視対象で、文書だけではrawを代替しない。

| 資料 | 場所・照合 |
| --- | --- |
| 正式127条件 | `formal-v1/`：2796files／3,973,046,459bytes。`final-independent-audit.json` が全件合格 |
| 最終3動画 | `final-overview-media-v1/basic-{normal,emergency,cost_shift}-1/hud.mp4`。全raw/HUD masterとMP4計12本、個別proof・全3proof・source・loaded-assetsを保存 |
| 人手確認 | `human-review-v1.json`：全3動画のSHA、確認範囲、利用者の明示回答 |
| 開発27条件 | `final-development-v1/`：499files／829,569,113bytes。全サイズ/SHA一致 |
| 330秒raw | `duration-seven-v1/`：7519files／5,137,376,485bytes。`duration-seven-recovery-inventory-v1.json` と対応 |
| 新run全体 | `final-delivery-remote-inventory-v1.json`、`final-delivery-local-verified-v1.json`：3608files／8,641,550,668bytes、全一致 |
| 実行source等 | `final-executed-source-v1/` の60source、`completion-execution-source-v*/`、`final-source-v*.tar`、`final-media-source-v1.tar`、入力・モデル・全ログ・設定 |
| freezeと前提 | `final-freeze-v1.json`、`all-prerequisite-artifacts-local-verified-v1.json`：全8697参照ファイルのSHA一致（上記rawとの重複あり） |

旧C70のローカル復元場所は `outputs/suspended_hybrid_20261024_r2a/control-diagnostics-full-v1/`。元の遠隔パスとフォルダが異なるが、全705ファイルをfreezeのSHAで照合済み。元qualification2669files／7,723,572,007bytesは別rootの既存回収資料で、`reference-local-raw-verification-v1.json` に全件の実hashとsourceを固定している。

新runと330秒rawの合計は11127files／13,778,927,153bytes。旧資格、C70、Genesisの保全資料は別に保持する。[回収要約](evidence/suspended-hybrid/mujoco-local-recovery.json) を参照。

## 中断・歴史と再実行の注意

今回専用namespaceは `suspended_mujoco_final_20261025_r3a`。`complete-mujoco-final-v3.py` は前提監査→freeze→正式127一回→最終3媒体まで終了した。`completion-v3.log` と `completion-machine-result-v3.json` を読む。後者のhuman pendingは機械撮影終了時の状態で、後発の `human-review-v1.json` が明示確認を記録する。完了済みhelperやcaseを再起動・上書きしない。

v1 freezeの旧C0 paired指定と初期幾何の再生成一致問題、v2のformal生成前の描画環境不足による安全中断を保存している。元manifestの厳密caseとSHAを使い、旧v1三readerの実SHAを限定した読取専用derivationを `reader-rebinding-v2.json` に保存した。制御・物理・routing・profile・依存の差、旧bindingでの正式実行、新規開発物理実行へ互換を広げない。全元raw/binding/失敗記録は変更しない。

旧開発専用entrypointのsummary wall判定は最終flush証明の代用にしない。今回の最終方式は `run_payload_hybrid_final.py` の実close receiptと独立gateを使用した。将来の新実験は新spec/namespaceで行い、held-outを調整に再利用しない。RAM領域の330秒証拠はfreezeの参照先なので、既存資格をそのまま再監査する際は元パス/bytesを保持する。旧RAM keeperを停止しない。

Genesisは利用者の範囲変更により停止し、partial raw/source/logを全215memberのtarへ保全・ローカル照合した。中断を合否へ読み替えない。到達点・失敗・未確認は [中断記録](suspended-payload-genesis-interruption.md)。再開を今回の完了前提にしない。

## Gitと次の作業

関係するコード・テスト・spec・証拠要約・文書だけを `dstaka <22057509+dstaka@users.noreply.github.com>` のAuthor/Committerでlocal commitに残す。commit hashはGit履歴と完了報告を参照。無関係な既存の未追跡ファイルは含めない。push・PR作成・merge・公開なし、接続情報・秘密値をGit管理資料へ記載しない。

このセッションに必須の残作業はない。新たな要件や実機転移は、今回の理想センサ・物理モデルの合格を流用して達成済みとせず、別作業として定義する。
