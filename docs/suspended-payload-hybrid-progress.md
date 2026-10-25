> **Revision3 machine progress:** Development15/15, validation12/12 (120legs=96transfers+24holds), unchanged C50+paired20 and all7×330s full native/RK replays passed the final prerequisite audit. Final freeze canonical digest is `6e5e331d8a59d5ebefdafd264eb77ebde8891e5c5d26d4fc37b87aba4645c4a2`. Formal127 completed once per case with unchanged control/source: basic9/9, endpoints18/18, heldout100/100 and evidence127/127 independently accepted. No held-out tuning. Earlier v1 reader failure and v2 pre-formal render-environment interruption remain preserved. All three enlarged-fixed-overview final videos passed full machine audits and explicit fresh continuous1× human review. All3608selected files/8,641,550,668bytes plus7519duration files/5,137,376,485bytes are locally size/SHA verified. Related local commit is the last closure step. The older Revision2 narrative below is historical and does not authorize Genesis/H work.

> **Camera correction:** User rejected the close-follow enlargement as a misunderstanding. The required main view is the **original fixed overview with all10destinations**, zoomed to reduce sky/floor. The close-follow video remains preserved as a display experiment and is **not camera-requirement completion**. Correct fixed overview reframe completed and locally verified: `outputs/suspended_mujoco_final_20261025_r3a/overview-normal-development-v1/hud.mp4`,11,660,695bytes SHA256 `fee710297af62c71abf81fcc9bb668bc50d16cdc03da8c7c4af481d41cc272eb`. All10destinations remain in the fixed main view; individual detail is unchanged. Zero new physical successes.

> **Rejected close-follow display experiment (historical):** `outputs/suspended_mujoco_final_20261025_r3a/close-normal-development-v1/hud.mp4`:45.30 physical seconds,4531frames,10/10 existing development destinations;15,603,477bytes SHA256 `aa985433a23160564d2f7b959fb8c265c10edacca66cc67873f52a3b2ccda187`. Exact full native/HUD recomposition,100Hz PTS, all-frame conservative body bounds and zero inset occlusion verified. Camera-only reuse: **zero new physical successes**; user rejected this layout; it is not the requested enlarged overview. Local qualification raw2669files/7,723,572,007bytes rehashed exactly.

> **2026-10-25 Revision3 scope override:** User selected MuJoCo/C-only completion. Genesis experiments and NN/parallel infrastructure work stopped. G2 powered-base was interrupted by scope change at60 simulated seconds; neither pass nor numerical failure. Previous results below remain historical. C70, duration7, development15 and validation12 passed the Revision3 freeze; formal127 passed127/127 and the fresh human final-video review passed. See the [current plan](superpowers/plans/2026-10-25-suspended-mujoco-completion.md).

# 吊り荷 Hybrid Revision 2 実行記録（進行中）

対象仕様は [Revision 2](../specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md)。本書は開発途中の記録であり、最終受入完了報告ではない。旧作業の失敗判定は変更していない。

## 実際に確認した成果

再照合した MuJoCo 3.15.0 / float64 / RK4 / 16 substeps / N=1 と、固定公称設計 `C-local-lqr-v1` による開発実証は次のとおり。飛行 NN は使用していない。

| 開発シナリオ | 実際の巡回完了 | シミュレーション時間 | reset から終端記録までの wall 時間 |
|---|---:|---:|---:|
| normal | 10/10 | 45.30 s | 17.323994 s |
| emergency | 10/10 | 44.50 s | 17.0622 s |
| cost_shift | 10/10 | 44.92 s | 17.3561 s |

全3動画について、全100 Hzフレームの native pose、raw/HUD のソフトウェア再構成、PTS、ローカル回収時のサイズ・SHA-256を確認した。保存先は `outputs/suspended_hybrid_20261024_r2a/development-{normal,emergency,cost_shift}-video/hud.mp4`。**開発実証動画**であり、最終受入済み動画ではない。1倍速の人手確認は未完了。

参照系の制御診断は C0=10/10、C1=10/10、C2=30/30、全試行で安全条件を維持した。振動フィードバックを切った C0 は0/10で、全試行が静定せず leg30秒で終了した。刻み幅を半分にした C0 は10/10。報酬改善による代用判定は使用していない。

C1/C2全40試行を回収済み raw から再監査し、Sでの静定位置から移動先での静定位置までの実変位を確認した。C1の最小は0.5033643056 m、C2の最小は0.5018018376 m。帰路もそれぞれ最小0.5066191140 m、0.5057998634 mだった。[目標改変試験と全試行の実変位](evidence/suspended-hybrid/target-and-motion-audit.json)を保持している。

新しい動画優先条件の成功と旧厳格条件の診断は区別している。normal 開発動画の旧厳格条件による同時静定は0/10。経路も厳密最適とは呼ばない。normal の既存目的値は299.571703333、同じ目的関数による厳密解は269.212316440、差は30.359386893であった。

## Genesis の確認状況

| 実行構成 | 確認済み | 未達・未確認 |
|---|---|---|
| GPU G1: float64 / implicitfast / 256→512 | 受動32、通電18、刻み幅比較の各単体トレース | 比較6組すべてで位置差2.172–2.309 cm > 2.00 cm。all62不合格 |
| GPU G2: float64 / implicitfast / 512→1024 | 受動32、motor15 | 通電・比較継続待ち。保存容量確保後、source/engine SHA一致を再確認して同一PIDのRAM状態から再開中 |
| CPU: 1 thread / Scene N32(base), N8(half) / 512→1024 | 全62条件、motor15、短い N1 C0 診断1試行。比較6組の最大位置差1.086–1.154 cm | この構成の結果を別の並列構成へ流用しない。通電・比較raw全331ファイル・23.36 GBをローカルsize/SHA検証済み |
| CPU: 実際の8 OSプロセス、各 Scene N1 | 短い同一入力試験で8 PIDの同時実行と全native状態の一致。専用の受動32条件にも合格 | 通電段階の記録が保存不足（ENOSPC）で停止。全62条件は未完了。motor15とsource-v4のreset30・分離・16独立再生に合格。記録専用flushを含む新sourceの確認とC0〜C2は未完了 |

短いCPU N1 C0は4.78 simulated seconds、62.0842 wall secondsで成功したが、これは1試行の成立性診断であり、完全なC0、8環境の制御gate、学習開始許可ではない。必要な記録を含む実際の10地点飛行がwall600秒内に収まるかはまだ確認していない。

実装中の8プロセス版では各ワーカーに3形状の永続bankを用意し、選択した荷物の質量・慣性と元の規則による減衰だけを更新する。NN/制御器に真の荷物条件を渡さない。新規構成について数値資格、motor15、reset30、継続中の隣接環境との分離、C0/C1/C2を確認するまで optimizer は開始しない。

8プロセス版の最初の全62条件試行は、CPU N32/N8の完了rawと同時に保持する容量の見積もりが不足した。受動32条件の独立合格記録と失敗した通電試行を保存し、全62条件の成功には扱わない。完了したCPU証拠の回収・照合・重複stage退避を先行し、次の試行には保存容量の事前条件を設ける。

## 証拠の保存と残作業

旧資格raw 2,669ファイル・7,723,572,007 bytesを再照合した。新規開発動画の全証拠165ファイル、G1通電・比較の全証拠324ファイル・11,899,806,947 bytes、追加Genesis証拠437ファイル・163,062,177 bytesをローカルに回収し、各ファイルのサイズ・SHA-256を検証した。保存容量を確保するため、新規サーバ側の重複stage/mediaだけをローカル照合後に退避している。旧raw・checkpoint・RAM保持プロセスは削除していない。

実行依存9 packageのコード/nativeファイルのhashを回収した。[要約](evidence/suspended-hybrid/runtime-provenance.json)に示すとおり、wheel RECORDとの不一致はなかった。wheelに含まれないGit commitは不明と明記し、release versionと実際のファイルhashでbuildを固定する。

未完了項目は、主系選択、実行系の全gate、有限C/H比較、330秒連続試験7条件、branch別validation・正式受入、正式な3動画と人手確認、日本語最終報告/handoff・英文PR description、最終レビュー・local commit。validation/held-outはまだ開いておらず、学習・採用判定も行っていない。事前固定した[12条件の開発比較manifest](evidence/suspended-hybrid/paired-development-declaration.json)と選択規則の実装は準備済みだが、比較自体は未実行。

## 永続bankのリセット診断で分かったこと

8プロセス・各Scene N1で、専用motor15は全15条件に独立合格した。続く最初のreset30診断はraw実行後の監査で失敗した。Genesisは各`scene.step`末に外力を消去するため、「直前に実際のモーター入力があれば外力fieldが残る」という監査側の仮定が成り立たなかった。実測ではRPMと運動は非ゼロ、外力fieldはゼロだった。これを合格へ読み替えていない。

次の診断では、直前の実STEPの後、元の4モーター入力を積分せずに一度だけ準備し、その保留中の力・トルクがリセットで消えることを直接検証する。姿勢・速度・制御器・時計を変えず、独立に元のKF/KM・回転方向・COMを再構成する。専用sidefileを持つこの診断は飛行・C gate証拠として拒否する。修正の単体試験62件は合格したが、source-v4の新しい物理リセット30試験と独立再生16本は合格し、native値・全内部stageの差は0だった。後続の記録専用flush追加は別sourceとして再確認する。
