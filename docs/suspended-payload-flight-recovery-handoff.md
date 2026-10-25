# 吊り荷飛行回復の実行引き継ぎ

2026-10-22。[回復spec](../specs/2026-10-22-suspended-payload-flight-recovery/spec.md) の人による実行指示を受け、計画提示に続いて実装・SSH・新環境・GPU診断を実行した。**物理資格・学習・254受入・動画は未達**であり、採用済みの吊り荷方策はない。[全gate](evidence/suspended-recovery/acceptance.json) を参照する。

## 起点と保護

HEADは想定どおりe3c145a096e9223804684b1e3510bba30f0f6676、branchはfeature/suspended-payload-flight-recoveryだった。stage済み差分はなく、新spec・本handoff・公開指示は未追跡、spec indexに4行の未stage差分があった。これらを保護し、今回の新commitへ含める。reset/amendはしない。個人PNG・.DS_Store・歴史コピーは変更・stageしない。

準備版の新spec/handoff/公開指示/indexは `outputs/suspended_recovery_20261022_112649_abce93/preparation-inputs/` へ元bytesのまま保存した。旧spec、旧handoff、Euler/V10証拠・failed profile・失敗checkpointは履歴として保持する。旧model2000は初期値・再開に使っていない。[実装計画](superpowers/plans/2026-10-22-suspended-payload-flight-recovery.md) に通常判断と停止条件を記録した。

## 新しい実測と診断

V10の実source archive、完全receipt、delivery archiveのhashは一致し、全1631payload member、55raw/sample trace、357176sampleを再照合した。開始時CPU source188ファイルも旧receiptに一致し、V10物理sourceとの差はprobeのvector-norm修正だった。今回の最終sourceとの差と混同しない。[forensics](evidence/suspended-recovery/forensics.json)。

新規GPUで90度・0.060m/質量比0.10・motor0の30秒を再現した。新rawのSHA-256は8601f5f5d8b6b4e6694b9ed2d1631c3ddfa81fe66cfb9308d0a16bcc4066b8c4で旧V10と完全一致、energy増加530.703%、最初の5%超過1.07秒だった。process成功を物理PASSへ読み替えない。

別実装の平面2DOF Lagrangianから、実COM質量/慣性を使い、各実観測を起点としたvelocity-first2substepを再生した。実ΔE=0.046955007284J、局所再生合計=0.046954918107J、差8.92e-8J。COM位置/速度最大差2.124e-7m/2.845e-6m/s。このtraceでは積分離散化が増分をほぼ説明するが、3D一般の根因や修理の証明ではない。[独立診断](evidence/suspended-recovery/planar-diagnosis.json)。

pinned sourceのstandalone midpointはDOF子孫/CONNECT/WELDを除外するため、CONNECTへの変更だけによるmidpoint化は支持されない。依存版・engine実装bytesは変更しなかった。CPU8/32substepには不合格条件が残ったため、128（半dt256）を別署名の有限診断として調べた。

| 実30秒受動trace | 正energy最大 | 最小energy/初期 | 独立screen |
| --- | ---: | ---: | --- |
| 90度・0.060m/0.10・128 | 0.7403% | 95.1066% | 範囲内 |
| nominal170度・128 | 1.1683% | 86.9668% | 範囲内 |
| 同90度・半dt256 | 0.6901% | 97.9775% | 範囲内 |

各3001sampleと3000件の実ゼロ外力場を再構成した。3秒の小診断2件は資格件数へ数えない。170度の約13%損失は未解決であり、90度dt間の同時刻差も最大0.642253m/約179.993度だった。これは受動の収束診断であり、poweredの2cm/2度gateへ流用して不合格数や合格数を作っていない。[全raw/source/property検査と比較](evidence/suspended-recovery/qualification.json)。

受動32中3件、全domain62中3件だけが実行済み。anchor0/9・large2/16・追加半dt1/7、powered0/18・powered-refine0/12、新motor0/15・新partial-reset0/30以上である。13%損失と大きい軌道差を無視してnative ball修理済みとは呼ばない。元free機体の質量/rotorは旧実測と同一pinned sourceを照合した範囲であり、新規15motor測定ではない。

## 実装したCPU成果と閉じた入口

- recovery62 manifestとpassive32独立barrier。実初期状態・mass/inertia/COM/collision・実dt/engine・source・preclip raw・全ゼロ外力場を確認し、重複directory/自己申告status/短診断/省略sourceを拒否する。最終powered/refinement CLIもGPU初期化前に実raw再構成を要求する。
- `payload_recovery_runtime.py`：現在action-v2、native reset/retarget、実getterのraw先行保存、100区間settling、数値/接触/拘束guard。reset後・step後の変換失敗でも現getterを保存し、episode/step完了をUNKNOWNと記録する。
- `payload_recovery_mission.py`：全10地点shared settling、30/120/600秒制限、最初の新完了後だけC43→100/F→A1→5、実pose/残タスクからのQ*/J*/suffix/反実仮想/成分コスト。CPU doublesでの論理検証に限る。
- `payload_recovery_batch.py`：3寸法それぞれ実Nscene、個別envs_idx setter、full[N,4] RPM、logical episode/target/history/mask、rawとquarantine。全slot数値異常を検出し、inactiveを学習sampleへ数えない。GPU geometry/隔離・30reset・N1等価性・memory/速度は未検証。上限128のためbase128のhalf256 batch比較も未対応で、完成した選択batch資格runnerではない。
- `payload_recovery_sampling.py`：S=N*T*K、optimizer更新/minibatch、stage経験の事前計画、invalid全rollout隔離、seed7/logical episodeに基づく独立mass/size抽選。計画の算術とCPU schedulingであり、実経験達成・実PPOではない。
- `payload_recovery_planar.py` と `diagnose_payload_planar.py`：sourceを変えずに再現できる独立CPU診断。`plot_payload_recovery.py` は実rawから図と入力hashを生成する。

完全62最終admission、training共通coreへの移行/checkpoint物理metadata、実validation12/120とfreeze、paired254coordinator/独立checker、qualified camera/HUD/全frame1x照合は未統合である。`require_recovery_runtime_admission()` は明示的に閉じており、statusを書き換えても運用には入らない。旧Euler runtimeへの新checkpoint流用は行っていない。

## 変更していない契約と全未達

全精度SとA-J・元乱数draw・相対高度は不変、delta_z=0。条件付き高度studyは物理資格未成立のため実行していない。重要度seed40029、A81/B42/C43/D64/E23/F84/G6/H98/I22/J56、alpha1/beta0.05/除数100、有向倍率1～5、C43→100/F→Aだけ1→5を維持する。今回の実shared completionやイベント時刻・Q*/J*は未測定で、旧実績を転用しない。

| gate | 今回の状態 |
| --- | --- |
| 完全物理・実batch資格 | 未完・admission false |
| randomized/fixed各300000総遷移 | 各0・未学習 |
| validation12条件/120実遷移 | 0/12・0/120 |
| randomized基本9/端点18/held-out100 | 0/9・0/18・0/100 |
| fixed比較127、合計254 | 0/127、0/254 |
| 3本の完全raw/HUD1x動画 | 0/3 |
| 全decode/画素/字幕/telemetry/新hash | 未実行 |
| 新規人手連続1x確認 | 映像がないため未依頼・未確認 |

成功率、Wilson区間、randomized優位性は未実行なので算出しない。採用loaded方策はなく、model4296を用いたfresh optimizerの両最終学習も始めていない。通常/緊急/コスト変化の旧無荷物承認を再利用しない。

## GPU・環境・予算停止

新しいworkspaceとPython3.12.3のflight/CPU routing環境を構築し、既存Linux lockの169/33版を全照合した。元engine23sourceはV10に一致し、新候補は実installed全264Python sourceを束縛した。環境構築時の容量不足とvenv再作成失敗はCUDA初期化前のCPU失敗として保存、hardlinkで同lockを同期した。旧env/path・依存版変更・レンタル追加はない。

開始194job・使用10824.12788819254秒。今回6job増分591.655848817434秒、累計 **11415.783737009973秒**、残り **17384.216262990027秒**、全200job・active0。両台帳のbytes、各elapsed合計、active不在を照合した。全初期化・実job内CPU/記録待ちを計上し、利用率で割引していない。[全attempt](evidence/suspended-recovery/attempts.json)、[予算](evidence/suspended-recovery/gpu-summary.json)。

記録を切らない実30秒loopは128で115.625/115.964秒、256で227.383秒だった。新規N1全62は約18645.3秒、3trace再利用の仮定でも残59は約18156.3秒という線形予測で、残予算を超える。さらにmotor/reset、両学習、validation、254件、撮影が必要である。物理phase3500秒の残りは約2908.3秒で、未完の受動だけでも約7449.7秒の予測となる。

未資格batchの高速化で架空の収容計画を作らず、数値品質未確定・予算不足の停止条件に従った。N1/8/32/128の実profiling、PPO timing、共通最終sample計画freezeは未達。6つの診断正常終了を全資格へ格上げしていない。

## 実行source・転送・検証

出力は `outputs/suspended_recovery_20261022_112649_abce93/`。baselineはsource-start、候補はcandidate-s128-source-v2へ固定。候補署名93794e24c9b1b688169bad71ba99bda73d27ca0d5de97b907c3b6ef6a21c47b7。最終probeのpowered32barrier、native例外保存、batch/mission等の後続CPU修正はGPU再資格していない。[source束縛](evidence/suspended-recovery/source-bindings.json)。

最終archiveは7270159bytes、SHA-256 3910f05507ac572cb3febd76d232d391ef310ef05552388383d85d867ad5e3b2、83payload memberをローカルでサイズ/SHA-256照合した。manifest自体を含むregular fileは84である。[全転送](evidence/suspended-recovery/transfer.json)。外部成果物・大容量raw・秘密/絶対pathはignored出力へ保存する。

local最終CPUは901passed・GPU1除外、lint/compile/空白検査も成功。fresh reviewの重複case/source/物性/force不足、powered前guard、batch積分例外、reset例外のraw保存不足を修正し、負例のRED/GREENと回帰を保存した。[review](evidence/suspended-recovery/review.json)。Linux3.12 exact-lockでも901passed・GPU1除外。転送漏れによる初回900pass/1failも保存した。PDF全page描画/目視・最終hashは [verification](evidence/suspended-recovery/verification.json) に記録した。

## CPU再現入口と次の具体案

```bash
export PYTHONPATH=src
.venv-routing/bin/python -m pytest -m 'not gpu' -q
.venv-routing/bin/python -m ruff check src scripts tests
.venv-routing/bin/python -m compileall -q src scripts tests
.venv-routing/bin/python scripts/diagnose_payload_planar.py \
  --probe outputs/suspended_retry_20261022_052336/delivery-final/outputs/qualification-v3-domain/large_0.060_0.10 \
  --substeps 2 --out outputs/NEW_PLANAR_RECEIPT.json
```

3traceを再検査するときは、`payload_recovery_qualification.verify_passive_probe()` にfinal-delivery内の各probe、source-candidate-v2、engine-source/genesisを渡す。旧V10は旧archive sourceと旧checkerを使い、現在sourceへ差し替えない。小診断3秒は完全資格checkerが拒否するのが正しい。

次案は、独立平面モデルと実substep/rawで数値散逸・dt軌道差を先に評価し、sourceが支持する範囲で同一物理/物性/仕事を保つ修理と記録経路を設計すること。CPU準備済みbatchを、資格を飛ばして学習へ使わない。支持された高速なqualification計画と全後段の実測予算が成立しない間はoptimizerを開始しない。coupled integratorのengine patchや依存版変更が必要なら、具体的な修正specとその変更だけの明示確認を先に得る。今回はpatch・upgrade・追加8時間枠を実行していない。

新spec・公開指示・本handoff・索引・実装/テスト/必要文書/compact evidenceを指定identityの新規local commitへまとめる。commit IDは最終応答とignored local-commit receiptに記録した。push、PR作成/merge、動画公開、コース提出、demo_full/提出用再編集は行わない。英文PR本文案は [こちら](suspended-payload-flight-recovery-pr-description.md)。

PDFは [全6ページ確認済み報告](../outputs/suspended_recovery_20261022_112649_abce93/report_ja_recovery_v1.pdf)。PNG化した全ページを目視し、日本語・図・数値・改ページ・footerの欠落やはみ出しがないことを確認した。入力・PDF・各ページhashはverification receiptに保存した。
