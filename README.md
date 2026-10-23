# 適応型ドローン巡回点検

2026-10-23：吊り荷solver recoveryの[新spec](specs/2026-10-22-suspended-payload-solver-recovery/spec.md)に基づく作業を、ユーザーの方針変更指示により未達を残して終了します。MuJoCo3.15.0・float64・RK4・control.01秒・16substepsのN1経路で、全62trace、motor15、実reset30、初期姿勢reset30を記録し、独立raw再構成を通過しました。元機体・motor・0.30m剛体リンク・荷物分布・床・settling・timeoutは維持しています。

完全資格後、pilot5回の計1,020,000遷移と、fresh4296・fresh optimizerによる両方式各3,000,000遷移を実学習しました。計7,020,000遷移で共有settling完了は0でした。30万／100万／300万の全rungでrandomized・fixedとも0/6、全36飛行は最初のhover30秒timeoutでした。全36件のnative再検査と全raw配送・全memberのsize/SHA・永続化は通過しましたが、学習方策は採用していません。[比較結果](docs/evidence/suspended-solver/paired-development-curve.json)を保存しました。

- 正式validation：0/12条件・0/120実遷移。paired受入：0/254件。
- 合格方策の全編raw/HUD動画：0/3本。新しい人手確認はpending。
- N8/32/128は診断入口を実装済みで、vector PPOの採用資格はありません。旧N1の物理資格は新backendやbatchの資格ではありません。
- 成功率・randomizedの優位性・動画承認はまだ報告できません。

[実行handoff](docs/suspended-payload-solver-recovery-handoff.md)に、修正候補、不採用試行、raw/source/hash、未達gateを記録しています。旧spec・旧handoff・台帳・失敗checkpointは履歴として保存しています。GPU累積時間上限と使用量trackingは撤廃済みです。同時1job、通常missionのwall600秒とdeadlock停止は維持します。

## 再実行と保存

今回の出力は `outputs/suspended_solver_20261022_e265f60d/`。大容量raw、checkpoint、動画、archiveはGit外に置き、転送時にsize/SHA-256を照合します。学習時の実source archiveと最終CPU sourceを区別します。コード・テスト・仕様・必要文書・compact evidenceは、今回の作業が揃った時点で1件の新規local commitにまとめます。

```bash
export PYTHONPATH=src
.venv-routing311/bin/python -m pytest -m 'not gpu' -q
.venv-routing311/bin/python -m ruff check --select E4,E7,E9,F src scripts tests
```

実backendの各入口は、完全資格receiptだけでなく、全raw・実solver build・現在sourceを再構成してから実行します。旧PASSの差し替えや、statusのみのfinal計画・freezeでは解除しません。実行手順は[公開指示](docs/suspended-payload-solver-recovery-instructions_ja.md)と[計画](docs/superpowers/plans/2026-10-22-suspended-payload-solver-recovery.md)を参照してください。接続情報と機械固有pathは含めません。

[日本語report](docs/report_ja.md)と[終了時handoff](docs/suspended-payload-solver-recovery-handoff.md)に、数値資格の達成と飛行未達を分けて記録しています。2026-10-23のユーザー指示により旧方針の追加学習を終了し、[ハイブリッド制御・Genesis優先の提案spec](specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md)を精査しました。[レビュー](docs/suspended-payload-hybrid-genesis-review_ja.md)には最終計画前の未確定事項を記録しています。新specの実装・実験は開始していません。[終了時PDF](outputs/suspended_solver_20261022_e265f60d/report_ja_solver_closure_v1.pdf)は全7ページを描画・目視確認しました。経過PDF v1〜v4も履歴として保持します。push、PR作成、merge、動画公開、コース提出は行いません。

## 継承済みの無荷物結果

分散3D配置と整数重要度では、採用model4296/seed44の固定9mission・90点検、全編1x動画の独立検査・人手確認が過去に成立しています。これは吊り荷の実績ではありません。[分散配置handoff](docs/dispersed-destination-layout-handoff.md)、[整数重要度handoff](docs/integer-priorities-handoff.md)、[第三者ライセンス](THIRD_PARTY_NOTICES.md) を参照してください。
