# 適応型ドローン巡回点検

2026-10-22：吊り荷の回復実装を進めましたが、**物理資格・学習・受入・動画は未達**です。[回復spec](specs/2026-10-22-suspended-payload-flight-recovery/spec.md) に基づき、実GPUで旧失敗を再現し、積分離散化を独立CPUモデルで切り分け、細かいdtを調べました。高さ・荷物分布・床・settling・評価分母は変更していません。

旧V10代表rawは新規GPU再現とSHA-256まで一致し、motor0でもenergyが最大530.703%増加しました。128-substep候補の30秒受動3traceは正energy増加5%以内でしたが、170度では約13.0%のenergy損失、90度のdt/半dt間には最大0.642m・約180度の軌道差が残ります。受動比較をpoweredの合格判定へ読み替えず、残29受動・全powered・motor15・reset30・batchを未資格としています。

native core、100区間の共有settlingを使うmission、3寸法×N環境のbatch backend、総遷移数/PPO予算・独立episode RNG、raw先行保存と異常rollout隔離をCPU実装しました。実trainingへの共通化、完全62 admission、validation/freeze、254件coordinator、camera/HUD・全frame1x検査は未完です。production入口は閉じたままです。

記録込みの実測では、新規N1完全62資格だけで約18645.3秒を要する予測に対し、残GPUは17384.216263秒です。後段を含められないため、specの停止条件に従って依存GPU作業を停止しました。batchによる未測定の高速化や22%稼働率からの倍率を仮定していません。

- 両学習：各0/300000総遷移。旧model2000・pilotからの再開はありません。
- validation：0/12条件・0/120実遷移。paired受入：0/254件。
- 新しい全編raw/HUD動画：0/3本。新規人手確認は未依頼です。
- 成功率・Wilson区間・randomizedの優位性は算出していません。

[今回の引き継ぎ](docs/suspended-payload-flight-recovery-handoff.md)、[全gate](docs/evidence/suspended-recovery/acceptance.json)、[実物理検査](docs/evidence/suspended-recovery/qualification.json)、[GPU集計と予測](docs/evidence/suspended-recovery/gpu-summary.json)、[日本語レポート](docs/report_ja.md) を参照してください。旧spec・旧handoff・失敗モデル・旧rawは履歴として保持しています。

[全6ページ確認済みPDF](outputs/suspended_recovery_20261022_112649_abce93/report_ja_recovery_v1.pdf) と [検証receipt](docs/evidence/suspended-recovery/verification.json) は、未達を含む今回の結果です。PDF・raw・archiveはローカル成果物です。

## 再実行と成果物

本実装はルートのsrc/scripts/configs/testsが対象です。歴史コピーと個人ファイルは含めません。大容量成果物は一意な `outputs/suspended_recovery_20261022_112649_abce93/` に保存し、Gitにはコード・テスト・必要文書・compact evidenceだけを含めます。接続情報と機械固有pathはGit外です。

```bash
export PYTHONPATH=src
.venv-routing/bin/python -m pytest -m 'not gpu' -q
.venv-routing/bin/python -m ruff check src scripts tests
.venv-routing/bin/python scripts/diagnose_payload_planar.py \
  --probe outputs/suspended_retry_20261022_052336/delivery-final/outputs/qualification-v3-domain/large_0.060_0.10 \
  --substeps 2 --out outputs/NEW_PLANAR_RECEIPT.json
```

GPU再開には現在有効な非公開接続先と、新規workspace・Python3.12の別flight/CPU routing環境を使います。既存の `requirements/flight-linux.lock.txt` と `requirements/routing-scalable-linux.lock.txt` を維持し、完全資格・実batch隔離・全後段の測定予算が成立するまでoptimizerを開始しません。再実行の具体的な順序と未達は引き継ぎに記載しています。

累積8時間は継続し、今回6jobの591.655849秒を含めて、使用11415.783737秒・残り17384.216263秒、全200job・active0です。初期化・失敗・再試行を消さず、push・PR作成・merge・動画公開・コース提出は行っていません。

## 継承済みの無荷物結果

分散3D配置と整数重要度では、採用model4296/seed44の固定9mission・90点検、全編1x動画の独立検査・人手確認が過去に成立しています。これは吊り荷の実績ではありません。[分散配置handoff](docs/dispersed-destination-layout-handoff.md)、[整数重要度handoff](docs/integer-priorities-handoff.md)、[第三者ライセンス](THIRD_PARTY_NOTICES.md) を参照してください。
