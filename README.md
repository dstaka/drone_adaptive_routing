# 適応型ドローン巡回点検

2026-10-21追記：**吊り荷版は物理実現性ゲートで不合格、受入未達です。** [対象spec](specs/2026-10-17-suspended-payload-flight/spec.md) に沿って受動0.30 m固定リンク、実質量・慣性・collision、episode再抽選、29観測、同時1秒settlingと評価checkerを実装しました。短いprobe後のランダム化学習で数値破綻し、2/8/16 substepの診断でも再現したためGPU作業を停止しました。採用済み吊り荷モデル、固定比較、254受入試行、全編動画3本、人手確認はありません。成功率やランダム化の優位性を主張しません。

[実装・失敗・hash・再実行の引き継ぎ](docs/suspended-payload-flight-handoff.md)、[実装計画と到達状況](docs/superpowers/plans/2026-10-21-suspended-payload-flight.md)、[独立raw検査](docs/evidence/suspended/failure-check.json)、[日本語レポート](docs/report_ja.md) を参照してください。GPU累計7,484.093172秒、残り21,315.906828秒。旧い短時間PASSで訓練を再開できないよう停止状態を記録しています。数値破綻の修理は未達です。

以下は**先行の無荷物・分散3D版**の成果と再実行記録です。吊り荷の成功や動画承認を表しません。

A〜Jの実目的地を不規則な3次元配置へ変更し、同じ学習済みrouting方策と互換な飛行方策で巡回します。重要度は実値・表示とも1〜100整数で、大きいほど重要ですが訪問順そのものではありません。

[今回の承認済みspec](specs/2026-10-17-dispersed-destination-layout/spec.md)、[実装計画](docs/superpowers/plans/2026-10-20-dispersed-destination-layout.md)、[モデル・測定・失敗・再実行手順](docs/dispersed-destination-layout-handoff.md)、[日本語レポート](docs/report_ja.md)を参照してください。[整数重要度](docs/integer-priorities-handoff.md)、[経路最適化](docs/scalable-routing-handoff.md)、[全地点飛行](docs/visible-flight-handoff.md)、[旧1倍速動画](docs/realtime-video-handoff.md)は歴史記録として保持しています。

## 配置と固定契約

S=(0,0,1)、10地点の範囲はx±2.7 m、y±1.8 m、地面からの絶対高度z=1〜3 m。実spanは5.060577/3.410568/1.827545 m、最小3D間隔0.865248 m、高さ3帯の点数は5/2/3です。x/yの任意の閉区間0.10 m帯に3点が入らないことを、ソート後の連続3点で検査します。

独立した配置乱数 `PCG64(SeedSequence([51000,300,attempt]))` の候補3353を採用し、全候補の棄却理由・hashを保存しました。ユーザーによる新旧空間プレビューの確認・配置承認を受領しています。全精度座標と承認範囲は [配置manifest](docs/evidence/dispersed/layout.json) にあります。公称水平領域の9倍と、実凸包面積12.274250 m²や飛行距離の比を混同しません。

重要度seed40029とA81/B42/C43/D64/E23/F84/G6/H98/I22/J56を維持します。3シナリオ・全execution seedで同じ配置・初期値を使い、再抽選しません。目的コストは移動コスト＋0.05×Σ（整数重要度×名目完了時刻）。alpha=1、beta=0.05、名目速度1 m/s、サービス0.50秒、帰還なし、重要度観測の除数100。有向倍率1〜5は論理コストだけに適用します。緊急は未点検Cを43→100、コスト変更は新解析から宣言したF→Aのみ1→5です。

## 新しい解析と飛行検証

採用済み整数routing seed44を維持し、新3D分布で全3モデルをvalidation100・静的held-out100・動的50例に評価しました。450 learned episodes完走、無効行動0、全モデルが静的品質条件を満たしました。採用44の平均AI/厳密解比は1.078192、greedyとの差−24.744215、fixedとの差−243.810273です。最悪比1.397014、動的49適用中6変更・4改善も保存し、常時最適とは主張しません。

新配置の30判断・225完全順序を標準ライブラリのみの独立checkerで再計算し、成分最大差1.819×10^-12で一致しました。Q*/J*、AIのsuffix、実飛行を区別し、先行の実績を新配置へ流用しません。

旧飛行model_1300/cfgのhashは一致しましたが、旧範囲x/y±1 m・z=1 mは新配置を支持しません。追加学習の初回モデルはvalidationで地面条件に達して失敗し、モデル・sample・画像を保存しました。採用済みmodel_1300から4段階・3000更新の学習を行い、新model_4296を採用しました。validation11/12・held-out31/32/33の45遷移とマーカー有無比較が合格した後、座標・モデル対・カメラ・閾値を凍結しました。到達半径.20 m・速度<=.30 m/s・保持.20秒・点検.50秒・地面.10 m・傾斜60度を維持します。

新規固定9試行は9/9成功・90/90点検で、独立物理checker・14,673元フレームのdecode/PTS照合・凍結入力照合がPASS。通常32.64秒、緊急33.45秒、コスト変化30.74秒です。最大傾斜49.614度、最低高度0.835222 m。最終動画5本・11,361フレームの独立照合と、画質修正後の全編3本・両イベント区間の人手確認も完了しました。全7受入条件を満たしました。詳細は [引き継ぎ](docs/dispersed-destination-layout-handoff.md) にあります。

## 成果物と再実行

一意な出力先は `outputs/dispersed_20261020_v1/`（Git管理外）。全モデル・元動画・大量ログ・接続情報をcommitへ含めません。飛行用環境とCPU routing用環境は `requirements/flight-linux.lock.txt` と `requirements/routing-scalable-linux.lock.txt` で分離します。新しい作業先・設定・出力名を使用し、過去のサーバ絶対パスを流用しません。

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
  --out outputs/dispersed_rerun_new/geometry-check.json
.venv-routing311/bin/python -m pytest -m 'not gpu' -q
.venv-routing311/bin/python -m ruff check src scripts tests
```

予算は累計8時間を継続し、開始3198.994308秒から全GPUジョブを加算しました。今回増分2580.479709秒、累計5779.474017秒、残り23020.525983秒です。同時1ジョブ・残予算内timeoutを守り、未確認・失敗を合格にしません。新しい自然尺3全編・統合・140.9秒提出用を生成し、全飛行を1倍速で保持しました。公開・push・PR作成・マージ・コース提出は行いません。継承元とライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

新しい映像（全飛行区間を省略せず1倍速、1920×1080/30fps）：[通常全編](outputs/dispersed_20261020_v1/delivery_v2/full_normal.mp4)、[緊急全編](outputs/dispersed_20261020_v1/delivery_v2/full_emergency.mp4)、[コスト変化全編](outputs/dispersed_20261020_v1/delivery_v2/full_cost_shift.mp4)、[統合版](outputs/dispersed_20261020_v1/delivery_v2/demo_full.mp4)、[140.9秒提出用](outputs/dispersed_20261020_v1/delivery_v2/submission.mp4)。提出用は統合版と同じ内容でカットなし、飛行96.9秒＋独立説明カード44秒です。

最終日本語PDF：[測定・受入レポート](outputs/dispersed_20261020_v1/delivery_v2/report/final-report.pdf)。成果物サイズ・SHA-256は [コンパクトmanifest](docs/evidence/dispersed/artifacts.json)、受入結果は [受入記録](docs/evidence/dispersed/acceptance.json) を参照してください。
