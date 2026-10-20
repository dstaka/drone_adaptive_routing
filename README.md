# 適応型ドローン巡回点検

A〜Jの10地点を、同じ学習済みルーティング方策と飛行方策で巡回します。重要度は実値・表示とも1〜100の整数で、大きいほど重要ですが訪問順そのものではありません。各地点を独立に復元抽選し、固定seedで再現します。配置は先行デモと同じ長方形のままです。

[今回の承認済み仕様](specs/2026-10-17-integer-priorities/spec.md)、[実装計画](docs/superpowers/plans/2026-10-20-integer-priorities.md)、[モデル・測定・全再実行手順](docs/integer-priorities-handoff.md)を参照してください。過去の[経路最適化](docs/scalable-routing-handoff.md)、[全地点飛行](docs/visible-flight-handoff.md)、[旧1倍速動画](docs/realtime-video-handoff.md)の記録・成果物は保持しています。

## 今回の実測

新しい重要度・対応routingモデルで固定3seed×3シナリオを走らせ、9/9ミッション・90/90点検を完了しました。通常17.61秒、緊急18.07秒、コスト変化18.76秒です。reset・終了・安全異常は0。最大傾斜37.686度、最低高度0.805877m。全90判断から物理説明を再生成し、重要度差10以上・実pose距離・厳密コストの条件を満たすL/H判断が33件あります。

初期重要度は A=81、B=42、C=43、D=64、E=23、F=84、G=6、H=98、I=22、J=56。3シナリオ・実行seed1/2/3で共通です。最初の実点検後（3.43秒）、緊急では未点検Cのみ43→100、有向コスト変化ではH→Fのみ倍率1→5を更新します。次の選択は更新なしのFに対し、CとJへそれぞれ変わります。同一の更新後条件で旧/新suffixを比較した改善は0.631889と14.734742です。これは論理目的コストの比較で、物理時間・電力の削減とは区別します。

新routingはscenario v3、`routing-v3-integer100-n10-324`、324要素・10行動です。3seed42/43/44を新規学習し、validationでseed44を選択・凍結してからheld-outを評価しました。全450 learned episodesが完走し、全3seedで既存品質条件を満たしました。採用seedの静的平均AI/厳密解比は1.040070、最悪1.156867。動的50例のうち49適用、7選択変更、6改善です。一般に常時最適とは主張しません。

## 目的関数と互換性

目的コスト＝移動コスト＋0.05×Σ（整数重要度×名目完了時刻）。alpha=1、beta=0.05、名目速度1m/s、サービス0.50秒、帰還なし。観測の重要度は固定値100で割り、現在の最大値では正規化しません。有向倍率は1〜5で、名目時間や物理速度を変更しません。生の重要度時刻成分と0.05適用後の寄与を別々に記録・表示します。

7.0を含む浮動小数、真偽値、文字列、非有限値、範囲外をnew-profileの重要度として受け付けません。v2・旧checkpointは元の解釈を維持し、形状が同じでもv3への読替えは拒否します。新checkpoint内部・sidecar・教師データにも契約を記録します。詳細は [互換性記録](docs/compatibility.md) にあります。

飛行には指定の追加学習後 `model_1300.pt` を継続使用します。新しいGPUサーバで長距離・反転・ホバーを再確認し、マーカー有無の1,508サンプルは一致しました。実到達は半径0.20m以内・速度0.30m/s以下を連続0.20秒、点検は同条件で合計0.50秒です。固定初期条件でのデモ安定性の結果であり、擾乱・障害物認識・実機安全性は未検証です。

## 成果物と再生成

一意な成果物ルートは `outputs/integer_20261020_v1/`（Git管理外）です。旧動画を今回の受入に流用していません。確定した機械検証・実フレーム/PDF確認・連続再生確認の状態は [引き継ぎ](docs/integer-priorities-handoff.md) と `docs/evidence/integer/acceptance.json` に分けて記録します。未確認の項目を合格とは扱いません。

- `training/`、`frozen/`：3seedの新モデル、教師データ、validation、凍結記録。
- `static/`、`dynamic/`、`demo/`：全新入力、評価、30候補の抽選と却下理由、厳密説明。
- `remote-acceptance/outputs/integer/acceptance_v1/genesis/`：9本の元動画、全物理・判断・イベントログ。
- `physical-verification/`：90判断の再生成説明と独立物理受入。
- `delivery/`：3本の自然尺全編、全3飛行を含む `demo_full.mp4` と `submission.mp4`、1倍速raw、対応表、manifest、最新PDF。

```bash
export PYTHONPATH=src
export PATH="$PWD/outputs/integer_20261020_v1/tools:$PATH"
export MPLCONFIGDIR=outputs/integer_rerun_new/mpl
.venv-routing311/bin/python -m pytest -m 'not gpu' -q
.venv-routing311/bin/python -m ruff check src scripts tests

# Use a fresh output directory each time.
.venv-routing311/bin/python scripts/build_video.py --config configs/video_integer.yaml \
  --input-root outputs/integer_20261020_v1/remote-acceptance/outputs/integer/acceptance_v1/genesis \
  --explanations outputs/integer_20261020_v1/physical-verification/decision_explanations.json \
  --mode full --out outputs/integer_video_new/demo_full.mp4
.venv-routing311/bin/python scripts/build_video.py --config configs/video_integer.yaml \
  --input-root outputs/integer_20261020_v1/remote-acceptance/outputs/integer/acceptance_v1/genesis \
  --explanations outputs/integer_20261020_v1/physical-verification/decision_explanations.json \
  --mode submission --duration 120 --out outputs/integer_video_new/submission.mp4
.venv-routing311/bin/python scripts/check_realtime_media.py --root outputs/integer_video_new \
  --ffprobe outputs/integer_20261020_v1/tools/ffprobe
.venv-routing311/bin/python scripts/inspect_realtime_frames.py --root outputs/integer_video_new
.venv-routing311/bin/python scripts/build_report.py --source docs/report_ja.md --date 2026-10-20 \
  --out outputs/integer_video_new/report/report.pdf
```

学習、manifest、モデル互換性、GPU実行設定の生成、物理再評価、動画の最終化は [引き継ぎの手順](docs/integer-priorities-handoff.md) を使用してください。新サーバでは新しい作業先と別venvを使い、`requirements/flight-linux.lock.txt` と `requirements/routing-scalable-linux.lock.txt` を分離します。

## 保存・予算・出典

今回GPU増分395.4215675460873秒、累計3,198.9943080281373秒、8時間枠の残り25,601.005691971863秒です。初期化・推論・撮影・契約テストを含む全3ジョブを台帳へ追記しました。モデル・設定・旧素材465ファイル、新規転送132ファイルをサイズ・SHA-256で照合しています。

Gitには今回のspec・コード・テスト・必要な文書・コンパクトな証拠のみを含めます。歴史資料、他spec、個人ファイル、動画・モデル・大量ログ・接続先情報は混ぜません。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に継承元とライセンスを記載しています。push、PR作成/マージ、動画公開、コース提出、サーバの契約変更・追加レンタル・破棄は行いません。
