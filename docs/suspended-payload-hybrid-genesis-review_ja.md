# ハイブリッド制御・Genesis優先specの精査

2026-10-23。対象は[提案spec](../specs/2026-10-23-suspended-payload-hybrid-genesis/spec.md)と[保存された日本語原文](../specs/2026-10-23-suspended-payload-hybrid-genesis/spec_ja.md)。旧作業の結果を踏まえ、主担当と独立したCPUレビュー2件で確認した。新specの実装・新規実験は開始していない。

## 判断

方向転換は妥当であり、旧方針の追加学習を終了してデータ・実装・未達を保存することを妨げる技術的反論はない。純粋なNN直接motor制御の失敗は、この物理モデルでの輸送不能やハイブリッド制御の成功を証明しない。

旧方針では完全数値資格を通過した一方、pilot5回1.02Mと両方式各3M、計7.02M有効遷移で共有settling0だった。30万・100万・300万時点の比較は両方式とも0/6、全36飛行で最初のhover30秒timeout、全native再検査は通過した。[実比較](evidence/suspended-solver/paired-development-curve.json)と[学習記録](evidence/suspended-solver/training-progress.json)が判断の根拠である。

位置・並進速度を満たしても、荷物の横方向回転が約4.2Hzで残る実記録がある。[診断](evidence/suspended-solver/long-credit-feedback-diagnosis.json)では1episode末尾100秒のspin平均95.833度/秒、直接spin costのcap分率99.36%だった。これは制振を独立課題として先に実測する根拠であり、報酬のcap除去だけで成功するという根拠ではない。後半107失敗episodeと遠方hoverの構成benchmark比較も、実方策の最適性や落下の意図を証明しない。

両端の受動ball jointは荷物への直接motorトルクを持たないが、重心から離れた取付点への拘束力まで遮断するものではない。既存の固定支点・小角度のmodal診断では高速modeへの水平支点加速度の係数が非ゼロだった。ただし元motorを含む実際の制御権限・位相・減衰時間の十分性は未検証であり、spec §4.3の実powered診断で確かめる。全姿勢・全modeの制御可能性は主張しない。

## 妥当な点

- 制御器単独で静定・実移動を確認してから、上限付きNN補正や蒸留と比較する。教師の成功をNNの成功に読み替えない。
- 質量・寸法等の真値をbaselineや推定器にも迂回入力しない。観測契約とpolicy形式を新versionにし、旧29次元checkpointを形状だけで読み込まない。
- 固定遷移数だけで難課題へ進めず、複数episodeでの技能と安全性を判定し、既習課題の忘却も検査する。
- Genesis GPU並列を新しく資格検証する。旧MuJoCo CPU N1のPASSを新backend・新batchへ流用せず、物理・motor・閾値・受入分母を維持する。
- nominalのみの共通teacherと共通初期値を使い、両最終方式の有効遷移数を揃える。制御器単独の改善、NNの寄与、GPU高速化を別に報告する。

## 次の最終計画までに明確化する2点

1. **controller-onlyの最終採用可否。** §3/§7.2は最も単純な安定構成の選択を認める一方、§7.3/§10は公平な両最終学習を必須としている。制御器単独が全要件を満たした場合もNN比較を実施するのか、最終採用を認めるならrandomized/fixed学習比較やS・optimizerをどう扱うのかを一意にする。全受入・動画を免除する理由にはならない。
2. **N1 fallbackと実並列必須の関係。** §5.4は根拠付きCPU N1主学習を認めるが、§10の完了条件(b)は実並列実行を要求する。全並列候補が未資格の場合も完了不可なのか、理由書による例外を認めるのかを明確にする。現段階でGenesisが不可能・遅いと判断したわけではない。

いずれも方針全体が不可能という反論ではなく、将来の最終採用・終了判定の未確定分岐である。今回のレビューではユーザーの提案spec本文を変更していない。

## 実装時の検証事項

Genesis優先は「物理資格を満たすGenesis候補の中で実時間を短縮する」という制約付き目標として一貫している。「全backend中で最速」を意味しない。[公式Genesis解説](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_solver/forward_dynamics.html)の積分法は候補の背景であり、採用する実版のsource・設定を優先する。fallback先の[MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)もGPU多数world向けで、単一stepの低遅延や本モデルの適格性を保証していない。

制御・推定のnominal設計定数、測定量、オンライン推定値、checker専用の真値を分離する。実寸法をattachment計算へ渡す経路、既存baselineや重みの荷物経験、教師の情報権限も監査する。報酬は実行軌道で成功・無進展・早期失敗を比較し、10番目の成功終了を回避する誘因も検査する。masteryは実目標の順序・timeout・stage境界を含め、単に完了countが10であることを全10課題の成功としない。

GPU対応precision、受動球関節、環境ごとの物性、force/work観測、部分reset、完全raw保存と実throughputはすべて今後の実測事項である。今回の24時間超というユーザーの状況説明に対して、累積GPU時間台帳を作成・推定していない。
