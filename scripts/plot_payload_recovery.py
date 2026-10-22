#!/usr/bin/env python
"""Plot actual passive energy recordings; plotting never grants admission."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from adaptive_drone_patrol.payload_passive_check import mechanical_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', required=True)
    parser.add_argument('--font', default='outputs/assets/NotoSansJP.ttf')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.delivery), Path(args.out)
    if out.exists() or out.with_suffix('.json').exists():
        raise FileExistsError('Preserve existing figures and receipts')
    font = FontProperties(fname=args.font, size=8)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
    inputs = {}
    cases = [('recovery-baseline', '旧V10・物理dt 0.005秒', 0),
             ('recovery-s128-full-large', '90度・128 substeps', 1),
             ('recovery-s128-full-170', '170度・128 substeps', 1),
             ('recovery-s128-half-large', '90度・256 substeps', 1)]
    for name, label, axis in cases:
        path = root/'outputs'/name
        summary = json.loads((path/'summary.json').read_text())
        rows = [json.loads(line) for line in (path/'samples.jsonl').open()]
        side, runtime = summary['case']['side_m'], summary['runtime']
        energy = np.array([mechanical_energy(row, runtime, side, .98) for row in rows])
        axes[axis].plot([row['time_s'] for row in rows], 100*(energy/energy[0]-1),
                        label=label, linewidth=.9)
        inputs[name] = {file:hashlib.sha256((path/file).read_bytes()).hexdigest()
                        for file in ['samples.jsonl', 'runtime-initial.json']}
    for axis, title in zip(axes, ['旧失敗の新規再現', '新候補の受動診断のみ']):
        axis.set_title(title, fontproperties=font)
        axis.set_xlabel('シミュレーション時間 [秒]', fontproperties=font)
        axis.set_ylabel('初期値からのenergy変化 [%]', fontproperties=font)
        axis.grid(alpha=.25)
        axis.legend(prop=font, loc='best', fontsize=7)
        axis.tick_params(labelsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    receipt = dict(schema='payload-recovery-energy-figure-v1', inputs=inputs,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   figure_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                   interpretation_ja='受動診断だけを描画。右の正energyが小さくても損失・dt軌道差・未実行条件が残り、物理資格ではない。')
    out.with_suffix('.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
