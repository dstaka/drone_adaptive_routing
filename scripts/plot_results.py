#!/usr/bin/env python
"""Generate Japanese figures from retained benchmark and real mission telemetry."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from adaptive_drone_patrol.util import unique_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', default='outputs/evaluation')
    p.add_argument('--genesis', default='outputs/genesis')
    p.add_argument('--font', default='outputs/assets/NotoSansJP.ttf')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = unique_dir(a.out)
    font = FontProperties(fname=a.font)
    def finish(fig, axis, title, ylabel, name):
        axis.set_title(title, fontproperties=font, fontsize=14)
        axis.set_ylabel(ylabel, fontproperties=font)
        axis.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, split in zip(axes, ['static', 'dynamic']):
        rows = list(csv.DictReader((Path(a.evaluation) / split / 'results.csv').open()))
        planners = ['fixed', 'greedy', 'learned'] + (['exact'] if split == 'static' else [])
        means = [np.mean([float(r['analytical_cost']) for r in rows if r['planner'] == n]) for n in planners]
        axis.bar(planners, means, color=['#a8b9c4', '#8abeb8', '#156e8c', '#e6b568'][:len(planners)])
        axis.set_title(('静的100例' if split == 'static' else '動的50例') + ' × 学習3シード', fontproperties=font)
        axis.set_ylabel('目的コスト（モデル単位）', fontproperties=font)
        axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'cost_comparison.png', dpi=160)
    plt.close(fig)
    all_attempts = json.loads((Path(a.genesis) / 'all_attempts.json').read_text())
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, name in enumerate(['normal', 'emergency', 'cost_shift']):
        rows = [r for r in all_attempts if r['scenario_id'] == name]
        for trial, r in enumerate(rows):
            ax.scatter(range(len(r['completed_order'])), [index + trial * .08] * len(r['completed_order']), s=80)
            for i, target in enumerate(r['completed_order']):
                ax.text(i, index + trial * .08, target, ha='center', va='center', color='white')
    ax.set_yticks([0, 1, 2], ['normal', 'emergency', 'cost_shift'])
    ax.set_xticks([0, 1, 2], ['1', '2', '3'])
    ax.set_xlabel('実際の訪問順序（各シナリオ3試行）', fontproperties=font)
    finish(fig, ax, '実物理で確認した巡回順序', 'シナリオ', 'scenario_order_comparison.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, changed = [], []
    for run in sorted(Path(a.genesis).glob('*/events.jsonl')):
        events = [json.loads(line) for line in run.read_text().splitlines()]
        for e in events:
            if e['type'] in ['PRIORITY_UPDATED', 'COST_UPDATED']:
                labels.append(run.parent.name)
                changed.append(int(e['actual_action'] != e['counterfactual_action']))
    ax.bar(range(len(changed)), changed, color='#156e8c')
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, 1.2)
    finish(fig, ax, '到着境界の反実仮想比較（プランナーのみ）', '次目標の変更 0/1', 'event_replanning.png')
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in ['normal', 'emergency', 'cost_shift']:
        run = Path(a.genesis) / f'{name}_seed_1'
        rows = list(csv.DictReader((run / 'telemetry.csv').open()))
        ax.plot([float(r['x']) for r in rows], [float(r['y']) for r in rows], label=name)
    scenario = json.loads((Path(a.genesis) / 'normal_seed_1' / 'scenario.json').read_text())
    for name, point in zip('SABC', scenario['coords']):
        ax.scatter(point[0], point[1], color='#e6b568')
        ax.annotate(name, point[:2], xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('実測 x（m）', fontproperties=font)
    ax.axis('equal')
    ax.legend()
    finish(fig, ax, '実Genesisテレメトリの軌跡（対応シード1）', '実測 y（m）', 'real_trajectory.png')


if __name__ == '__main__':
    main()
