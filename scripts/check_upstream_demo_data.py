"""Check what the upstream demo trajectories can and cannot support.

The upstream snapshot ships 78 "exploration" trajectories. They are not our model's rollouts, so
this script measures what a reviewer could actually read from them before someone quotes them as
evidence:

- 31 turns each, but a large share of assistant turns are truncated/unparsable, so a
  trajectory-level label mixes real failure with broken generation;
- every trajectory contains a `click[buy now]`, yet most carry `Price:` text before it, so the
  price/specification text path can be calibrated on real WebShop observations;
- there is no per-step reward and no availability list, and `agent_final_reward` is 0/1 only.

Conclusion baked into the output: these files can calibrate text parsing, never our failure rates
or failure cases. Real cases must come from `webshop-lab rollout` logs.

Usage: python scripts/check_upstream_demo_data.py --output results/upstream-demo-data-check.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from webshop_lab.common import parse_action  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'upstream/exploration/webshop/exploration_outputs/explore/webshop_traj_0.json'
PRICE = re.compile(r'Price:\s*\$', re.I)
INVALID_FEEDBACK = re.compile(r'Invalid format', re.I)
INSTRUCTION = re.compile(r'Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]', re.S)


def inspect(trajectories):
    stats = collections.Counter()
    rewards = collections.Counter()
    instructions = set()
    turns_per_trajectory = []
    for trajectory in trajectories:
        turns = trajectory['agent_conversations']
        turns_per_trajectory.append(len(turns))
        rewards[str(float(trajectory['agent_final_reward']))] += 1
        for position, turn in enumerate(turns):
            if turn['role'] != 'assistant':
                continue
            stats['assistant_turns'] += 1
            try:
                action = parse_action(turn['content'])
            except ValueError:
                stats['unparsable_assistant_turns'] += 1
                continue
            stats['parseable_assistant_turns'] += 1
            observation = turns[position - 1]['content'] if position else ''
            if INVALID_FEEDBACK.search(observation):
                stats['parseable_action_after_invalid_feedback'] += 1
            if action.startswith('click[buy now]'):
                stats['buy_actions'] += 1
                if PRICE.search(observation):
                    stats['buy_actions_with_price_in_observation'] += 1
        for turn in turns:
            if turn['role'] == 'user' and INSTRUCTION.search(turn['content']):
                instructions.add(' '.join(INSTRUCTION.search(turn['content']).group(1).split()))
    return {
        'trajectories': len(trajectories),
        'turns_schema': f'role/content, {min(turns_per_trajectory)}-{max(turns_per_trajectory)} turns per trajectory',
        'assistant_turns': stats['assistant_turns'],
        'parseable_assistant_turns': stats['parseable_assistant_turns'],
        'unparsable_assistant_turns': stats['unparsable_assistant_turns'],
        'parseable_ratio': round(stats['parseable_assistant_turns'] / max(1, stats['assistant_turns']), 4),
        'buy_actions': stats['buy_actions'],
        'buy_actions_with_price_in_observation': stats['buy_actions_with_price_in_observation'],
        'parseable_action_after_invalid_feedback': stats['parseable_action_after_invalid_feedback'],
        'rewards': dict(rewards),
        'distinct_instructions': len(instructions),
        'usable_for': ['calibrating that the triage reads real WebShop observation text '
                       '(prices, buttons, "Invalid format" feedback)'],
        'not_usable_for': ['our failure-case section', 'any success-rate or failure-rate number'],
        'reasons': [
            'No per-step environment reward and no availability list in the source, so grounding can only measure parseability.',
            f'{stats["unparsable_assistant_turns"]} of {stats["assistant_turns"]} assistant turns are unparsable (truncated generation), '
            'so a trajectory-level label mixes real failure with broken generation.',
            'agent_final_reward is only 0 or 1: '
            f"{rewards.get('0.0', 0)} of {len(trajectories)} trajectories are 0 while every trajectory still contains a purchase action.",
            'These are upstream demo trajectories, not this group\'s model output.',
        ],
        'consequence': 'Use them to calibrate parsing only. Real failure cases (format/retrieval/specification/'
                       'budget/loop) must come from webshop-lab rollout logs on our own tasks.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--output', default='results/upstream-demo-data-check.json')
    args = parser.parse_args()
    report = inspect(json.loads(Path(args.source).read_text(encoding='utf-8')))
    report['source'] = str(Path(args.source).relative_to(ROOT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
