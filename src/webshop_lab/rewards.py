"""Reward interfaces for experiments; exporting records is not PPO training."""
from .common import combine_rewards, write_json
from .evaluation import load_episodes, terminal_reward


def run(args):
    rows = load_episodes(args.input)
    records = []
    for row in rows:
        steps = row['steps']
        if args.mode == 'spa' and any('progress' not in s for s in steps):
            raise ValueError('SPA export requires predicted progress for EVERY step. Never substitute zeros or final reward.')
        progress = [s.get('progress', 0.) for s in steps]
        task_reward = terminal_reward(row)
        rewards = combine_rewards(progress, [s['grounding'] for s in steps], task_reward,
                                  args.mode, args.alpha, args.beta)
        records.append({'task_id': row['task_id'], 'repeat': row.get('repeat', 0), 'mode': args.mode,
                        'instruction': row['instruction'], 'observations': [s['observation'] for s in steps],
                        'responses': [s['response'] for s in steps], 'rewards': rewards,
                        'progress_sum': sum(progress), 'final_reward': row['final_reward'],
                        'terminal_reward': task_reward,
                        'goal_fingerprint': row.get('goal_fingerprint')})
    write_json(args.output, records)
    print(f'Exported {len(records)} trajectories. This command does not update a policy.')
