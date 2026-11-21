"""CPU-only, loopback-only bridge to the original text environment; one client at a time."""
import hashlib
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from .common import ROOT, parse_action, is_executable, write_json, read_json, goal_text, normalize_instruction, instruction_hash


def serve(args):
    # Import only inside the environment venv, avoiding old WebShop dependencies in GPU venv.
    from webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    random.seed(42)
    env = WebAgentTextEnv(observation_mode='text', human_goals=True, num_products=None)
    goals = env.server.goals
    fingerprint = hashlib.sha256(json.dumps(goals, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    known = ROOT / 'data/raw/webshop_sft.json'
    alignment = {'checked': 0, 'mismatched': [], 'budget_variant_ids': [],
                 'note': 'Compare underlying instructions; random budget variants are recorded separately.'}
    if known.exists():
        for row in read_json(known):
            idx = int(row['id'])
            actual = ' '.join(goals[idx]['instruction_text'].lower().split()) if idx < len(goals) else None
            alignment['checked'] += 1
            if actual is None or normalize_instruction(actual) != normalize_instruction(goal_text(row)):
                alignment['mismatched'].append(idx)
            elif actual != goal_text(row):
                alignment['budget_variant_ids'].append(idx)
    write_json(ROOT / 'artifacts/environment-alignment.json', alignment)
    write_json(ROOT / 'artifacts/environment-manifest.json', {'goal_fingerprint': fingerprint, 'n_goals': len(goals), 'seed': 42})
    write_json(ROOT / 'artifacts/goal-instructions.json', [
        {'task_id': i, 'instruction': g['instruction_text']} for i, g in enumerate(goals)])
    def split_status():
        directory = ROOT / 'data/processed'
        required = [directory / f'{s}_ids.json' for s in ['train', 'val', 'test']]
        if not all(p.exists() for p in required):
            return {'verified': False, 'reason': 'Missing processed split ID files'}
        sets = [set(read_json(p)) for p in required]
        for ids in sets:
            if any(type(i) is not int or i < 0 or i >= len(goals) for i in ids):
                return {'verified': False, 'reason': 'Invalid task ID in split files'}
        hashes = [{instruction_hash(goals[i]['instruction_text']) for i in ids} for ids in sets]
        overlaps = {f'{a}-{b}': len(hashes[a] & hashes[b]) for a, b in [(0, 1), (0, 2), (1, 2)]}
        id_overlaps = [len(sets[a] & sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
        return {'verified': not any(overlaps.values()) and not any(id_overlaps), 'instruction_overlaps': overlaps}
    def available_actions():
        available = env.get_available_actions()
        if getattr(args, "option_state", False):
            session = env.server.user_sessions.get(env.session, {})
            available["selected_options"] = (
                dict(session.get("options", {})) if "buy now" in available["clickables"] else {}
            )
        return available

    class Handler(BaseHTTPRequestHandler):
        def send(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == '/health':
                self.send({'ready': True, 'n_goals': len(goals), 'goal_fingerprint': fingerprint,
                           'alignment_checked': alignment['checked'], 'alignment_mismatches': len(alignment['mismatched']),
                           'split_status': split_status()})
            else:
                self.send({'error': 'Not found'}, 404)
        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1_000_000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if self.path == '/reset':
                    task = int(body['task_id'])
                    if not 0 <= task < len(goals):
                        raise ValueError('Task ID outside environment goal range')
                    obs, _ = env.reset(task)
                    self.send({'observation': obs, 'instruction': env.instruction_text,
                               'available': available_actions(), 'goal_fingerprint': fingerprint})
                elif self.path == '/step':
                    action = parse_action('Action: ' + body['action'])
                    if not is_executable(action, env.get_available_actions()):
                        raise ValueError('Action is not available in current state')
                    obs, reward, done, _ = env.step(action)
                    self.send({'observation': obs, 'reward': float(reward), 'done': bool(done),
                               'available': available_actions()})
                else:
                    self.send({'error': 'Not found'}, 404)
            except (ValueError, KeyError, AssertionError, TypeError) as e:
                self.send({'error': str(e)}, 400)
    print(f'WebShop ready on http://127.0.0.1:{args.port}; goals={len(goals)}; fingerprint={fingerprint}')
    print(f'SFT alignment mismatches={len(alignment["mismatched"])}; inspect artifacts/environment-alignment.json')
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
