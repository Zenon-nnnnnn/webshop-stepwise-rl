"""Cross-check fixed PR #10/#11 Git blobs; run from the repository root.

Uses only the Python standard library. This validates committed evidence,
not the absent raw goal map, a live environment or a trained policy.
"""
import hashlib,json,subprocess
from pathlib import Path
ENV='d1d160bcfa44c7caa2c92609d8e5012662a23de6'
OFF='12ce2b11dffb5e0956b8a0d6750c9e05ad62cb41'
PREFIX='results/env-2026-11-17/'
def blob(ref,path):return subprocess.check_output(['git','show',f'{ref}:{path}'])
def obj(ref,path):return json.loads(blob(ref,path))
files={}
checks={}
for name in ['train_ids.json','val_ids.json','test_ids.json']:
 a,b=blob(ENV,PREFIX+name),blob(OFF,'results/offline-split-20261117-zqq/'+name)
 checks[name+'_byte_identical']=a==b
 files[name]={'sha256':hashlib.sha256(a).hexdigest(),'count':len(json.loads(a))}
sets=[set(obj(ENV,PREFIX+x)) for x in ['train_ids.json','val_ids.json','test_ids.json']]
checks['id_sets_disjoint']=all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
a=obj(ENV,PREFIX+'split-manifest.json');b=obj(OFF,'results/offline-split-20261117-zqq/split-manifest.json')
checks['manifest_matches_except_goal_map']=all(a[k]==b[k] for k in set(a)|set(b) if k!='goal_map_sha256')
checks['goal_map_sha_present']=isinstance(a['goal_map_sha256'],str) and len(a['goal_map_sha256'])==64
health=obj(ENV,PREFIX+'health-frozen.json');alignment=obj(ENV,PREFIX+'environment-alignment.json');manifest=obj(ENV,PREFIX+'environment-manifest.json');meta=obj(ENV,PREFIX+'rollout-3val.meta.json')
checks['reported_health_ready']=health['ready'] is True
checks['reported_alignment_complete']=health['alignment_checked']==alignment['checked']==1624 and health['alignment_mismatches']==len(alignment['mismatched'])==0
checks['reported_health_split_verified']=health['split_status']['verified'] is True and all(x==0 for x in health['split_status']['instruction_overlaps'].values())
checks['fingerprint_consistent']=health['goal_fingerprint']==manifest['goal_fingerprint']==meta['health']['goal_fingerprint']==meta['config']['goal_fingerprint']
checks['counts_consistent']=len(sets[0])==a['train_rows']==1451 and len(sets[1])==a['val_rows']==160 and len(sets[2])==a['test_tasks']==200
trace=blob(ENV,PREFIX+'rollout-3val.jsonl');rows=[json.loads(x) for x in trace.splitlines()]
supp=obj(ENV,PREFIX+'rollout-3val-terminal-supplement.json')
terminal=[float(r['final_reward']) if r['done'] else 0. for r in rows]
checks['rollout_tasks_inside_validation']=all(r['task_id'] in sets[1] for r in rows)
checks['nonpurchase_terminal_rewards_zero']=all(not r['done'] for r in rows) and terminal==[0.,0.,0.]
checks['supplement_source_hash_matches']=supp['source_jsonl_sha256']==hashlib.sha256(trace).hexdigest() and supp['source_jsonl_bytes']==len(trace)
checks['supplement_terminal_mean_matches']=supp['aggregates']['mean_terminal_reward']==sum(terminal)/len(rows)
if not all(checks.values()):
    raise ValueError({name: ok for name, ok in checks.items() if not ok})
result={'date':'2026-11-18','executor':'Codex tutor demonstration; not student-independent execution','environment_evidence_commit':ENV,'offline_evidence_commit':OFF,'checks':checks,'id_files':files,'reported_goal_map_sha256':a['goal_map_sha256'],'goal_fingerprint':health['goal_fingerprint'],'rollout_diagnostic':{'source_sha256':hashlib.sha256(trace).hexdigest(),'episodes':len(rows),'purchase_terminated':sum(bool(r['done']) for r in rows),'terminal_rewards':terminal,'mean_terminal_reward':sum(terminal)/len(rows),'last_environment_scores':[r['final_reward'] for r in rows],'mean_last_environment_score':sum(r['final_reward'] for r in rows)/len(rows),'episode_seconds':sum(r['seconds'] for r in rows)},'status':'handoff_artifacts_cross_checked_raw_goal_map_pending','not_verified':['Full goal-instructions.json absent: its file hash, instruction coverage and split --goal-map not independently rerun on Mac.','Health JSON checked as committed snapshot only; no live WSL service/GPU tested.','No final data freeze, SFT training or RL update performed.']}
out=Path('results/data-handoff-20261118-zqq');out.mkdir(parents=True,exist_ok=True)
(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'passed':len(checks),'status':result['status'],'diagnostic':result['rollout_diagnostic']},ensure_ascii=False,indent=2))
