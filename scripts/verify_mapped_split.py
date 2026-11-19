"""Verify mapped-20261119-zqq: arguments are source SFT and raw goal-map paths."""
import json,hashlib,re,subprocess,sys
from pathlib import Path
source=Path(sys.argv[1]);mapfile=Path(sys.argv[2]);directory=Path('data/processed/mapped-20261119-zqq')
rows=json.loads(source.read_text());goals=json.loads(mapfile.read_text());mapping={r['task_id']:r['instruction'] for r in goals}
def norm(s):return re.sub(r',?\s*and price lower than [\d.]+ dollars\.?$','',' '.join(s.lower().split()).rstrip('.')).strip().rstrip('.')
def instruction(row):
 for t in row['conversations']:
  if t['from']=='human' and 'Instruction:' in t['value']:
   return t['value'].split('Instruction:',1)[1].removeprefix(' [SEP]').strip().split('[SEP]',1)[0].strip()
sets=[set(json.loads((directory/(n+'_ids.json')).read_text())) for n in ['train','val','test']]
checks={'map_sha_matches':hashlib.sha256(mapfile.read_bytes()).hexdigest()=='6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88','map_unique_contiguous_ids':len(goals)==len(mapping)==12087 and set(mapping)==set(range(12087)),'all_sft_instructions_align':all(norm(instruction(r))==norm(mapping[r['id']]) for r in rows),'test_coverage':sets[2]<=mapping.keys(),'all_split_ids_disjoint':all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])}
hashes=[{norm(mapping[i]) for i in ids} for ids in sets]
checks['all_actual_environment_instructions_disjoint']=all(not hashes[a]&hashes[b] for a,b in [(0,1),(0,2),(1,2)])
env='d1d160bcfa44c7caa2c92609d8e5012662a23de6'
for name in ['train_ids.json','val_ids.json','test_ids.json','split-manifest.json']:
 checks[name+'_matches_environment_blob']=(directory/name).read_bytes()==subprocess.check_output(['git','show',env+':results/env-2026-11-17/'+name])
original={r['id']:r for r in rows}
for name in ['train','val']:
 kept=json.loads((directory/(name+'.json')).read_text());checks[name+'_records_preserved']=all(r==original[r['id']] for r in kept)
if not all(checks.values()):raise ValueError(checks)
out=Path('results/mapped-split-20261119-zqq');out.mkdir(parents=True,exist_ok=True)
result={'date':'2026-11-19','executor':'Codex Tutor; not independent student execution','status':'static_mapping_split_verified_live_environment_not_rerun','goal_map_sha256':hashlib.sha256(mapfile.read_bytes()).hexdigest(),'map_bytes':mapfile.stat().st_size,'map_tasks':len(goals),'counts':{'train':len(sets[0]),'val':len(sets[1]),'test':len(sets[2]),'excluded':len(rows)-len(sets[0])-len(sets[1])},'checks':checks,'environment_evidence_commit':env,'reported_goal_fingerprint':'e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8','not_verified':['No local live WebShop/GPU service started. Fingerprint is from HowseaonLink evidence, not recomputed from the full environment goals.','Training must still check live health against the adopted IDs and goal fingerprint.']}
(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(out/'split-manifest.json').write_bytes((directory/'split-manifest.json').read_bytes())
print(json.dumps({'checks':len(checks),'passed':all(checks.values()),'counts':result['counts']}))
