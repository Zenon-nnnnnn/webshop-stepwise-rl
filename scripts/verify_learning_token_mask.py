"""CPU tokenizer-only check: pass the source SFT path as the first argument."""
import json,hashlib,sys
from pathlib import Path
from transformers import AutoTokenizer
from webshop_lab.model import training_messages,fit_history
revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',revision=revision,trust_remote_code=False)
rows=json.load(open(sys.argv[1]))
row=rows[0];messages=training_messages(row);examples=[]
for i,m in enumerate(messages):
 if m['role']!='assistant':continue
 kept,ids=fit_history(tok,messages[:i+1],2048)
 prefix=tok.apply_chat_template(kept[:-1],tokenize=True,add_generation_prompt=True)
 if ids[:len(prefix)]!=prefix or len(prefix)>=len(ids):raise ValueError('prefix mismatch')
 labels=[-100]*len(prefix)+ids[len(prefix):]
 decoded=tok.decode(ids[len(prefix):],skip_special_tokens=False)
 if not decoded.startswith(m['content']): raise ValueError('target differs from assistant text')
 examples.append({'assistant_turn':len(examples)+1,'sequence_tokens':len(ids),'masked_prefix_tokens':len(prefix),'supervised_tokens':len(ids)-len(prefix),'supervised_token_ids':ids[len(prefix):],'prefix_labels_all_ignore':all(x==-100 for x in labels[:len(prefix)]),'target_labels_equal_ids':labels[len(prefix):]==ids[len(prefix):],'target_contains_thought':decoded.startswith('Thought:'),'target_contains_action':'Action:' in decoded,'target_tail':repr(decoded[-40:])})
# Check the collator's exact padding contract without importing torch.
size=max(x['sequence_tokens'] for x in examples)
for x in examples:
 padding=size-x['sequence_tokens'];x['padding_tokens_in_combined_batch']=padding
 x['padding_labels_ignore']=all(v==-100 for v in [-100]*padding)
out=Path('results/learning-20261119-zqq');out.mkdir(parents=True,exist_ok=True)
result={'kind':'real-tokenizer-preprocessing-check-not-training','model':'Qwen/Qwen2.5-1.5B-Instruct','revision':revision,'task_id':row['id'],'examples':examples,'note':'No model weights loaded; no torch, Trainer forward/backward or RL mask validated. Padding check reproduces source construction, not Trainer execution.'}
(out/'token-mask-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'example_count':len(examples),'counts':[{k:v for k,v in x.items() if k!='supervised_token_ids'} for x in examples]},ensure_ascii=False))
