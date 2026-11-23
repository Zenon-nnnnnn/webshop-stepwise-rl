"""独立进程恢复Qwen/B1初始化、PRM LoRA和head，不读取标签预测。"""
import argparse
import json
import math
from pathlib import Path
from .config import load_config
from .runtime import build_prm, write_json, verify_runtime_inputs
from .train import load_samples, predict


def compare_predictions(actual,expected,atol,rtol):
    if len(actual) != len(expected):
        raise ValueError('重载样本数不一致')
    for a,b in zip(actual,expected):
        if a['task_id'] != b['task_id'] or a['turn_end_indices'] != b['turn_end_indices']:
            raise ValueError('重载任务或动作位置不一致')
        if len(a['step_scores']) != len(b['step_scores']):
            raise ValueError('重载动作分数数量不一致')
        for x,y in zip(a['step_scores']+[a['total']],b['step_scores']+[b['total']]):
            if not math.isfinite(x) or not math.isfinite(y) or not math.isclose(x,y,abs_tol=atol,rel_tol=rtol):
                raise ValueError('独立进程逐步或总分超出预定容差')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','samples','output'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--atol',type=float,default=1e-5)
    p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=p.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.atol,args.rtol)):
        raise ValueError('容差必须为非负有限数')
    if args.output.exists() or args.checkpoint.resolve() in args.output.resolve().parents:
        raise ValueError('新输出须位于检查点之外，不能覆盖旧证据')
    cfg=load_config(args.config)
    model,tok,identity,inputs=build_prm(cfg,args.checkpoint,args.device,False)
    rows,rejected,total=load_samples(args.samples,tok,cfg.max_length,training=False)
    if not rows:
        raise ValueError('无可评分的完整轨迹，不能用空比较宣布重载通过')
    actual=predict(model,tok,rows)
    if args.reference:
        compare_predictions(actual,json.loads(args.reference.read_text(encoding='utf-8')),args.atol,args.rtol)
    if verify_runtime_inputs(cfg) != inputs:
        raise ValueError('推理过程中初始化文件改变')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,{'predictions':actual,'identity':identity,'device':args.device,
                          'total_samples':total,'rejected':rejected,
                          'reference_match':True if args.reference else None,
                          'atol':args.atol,'rtol':args.rtol,'test_authorization':False})


if __name__ == '__main__':
    main()
