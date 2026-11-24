import importlib.metadata
import platform
import shutil
import subprocess
import sys
from .common import ROOT, write_json


def run(args):
    report = {"python": sys.version, "platform": platform.platform(), "executable": sys.executable,
              "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 2), "packages": {}, "assets": {}}
    for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "gym", "pyserini", "spacy"]:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    for name in ['data/raw/webshop_sft.json', 'data/processed/train.json',
                 'upstream/envs/webshop/data/items_shuffle.json',
                 'upstream/envs/webshop/data/items_ins_v2.json',
                 'upstream/envs/webshop/data/items_human_ins.json',
                 'upstream/envs/webshop/search_index/indexes/segments_1']:
        report['assets'][name] = (ROOT / name).exists()
    try:
        import torch
        report['torch_cuda_build'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report['gpu'] = torch.cuda.get_device_name(0)
            report['compute_capability'] = list(torch.cuda.get_device_capability(0))
            report['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            # Detect a wheel that imports successfully but cannot launch kernels on this GPU.
            a = torch.randn(32, 32, device='cuda')
            (a @ a).sum().item()
            torch.cuda.synchronize()
            report['cuda_kernel_smoke'] = 'passed'
    except Exception as e:
        report['torch_error'] = f'{type(e).__name__}: {e}'
    if shutil.which('java'):
        r = subprocess.run(['java', '-version'], capture_output=True, text=True)
        report['java'] = (r.stderr or r.stdout).strip()
    else:
        report['java'] = None
    write_json(args.output, report)
    print(f'Diagnostic saved to {args.output}; CUDA={report.get("cuda_available", False)}')
