from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from webshop_lab import rollout

arm = sys.argv[1]
root = Path(__file__).resolve().parent
config = root / f"{arm}.config.json"
settings = json.loads(config.read_text())
real_load = rollout.load
real_request = rollout.request
def load(cfg):
    model, tokenizer, dtype = real_load(cfg)
    if cfg["enable_thinking"] is False:
        tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, enable_thinking=False)
    return model, tokenizer, dtype
def request(url, body=None):
    start = time.monotonic()
    endpoint = url.rsplit("/", 1)[-1]
    try:
        result = real_request(url, body)
    except Exception:
        print(f"{endpoint} failed after {time.monotonic() - start:.3f}s", flush=True)
        raise
    if endpoint == "reset":
        print(f"reset {body['task_id']} in {time.monotonic() - start:.3f}s", flush=True)
    return result
rollout.load = load
rollout.request = request
print("source", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "model", settings["model_id"], settings["revision"])
rollout.run(SimpleNamespace(config=config, ids=root/"val-first8.json",
                            output=root/f"{arm}.jsonl", repeats=1, limit=None))
