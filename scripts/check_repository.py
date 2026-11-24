"""Validate committed-source hygiene and local relative Markdown links."""
import re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
errors = []
# Only first-party docs: upstream contains historical links not maintained here.
for path in [root / 'README.md', *root.glob('docs/*.md'), *root.glob('reports/*.md')]:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        if target.startswith(('https://', 'http://', '#', 'mailto:')):
            continue
        dest = (path.parent / target.split('#', 1)[0]).resolve()
        if not dest.exists():
            errors.append(f'Broken link: {path.relative_to(root)} -> {target}')
# Do not print credentials, even when a pattern matches.
secret = re.compile(r'(?:sk-[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
for path in [*root.glob('src/**/*.py'), *root.glob('docs/*.md'), *root.glob('upstream/**/*')]:
    if path.is_file() and path.suffix in {'.py', '.sh', '.json', '.md', '.yaml', '.yml'}:
        if secret.search(path.read_text(encoding='utf-8', errors='ignore')):
            errors.append(f'Credential-like string: {path.relative_to(root)}')
if errors:
    raise SystemExit('\n'.join(errors))
print('Relative docs links and credential patterns: passed')
