"""Verify downloaded release assets and rebuild indexes.zip without external tools."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(directory, name):
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Unexpected manifest filename')
    return directory / name


def verify(directory, item):
    p = checked_path(directory, item['name'])
    if not p.is_file() or p.stat().st_size != item['bytes']:
        raise ValueError(f'Missing or incomplete file: {p}')
    if digest(p) != item['sha256']:
        raise ValueError(f'SHA-256 mismatch: {p}')
    print(f'OK {p.name}', flush=True)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('.'))
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    manifest = json.loads((directory / 'dataset-manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version')
    originals = {x['name']: x for x in manifest['files']}
    if set(originals) != {'data.zip', 'indexes.zip', 'webshop_sft.json'}:
        raise ValueError('Unexpected original files')
    for name in ['webshop_sft.json', 'data.zip']:
        verify(directory, originals[name])
    target = directory / 'indexes.zip'
    if target.exists():
        verify(directory, originals['indexes.zip'])
    else:
        parts = [verify(directory, part) for part in manifest['index_parts']]
        tmp = directory / 'indexes.zip.partial'
        h = hashlib.sha256()
        size = 0
        with tmp.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                        out.write(block)
                        h.update(block)
                        size += len(block)
        wanted = originals['indexes.zip']
        if size != wanted['bytes'] or h.hexdigest() != wanted['sha256']:
            raise ValueError('Rebuilt index does not match original; .partial retained for inspection')
        tmp.replace(target)
        print('OK indexes.zip rebuilt and verified', flush=True)
    print('All original files match. Next, from the repository root:')
    print(f'webshop-lab import-assets --source "{directory}"')


if __name__ == '__main__':
    main()
