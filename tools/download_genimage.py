"""Download the official full GenImage split archives with resume and provenance.

Listing is the default. Use --download and an explicit --output to fetch data.
The versioned catalog is a public filename/Drive-ID/byte-size snapshot, not images.
An explicitly selected community mirror is pinned separately and SHA-256 checked.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aig.io import atomic_write_json

OFFICIAL_FOLDER = 'https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS'
REFERENCE = 'https://github.com/GenImage-Dataset/GenImage'
CATALOG = ROOT / 'configs/genimage_downloads.json'
MIRROR_CATALOG = ROOT / 'configs/genimage_huggingface.json'
GIB = 1024 ** 3
GENERATORS = ('midjourney', 'sdv4', 'sdv5', 'adm', 'glide', 'wukong', 'vqdm', 'biggan')
FOLDERS = {'Midjourney': 'midjourney', 'stable_diffusion_v_1_4': 'sdv4',
           'stable_diffusion_v_1_5': 'sdv5', 'ADM': 'adm', 'glide': 'glide',
           'wukong': 'wukong', 'VQDM': 'vqdm', 'BigGAN': 'biggan'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def gdown_module():
    try:
        import gdown
        if version('gdown') != '6.4.2':
            raise ImportError('Use the tested gdown version in requirements/download.txt.')
        return gdown
    except ImportError as error:
        raise RuntimeError('Install optional download dependencies first: '
                           f'"{sys.executable}" -m pip install -r requirements/download.txt') from error


def safe_relative(value):
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or any(part in ('..', '') for part in path.parts) or ':' in str(path):
        raise ValueError(f'Unsafe download path: {value}')
    if len(path.parts) != 2 or path.parts[0] not in FOLDERS:
        raise ValueError(f'Unknown official generator folder: {value}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.(?:zip|z\d{2,})', path.name):
        raise ValueError(f'Unexpected archive filename: {value}')
    return path.as_posix()


def validate_catalog(catalog):
    if catalog.get('official_folder') != OFFICIAL_FOLDER:
        raise ValueError('The catalog must identify the official GenImage Drive folder.')
    groups, ids, paths = defaultdict(list), set(), set()
    for entry in catalog['files']:
        path = safe_relative(entry['path'])
        if path in paths or entry['generator'] != FOLDERS[PurePosixPath(path).parts[0]]:
            raise ValueError('Duplicate catalog path or incorrect generator mapping.')
        paths.add(path)
        if type(entry['bytes']) is not int or entry['bytes'] <= 0 or not entry['ids']:
            raise ValueError('Every file requires a positive byte count and Drive IDs.')
        for file_id in entry['ids']:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', file_id) or file_id in ids:
                raise ValueError('Invalid or repeated Drive file ID.')
            ids.add(file_id)
        groups[entry['generator']].append(PurePosixPath(path))
    if set(groups) != set(GENERATORS):
        raise ValueError('The full catalog must contain all eight generators, including SD1.4.')
    for generator, files in groups.items():
        zips = [path for path in files if path.suffix == '.zip']
        if len(zips) != 1:
            raise ValueError(f'Expected one final .zip per generator: {generator}')
        parts = [path for path in files if path.suffix != '.zip']
        if any(path.stem != zips[0].stem for path in parts):
            raise ValueError(f'Mismatched split-archive basename: {generator}')
        numbers = sorted(int(path.suffix[2:]) for path in parts)
        if not numbers or numbers != list(range(1, len(parts) + 1)):
            raise ValueError(f'Missing split-archive volume: {generator}')
    return catalog


def public_file_size(file_id):
    """Read only public Drive viewer metadata; fail closed if its schema changes."""
    import requests
    with requests.get(f'https://drive.google.com/file/d/{file_id}/view', timeout=45) as response:
        response.raise_for_status()
        match = re.search(r'itemJson:\s*(\[)', response.text)
        if not match:
            raise ValueError(f'Public Drive metadata unavailable: {file_id}')
        item, _ = json.JSONDecoder().raw_decode(response.text[match.start(1):])
    try:
        size = int(item[25][2])
        if file_id not in item[16] or size <= 0:
            raise ValueError('Unexpected identity or size')
        return size
    except (TypeError, IndexError, ValueError) as error:
        raise ValueError(f'Drive metadata schema changed for {file_id}; cached catalog is unchanged.') from error


def discover_catalog():
    files = gdown_module().download_folder(url=OFFICIAL_FOLDER, skip_download=True,
                                          quiet=True, use_cookies=False, timeout=45)
    grouped = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = pool.map(public_file_size, [item.id for item in files])
        for item, size in zip(files, sizes):
            path = safe_relative(item.path)
            entry = grouped.setdefault(path, {'path': path, 'generator': FOLDERS[PurePosixPath(path).parts[0]],
                                              'bytes': size, 'ids': []})
            if entry['bytes'] != size:
                raise ValueError(f'Duplicate Drive names have different sizes; review required: {path}')
            entry['ids'].append(item.id)
    catalog = {'format_version': 1, 'official_folder': OFFICIAL_FOLDER, 'reference': REFERENCE,
               'captured_utc': utc_now(),
               'checksum_policy': 'Sizes from public Drive metadata. SHA-256 is recorded locally on download; no publisher digest is asserted.',
               'files': sorted(grouped.values(), key=lambda entry: entry['path'])}
    return validate_catalog(catalog)


def validate_mirror(mirror, catalog):
    if (mirror.get('repository') != 'jzousz/GenImage'
            or not re.fullmatch(r'[0-9a-f]{40}', mirror.get('revision', ''))):
        raise ValueError('Expected the pinned community GenImage mirror revision.')
    official = {entry['path']: entry['bytes'] for entry in catalog['files']}
    files = {}
    for entry in mirror['files']:
        path = safe_relative(entry['path'])
        if (path in files or entry['bytes'] != official.get(path)
                or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', ''))):
            raise ValueError('Mirror path, size or SHA-256 differs from the reviewed catalog.')
        files[path] = entry
    if set(files) != set(official):
        raise ValueError('Mirror catalog must cover every official archive path.')
    return files


def file_tasks(catalog, generators, source='google-drive', selected_files=None, mirror=None):
    """Keep duplicate Drive IDs separately until their downloaded bytes agree."""
    mirror_files = validate_mirror(mirror, catalog) if source == 'huggingface' else None
    if source not in ('google-drive', 'huggingface'):
        raise ValueError('Unknown download source.')
    selected = {safe_relative(path) for path in selected_files} if selected_files else None
    available = {entry['path'] for entry in catalog['files'] if entry['generator'] in generators}
    if selected and not selected.issubset(available):
        raise ValueError('Selected --files must exist in the selected generators and reviewed catalog.')
    tasks = []
    for entry in catalog['files']:
        if entry['generator'] not in generators or (selected and entry['path'] not in selected):
            continue
        if mirror_files is not None:
            identity = f"{mirror['repository']}@{mirror['revision']}:{entry['path']}"
            url = f"https://huggingface.co/datasets/{mirror['repository']}/resolve/{mirror['revision']}/{quote(entry['path'], safe='/')}"
            tasks.append(dict(entry, id=identity, url=url, local_path=entry['path'],
                              sha256=mirror_files[entry['path']]['sha256'], provider='huggingface', duplicate=False))
            continue
        for i, file_id in enumerate(entry['ids']):
            path = entry['path'] if i == 0 else f"_duplicate_sources/{file_id}/{PurePosixPath(entry['path']).name}"
            tasks.append(dict(entry, id=file_id, local_path=path, provider='google-drive', duplicate=i > 0))
    return tasks


def target_path(output, task):
    path = (output / task['local_path']).resolve()
    if not path.is_relative_to(output) or path == output:
        raise ValueError('Download target escapes its selected directory.')
    return path


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_file(path, task, saved=None):
    if not path.is_file() or path.stat().st_size != task['bytes']:
        raise ValueError(f"Wrong or incomplete file size: {path}; expected {task['bytes']} bytes. Existing data was preserved.")
    digest = sha256(path)
    if task.get('sha256') and digest != task['sha256']:
        raise ValueError(f'Download SHA-256 does not match the pinned mirror hash: {path}')
    if saved and (saved['id'] != task['id'] or saved['bytes'] != task['bytes'] or saved['sha256'] != digest):
        raise ValueError(f'Existing download differs from its recorded identity/checksum: {path}')
    return {'id': task['id'], 'bytes': task['bytes'], 'sha256': digest, 'verified_utc': utc_now(),
            'provider': task.get('provider', 'google-drive'),
            'hash_reference': 'pinned community mirror' if task.get('sha256') else 'local snapshot only'}


def partial_bytes(target):
    stage = target.with_name(target.name + '.downloading')
    candidates = [stage] if stage.exists() else list(target.parent.glob(stage.name + '*.part')) if target.parent.exists() else []
    if len(candidates) > 1:
        raise ValueError(f'Multiple partial downloads for {target}; preserve and inspect them before retrying.')
    return candidates[0].stat().st_size if candidates else 0


def check_space(output, tasks, reserve_gib):
    ancestor = output
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise ValueError(f'Output drive does not exist: {output}')
        ancestor = ancestor.parent
    remaining = 0
    for task in tasks:
        target = target_path(output, task)
        if target.exists():
            if target.stat().st_size != task['bytes']:
                raise ValueError(f'Unexpected existing file size: {target}')
        else:
            remaining += max(0, task['bytes'] - partial_bytes(target))
    free = shutil.disk_usage(ancestor).free
    if free < remaining + reserve_gib * GIB:
        raise ValueError(f'Insufficient storage: download needs {remaining/GIB:.2f} GiB more plus '
                         f'{reserve_gib:g} GiB reserve; this drive has {free/GIB:.2f} GiB free. '
                         'Choose a larger drive or use --generators to download one generator at a time. '
                         'Extraction requires additional space.')
    return {'additional_download_gib': remaining/GIB, 'free_gib': free/GIB, 'reserve_gib': reserve_gib}


def fetch_file(task, target, retries, speed_mib=None, downloader=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + '.downloading')
    download = downloader or gdown_module().download
    if not stage.exists():
        def guard(downloaded, _total):
            if downloaded > task['bytes']:
                raise ValueError('Transfer exceeded the recorded source size; partial data preserved.')
        source = {'url': task['url']} if task.get('url') else {'id': task['id']}
        result = download(**source, output=str(stage), resume=True, use_cookies=False,
                          verify=True, timeout=60, retries=retries, progress=guard,
                          speed=speed_mib * 1024**2 if speed_mib else None)
        if not result:
            raise RuntimeError('Google Drive did not return a downloaded file.')
    record = verify_file(stage, task)
    if target.exists():
        raise ValueError(f'Refusing to replace an existing file: {target}')
    stage.rename(target)
    return record


def check_duplicates(tasks, records):
    by_path = defaultdict(list)
    for task in tasks:
        if task['local_path'] in records:
            by_path[task['path']].append(records[task['local_path']]['sha256'])
    for path, digests in by_path.items():
        if len(set(digests)) > 1:
            raise ValueError(f'Duplicate public Drive filenames contain different bytes: {path}. '
                             'Both files are preserved; inspect the source before choosing one.')


def test_archives(output, catalog, generators, seven_zip):
    executable = shutil.which(seven_zip) or (seven_zip if Path(seven_zip).is_file() else None)
    if not executable:
        raise ValueError('7-Zip was not found. Install it and pass --seven-zip with its executable path.')
    for entry in catalog['files']:
        if entry['generator'] in generators and entry['path'].endswith('.zip'):
            print(f"Testing complete split archive: {entry['path']}", flush=True)
            subprocess.run([executable, 't', str(output / entry['path'])], check=True)


def run_download(catalog, output, generators, retries=3, reserve_gib=10,
                 verify_only=False, speed_mib=None, downloader=None,
                 source='google-drive', selected_files=None, mirror=None):
    output = Path(output).expanduser().resolve()
    tasks = file_tasks(catalog, generators, source, selected_files, mirror)
    storage = check_space(output, tasks, reserve_gib) if not verify_only else None
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'download_status.json'
    old = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    if old and old.get('source', 'google-drive') != source:
        raise ValueError('Use a separate output directory for each download provider; prior bytes remain preserved.')
    records = old.get('files', {})
    state = {'status': 'running', 'source': source, 'official_folder': OFFICIAL_FOLDER, 'selected_generators': list(generators),
             'selected_files': selected_files, 'mirror': {key: mirror[key] for key in ('repository', 'revision')} if mirror else None,
             'selected_file_count': len(tasks), 'files': records, 'failures': {}, 'storage': storage}
    catalog_path = output / 'download_catalog.json'
    if catalog_path.exists() and json.loads(catalog_path.read_text(encoding='utf-8'))['files'] != catalog['files']:
        raise ValueError('This directory belongs to a different source catalog. Use a new directory for a changed snapshot.')
    atomic_write_json(catalog_path, catalog)
    atomic_write_json(status_path, state)
    try:
        for index, task in enumerate(tasks, 1):
            relative = task['local_path']
            print(f"[{index}/{len(tasks)}] {relative} ({task['bytes']/GIB:.2f} GiB)", flush=True)
            try:
                target = target_path(output, task)
                if target.exists() or verify_only:
                    record = verify_file(target, task, records.get(relative))
                    print('  SHA-256 verified; no download.', flush=True)
                else:
                    record = fetch_file(task, target, retries, speed_mib, downloader)
                records[relative] = record
            except (OSError, ValueError, RuntimeError) as error:
                state['failures'][relative] = str(error)
                print(f'  FAILED: {error}', file=sys.stderr, flush=True)
            except Exception as error:
                # gdown has its own typed HTTP/quota errors. Keep remaining files runnable.
                state['failures'][relative] = f'{type(error).__name__}: {error}'
                print(f'  FAILED: {error}', file=sys.stderr, flush=True)
            state['updated_utc'] = utc_now()
            atomic_write_json(status_path, state)
        check_duplicates(tasks, records)
        if state['failures']:
            raise RuntimeError(f"{len(state['failures'])} file(s) incomplete. Rerun the same command to resume; see {status_path}")
        state['status'] = 'completed'
        state['scope'] = ('selected_archive_files_only' if selected_files else
                          'all_eight_generators' if set(generators) == set(GENERATORS) else 'selected_generators_only')
        state['completed_utc'] = utc_now()
    except BaseException:
        state['status'] = 'interrupted' if sys.exc_info()[0] is KeyboardInterrupt else 'failed'
        state['updated_utc'] = utc_now()
        atomic_write_json(status_path, state)
        raise
    atomic_write_json(status_path, state)
    print(f"Download scope complete: {len(tasks)} files; {state['scope']}. Archives remain compressed.", flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list', action='store_true', help='List source archive counts and sizes; downloads nothing (default).')
    mode.add_argument('--download', action='store_true', help='Download every selected archive part and verify byte counts/SHA-256.')
    mode.add_argument('--verify-only', action='store_true', help='Rehash selected local files; no dataset downloads.')
    parser.add_argument('--output', help='Archive directory on a drive with sufficient storage. Required for download/verify.')
    parser.add_argument('--generators', nargs='+', choices=GENERATORS, default=list(GENERATORS))
    parser.add_argument('--files', nargs='+', help='Download/verify only these exact catalog paths, e.g. glide/imagenet_glide.zip.')
    parser.add_argument('--source', choices=('google-drive', 'huggingface'), default='google-drive',
                        help='Official Google Drive or explicitly selected pinned community mirror.')
    parser.add_argument('--retries', type=int, default=3, help='Automatic retries per transient file transfer failure.')
    parser.add_argument('--reserve-gib', type=float, default=10)
    parser.add_argument('--speed-mib', type=float, help='Optional per-file download speed limit in MiB/second.')
    parser.add_argument('--refresh-index', metavar='JSON_PATH', help='Discover current public filenames/sizes and save a new catalog here, without replacing the versioned snapshot.')
    parser.add_argument('--catalog', default=str(CATALOG), help='Versioned catalog or a previously reviewed refreshed catalog.')
    parser.add_argument('--mirror-catalog', default=str(MIRROR_CATALOG))
    parser.add_argument('--test-archives', action='store_true', help='After download/verify, test all archive volumes with 7-Zip; does not extract.')
    parser.add_argument('--seven-zip', default='7z', help='7z/7zz command or absolute executable path.')
    args = parser.parse_args(argv)
    if (args.retries < 0 or args.reserve_gib < 0 or not args.reserve_gib < float('inf')
            or (args.speed_mib is not None and not 0 < args.speed_mib < float('inf'))
            or len(set(args.generators)) != len(args.generators)):
        parser.error('Use nonnegative retries/reserve, positive speed and distinct generators.')
    if (args.download or args.verify_only) and not args.output:
        parser.error('--output is required for download/verify.')
    if args.test_archives and not (args.download or args.verify_only):
        parser.error('--test-archives requires --download or --verify-only.')
    if args.test_archives and args.files:
        parser.error('--test-archives requires complete generator downloads, not --files selection.')
    if args.refresh_index:
        catalog = discover_catalog()
        destination = Path(args.refresh_index).expanduser().resolve()
        if destination.exists() or destination == CATALOG.resolve():
            parser.error('Save the refreshed index to a new JSON path; existing snapshots are preserved.')
        atomic_write_json(destination, catalog)
        print(f'Current public catalog saved: {destination}', flush=True)
    else:
        catalog = validate_catalog(json.loads(Path(args.catalog).read_text(encoding='utf-8')))
    mirror = json.loads(Path(args.mirror_catalog).read_text(encoding='utf-8')) if args.source == 'huggingface' else None
    tasks = file_tasks(catalog, args.generators, args.source, args.files, mirror)
    print('Source: ' + ('official Google Drive' if args.source == 'google-drive' else
                       f"community Hugging Face mirror {mirror['repository']}@{mirror['revision']} (not publisher-authenticated)"), flush=True)
    print('Full GenImage archive inventory (GiB = 1024^3 bytes):', flush=True)
    for generator in args.generators:
        selected = [task for task in tasks if task['generator'] == generator]
        if not selected:
            continue
        print(f"  {generator:12s} {len(selected):3d} files  {sum(t['bytes'] for t in selected)/GIB:8.2f} GiB", flush=True)
    print(f"Total: {len(tasks)} downloads, {sum(t['bytes'] for t in tasks)/GIB:.2f} GiB. "
          'Extraction needs additional space.', flush=True)
    if not args.download and not args.verify_only:
        return
    # A lock prevents two scripts writing the same partial files concurrently.
    from filelock import FileLock
    output = Path(args.output).expanduser().resolve()
    check_space(output, tasks, args.reserve_gib) if args.download else None
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'download.lock'), timeout=0):
        run_download(catalog, output, args.generators, args.retries, args.reserve_gib,
                     args.verify_only, args.speed_mib, source=args.source,
                     selected_files=args.files, mirror=mirror)
        if args.test_archives:
            test_archives(output, catalog, args.generators, args.seven_zip)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('Interrupted. Partial files are preserved; rerun the same command to resume.')
    except Exception as error:
        sys.exit(str(error))
