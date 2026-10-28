"""Full GenImage inventory, disk preflight, download provenance and resume checks."""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools.download_genimage import (CATALOG, MIRROR_CATALOG, GENERATORS, GIB, check_duplicates,
                                    check_space, fetch_file, file_tasks, partial_bytes,
                                    run_download, safe_relative, validate_catalog, validate_mirror, verify_file)


class GenImageDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = validate_catalog(json.loads(CATALOG.read_text(encoding='utf-8')))
        cls.mirror = json.loads(MIRROR_CATALOG.read_text(encoding='utf-8'))
        cls.temp_root = Path(__file__).resolve().parents[1] / 'tmp' / 'genimage_download_tests'
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = {'id': 'public-file', 'bytes': 10, 'path': 'ADM/data.z01',
                     'local_path': 'ADM/data.z01', 'generator': 'adm'}

    def test_snapshot_covers_eight_generators_and_every_volume(self):
        tasks = file_tasks(self.catalog, GENERATORS)
        self.assertEqual(len(self.catalog['files']), 207)
        self.assertEqual(len(tasks), 208)
        self.assertEqual({task['generator'] for task in tasks}, set(GENERATORS))
        self.assertAlmostEqual(sum(t['bytes'] for t in tasks)/GIB, 611.3339602453634)
        duplicate = [task for task in tasks if task['duplicate']]
        self.assertEqual(len(duplicate), 1)
        self.assertIn('sdv5.z26', duplicate[0]['path'])
        self.assertNotEqual(duplicate[0]['path'], duplicate[0]['local_path'])
        self.assertEqual(len(file_tasks(self.catalog, ['sdv4'])), 30)

    def test_missing_volume_generator_duplicate_id_and_unsafe_paths_rejected(self):
        for change in ('volume', 'generator', 'id'):
            broken = deepcopy(self.catalog)
            if change == 'volume':
                broken['files'] = [entry for entry in broken['files'] if not entry['path'].endswith('adm.z01')]
            elif change == 'generator':
                broken['files'] = [entry for entry in broken['files'] if entry['generator'] != 'sdv4']
            else:
                broken['files'][1]['ids'] = broken['files'][0]['ids']
            with self.assertRaises(ValueError):
                validate_catalog(broken)
        for path in ('../x.zip', '/ADM/x.zip', 'C:/ADM/x.zip', 'ADM/../../x.zip', 'ADM/readme.txt'):
            with self.assertRaises(ValueError):
                safe_relative(path)

    def test_pinned_mirror_covers_official_paths_sizes_and_hashes(self):
        validate_mirror(self.mirror, self.catalog)
        tasks = file_tasks(self.catalog, GENERATORS, source='huggingface', mirror=self.mirror)
        self.assertEqual(len(tasks), 207)
        self.assertAlmostEqual(sum(t['bytes'] for t in tasks)/GIB, 608.3339602453634)
        self.assertTrue(all(self.mirror['revision'] in t['url'] and len(t['sha256']) == 64 for t in tasks))
        bad = deepcopy(self.mirror)
        bad['files'][0]['bytes'] += 1
        with self.assertRaises(ValueError):
            validate_mirror(bad, self.catalog)
        bad = deepcopy(self.mirror)
        bad['files'].pop()
        with self.assertRaises(ValueError):
            validate_mirror(bad, self.catalog)

    def test_exact_file_selection_is_bounded_and_unknown_paths_are_rejected(self):
        selected = ['glide/imagenet_glide.zip']
        tasks = file_tasks(self.catalog, GENERATORS, 'huggingface', selected, self.mirror)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['bytes'], 51607857)
        with self.assertRaises(ValueError):
            file_tasks(self.catalog, ['adm'], selected_files=selected)
        with self.assertRaises(ValueError):
            file_tasks(self.catalog, GENERATORS, selected_files=['ADM/not_in_catalog.zip'])

    def test_wrong_mirror_hash_is_never_published(self):
        target = self.root / self.task['local_path']
        task = dict(self.task, sha256='0'*64, url='https://example.invalid/pinned/data.z01')
        def wrong_bytes(**kwargs):
            self.assertIn('url', kwargs)
            self.assertNotIn('id', kwargs)
            Path(kwargs['output']).write_bytes(b'0123456789')
            return kwargs['output']
        with self.assertRaisesRegex(ValueError, 'pinned mirror hash'):
            fetch_file(task, target, retries=0, downloader=wrong_bytes)
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name(target.name + '.downloading').exists())

    def test_provider_change_is_rejected_without_mutating_previous_status(self):
        status = self.root / 'download_status.json'
        content = json.dumps({'source': 'google-drive', 'files': {}})
        status.write_text(content)
        with self.assertRaisesRegex(ValueError, 'separate output directory'):
            run_download(self.catalog, self.root, GENERATORS, reserve_gib=0, source='huggingface',
                         selected_files=['glide/imagenet_glide.zip'], mirror=self.mirror)
        self.assertEqual(status.read_text(), content)

    def test_insufficient_storage_stops_before_creating_output(self):
        output = self.root / 'not-created'
        with patch('tools.download_genimage.shutil.disk_usage') as usage:
            usage.return_value.free = 1
            with self.assertRaisesRegex(ValueError, 'Insufficient storage'):
                check_space(output, [self.task], 0)
        self.assertFalse(output.exists())

    def test_disk_check_accounts_for_partial_and_complete_downloads(self):
        target = self.root / self.task['local_path']
        target.parent.mkdir()
        partial = target.with_name(target.name + '.downloading123.part')
        partial.write_bytes(b'abc')
        self.assertEqual(partial_bytes(target), 3)
        with patch('tools.download_genimage.shutil.disk_usage') as usage:
            usage.return_value.free = 7
            self.assertEqual(check_space(self.root, [self.task], 0)['additional_download_gib'], 7/GIB)
        partial.rename(target)
        with self.assertRaisesRegex(ValueError, 'Unexpected existing file size'):
            check_space(self.root, [self.task], 0)

    def test_completion_publishes_only_size_verified_bytes_and_hash(self):
        target = self.root / self.task['local_path']
        seen = []
        def downloader(**kwargs):
            seen.append(kwargs)
            Path(kwargs['output']).write_bytes(b'0123456789')
            kwargs['progress'](10, 10)
            return kwargs['output']
        record = fetch_file(self.task, target, retries=3, downloader=downloader)
        self.assertEqual(target.read_bytes(), b'0123456789')
        self.assertEqual(record['sha256'], verify_file(target, self.task, record)['sha256'])
        self.assertTrue(seen[0]['resume'])
        self.assertTrue(seen[0]['verify'])
        self.assertFalse(seen[0]['use_cookies'])
        self.assertEqual(seen[0]['retries'], 3)
        target.write_bytes(b'9876543210')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_file(target, self.task, record)

    def test_truncated_or_excessive_response_is_not_published(self):
        target = self.root / self.task['local_path']
        def short(**kwargs):
            Path(kwargs['output']).write_bytes(b'abc')
            return kwargs['output']
        with self.assertRaisesRegex(ValueError, 'incomplete file size'):
            fetch_file(self.task, target, retries=0, downloader=short)
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name(target.name + '.downloading').exists())

    def test_duplicate_ids_must_agree_instead_of_overwriting(self):
        tasks = [{'path': 'same.z26', 'local_path': 'same.z26'},
                 {'path': 'same.z26', 'local_path': 'duplicate.z26'}]
        check_duplicates(tasks, {'same.z26': {'sha256': 'a'}, 'duplicate.z26': {'sha256': 'a'}})
        with self.assertRaisesRegex(ValueError, 'different bytes'):
            check_duplicates(tasks, {'same.z26': {'sha256': 'a'}, 'duplicate.z26': {'sha256': 'b'}})

    def test_rerun_skips_verified_files_and_labels_selected_scope(self):
        catalog = {'files': [dict(self.task, ids=['public-file'])]}
        calls = []
        def downloader(**kwargs):
            calls.append(kwargs)
            Path(kwargs['output']).write_bytes(b'0123456789')
            return kwargs['output']
        first = run_download(catalog, self.root, ['adm'], reserve_gib=0, downloader=downloader)
        second = run_download(catalog, self.root, ['adm'], reserve_gib=0, downloader=downloader)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first['scope'], 'selected_generators_only')
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(second['files']['ADM/data.z01']['sha256'], first['files']['ADM/data.z01']['sha256'])

    def test_failed_file_does_not_prevent_remaining_files_or_claim_success(self):
        catalog = {'files': [dict(self.task, ids=['bad']),
                             dict(self.task, ids=['good'], path='ADM/next.zip')]}
        def downloader(**kwargs):
            if kwargs['id'] == 'bad':
                raise RuntimeError('quota')
            Path(kwargs['output']).write_bytes(b'0123456789')
            return kwargs['output']
        with self.assertRaisesRegex(RuntimeError, '1 file'):
            run_download(catalog, self.root, ['adm'], reserve_gib=0, downloader=downloader)
        self.assertTrue((self.root / 'ADM/next.zip').is_file())
        state = json.loads((self.root / 'download_status.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertIn('quota', state['failures']['ADM/data.z01'])

    @unittest.skipUnless(os.environ.get('AIG_TEST_DOWNLOAD_HTTP') == '1', 'Opt-in localhost HTTP integration check')
    def test_real_gdown_resumes_valid_ranges_and_preserves_rejected_ranges(self):
        try:
            import gdown
        except ImportError:
            self.skipTest('Optional gdown is not installed; all core downloader checks still run.')
        payload = b'0123456789' * 2048
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                requested = self.headers.get('Range')
                requests.append((self.path, requested))
                start = int(requested.removeprefix('bytes=').removesuffix('-')) if requested else 0
                if self.path == '/refuse':
                    start = 0
                self.send_response(206 if start else 200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(payload) - start))
                if start:
                    self.send_header('Content-Range', f'bytes {start}-{len(payload)-1}/{len(payload)}')
                self.end_headers()
                try:
                    self.wfile.write(payload[start:])
                except (ConnectionError, OSError):
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            task = dict(self.task, bytes=len(payload))
            for route in ('valid', 'refuse'):
                target = self.root / route / 'part.z01'
                target.parent.mkdir()
                partial = target.with_name(target.name + '.downloading-existing.part')
                partial.write_bytes(payload[:1024])
                def download(**kwargs):
                    kwargs.pop('id')
                    return gdown.download(url=f'http://127.0.0.1:{server.server_port}/{route}',
                                          quiet=True, **kwargs)
                if route == 'valid':
                    record = fetch_file(task, target, retries=0, downloader=download)
                    self.assertEqual(target.read_bytes(), payload)
                    verify_file(target, task, record)
                else:
                    with self.assertRaisesRegex(Exception, 'requested byte range'):
                        fetch_file(task, target, retries=0, downloader=download)
                    self.assertFalse(target.exists())
                    self.assertEqual(partial.read_bytes(), payload[:1024])
            self.assertIn(('/valid', 'bytes=1024-'), requests)
            self.assertIn(('/refuse', 'bytes=1024-'), requests)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
