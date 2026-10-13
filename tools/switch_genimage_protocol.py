"""Switch a live pooled queue at the Tiny boundary, preserving current ProGAN work."""

import argparse
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aig.genimage_protocol import audit_manifest_roots
from aig.io import atomic_write_json
from tools.run_full_training import pid_alive, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    args = parser.parse_args()
    batch = Path(args.batch_dir)
    batch = batch.resolve() if batch.is_absolute() else (ROOT / batch).resolve()
    if not batch.is_relative_to(ROOT / 'runs') or os.name != 'nt' or not re.fullmatch(r'[A-Za-z0-9_-]+', batch.name):
        parser.error('This live migration helper targets a Windows batch under runs/.')
    audit = audit_manifest_roots(ROOT / 'dataset/prepared')
    control_path = batch / 'genimage_protocol_switch.json'
    old_plan = read_json(batch / 'plan.json')
    if not old_plan or old_plan.get('genimage_protocol') == 'cross-generator':
        parser.error('Select the currently pooled batch.')
    control = {'status': 'waiting_for_tiny_boundary', 'coordinator_pid': os.getpid(),
               'requested_utc': datetime.now(timezone.utc).isoformat(), 'protocol': 'cross-generator',
               'training_runs_after_switch': 45, 'tiny_matrix_cells': 245, 'dataset_audit': audit}
    atomic_write_json(control_path, control)
    while True:
        state = read_json(batch / 'status.json')
        if not state:
            time.sleep(1)
            continue
        manager = state['manager_pid']
        if not pid_alive(manager):
            control.update(status='ready_after_manager_exit')
            break
        active = state.get('active_job') or ''
        if active.startswith('tiny_genimage/') or state['status'] in ('failed', 'completed', 'interrupted'):
            # Stop only the verified old manager tree. Its current Tiny run is replaced
            # by separate generator runs; completed ProGAN checkpoints stay untouched.
            atomic_write_json(batch / 'status_before_genimage_replan.json', state)
            guard = (
                f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(manager)}'; "
                "if(-not $p -or $p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*run_full_training.py*' "
                f"-or $p.CommandLine -notlike '*{batch.name}*'){{throw 'Manager identity differs; refusing to stop.'}}"
            )
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', guard], check=True)
            subprocess.run(['taskkill.exe', '/PID', str(manager), '/T', '/F'], check=True)
            while pid_alive(manager):
                time.sleep(.2)
            control.update(status='switching', old_manager_pid=manager)
            atomic_write_json(control_path, control)
            break
        control['waiting_on_job'] = active
        control['checked_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(control_path, control)
        time.sleep(.25)
    command = [sys.executable, '-u', str(ROOT / 'tools/run_full_training.py'), '--batch-dir', str(batch),
               '--datasets', *old_plan['datasets'], '--models', *old_plan['models'], '--seeds', *map(str, old_plan['seeds']),
               '--epochs', str(old_plan['epochs']), '--patience', str(old_plan['patience']),
               '--batch-size', str(old_plan['batch_size']), '--workers', str(old_plan['workers']),
               '--port', '8765', '--genimage-protocol', 'cross-generator',
               '--source-revision', 'source_genimage_protocol', '--replan']
    with (batch / 'manager_genimage.stdout.log').open('a', encoding='utf-8') as out, (batch / 'manager_genimage.stderr.log').open('a', encoding='utf-8') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    control.update(status='launched', new_launcher_pid=process.pid, command=command,
                   switched_utc=datetime.now(timezone.utc).isoformat())
    atomic_write_json(control_path, control)
    for _ in range(120):
        state = read_json(batch / 'status.json')
        if state and state.get('settings', {}).get('genimage_protocol') == 'cross-generator' and state['status'] == 'running':
            control.update(status='active', new_manager_pid=state['manager_pid'])
            atomic_write_json(control_path, control)
            return
        if process.poll() is not None:
            raise RuntimeError('The new manager exited; inspect manager_genimage.stderr.log.')
        time.sleep(1)
    raise RuntimeError('Protocol switch has not reached a running manager; inspect the saved logs.')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--batch-dir' in sys.argv:
            path = Path(sys.argv[sys.argv.index('--batch-dir') + 1])
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            if path.is_relative_to(ROOT / 'runs'):
                record = read_json(path / 'genimage_protocol_switch.json', {})
                atomic_write_json(path / 'genimage_protocol_switch.json', dict(record, status='failed', error=str(error)))
        raise
