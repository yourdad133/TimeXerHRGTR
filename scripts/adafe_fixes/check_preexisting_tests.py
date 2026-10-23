"""Compare existing GTR suite failures with the pre-change Git snapshot in memory."""
import argparse
import importlib
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.adafe_diagnostics.common import runtime, save_json

SNAPSHOT = 'adafe-before-fixes-20261011'


def run_suite(output, label):
    for name in ('test_timexer_gtr',):
        sys.modules.pop(name, None)
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / 'tests'), pattern='test_timexer*.py')
    with (output / f'{label}_gtr_tests.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return dict(tests_run=result.testsRun,
        failures=sorted((test.id(), trace.strip().splitlines()[-1]) for test, trace in result.failures),
        errors=sorted((test.id(), trace.strip().splitlines()[-1]) for test, trace in result.errors))


def install_snapshot(relative, attribute):
    source = subprocess.check_output(['git', 'show', f'{SNAPSHOT}:{relative}'], text=True)
    module = types.ModuleType('prechange_' + relative.replace('/', '_').replace('.py', ''))
    module.__file__ = str(ROOT / relative)
    exec(compile(source, f'{SNAPSHOT}:{relative}', 'exec'), module.__dict__)
    live = importlib.import_module(relative[:-3].replace('/', '.'))
    setattr(live, attribute, getattr(module, attribute))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    output = Path(options.output)
    runtime(output / 'preexisting_check_runtime')
    current = run_suite(output, 'current')
    # No checkout/reset/file changes: restore old classes only in this process.
    install_snapshot('models/TimeXerGTR.py', 'Model')
    install_snapshot('data_provider/data_factory.py', 'data_provider')
    before = run_suite(output, 'prechange')
    same = current == before
    record = dict(snapshot=SNAPSHOT, current=current, prechange=before,
                  identical_failure_sets=same,
                  interpretation='Compare retained ordinary GTR contracts against the pre-change model.')
    save_json(output / 'preexisting_test_comparison.json', record)
    print(json.dumps(dict(identical_failure_sets=same, tests_run=current['tests_run'],
                         failures=len(current['failures']), errors=len(current['errors']))))
    if not same:
        raise AssertionError('GTR failure sets changed; investigate before delivery')


if __name__ == '__main__':
    main()
