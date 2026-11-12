import sys

import pytest


def _run_main(monkeypatch, tmp_path, extra):
    import main as main_mod
    seen = {}

    def fake_run(cmd, *a, **k):
        seen['cmd'] = cmd
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(main_mod.subprocess, 'run', fake_run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--mode', 'demo', '--min-free-gb', '0', '--cache-dir', str(tmp_path / 'c')] + extra)
    main_mod.main()
    return seen['cmd']


def test_main_forwards_threshold_criterion_default_point(monkeypatch, tmp_path):
    cmd = _run_main(monkeypatch, tmp_path, [])
    i = cmd.index('--threshold-criterion')
    assert cmd[i + 1] == 'point'
    assert cmd.index('--target-precision') < i


def test_main_forwards_threshold_criterion_wilson(monkeypatch, tmp_path):
    cmd = _run_main(monkeypatch, tmp_path, ['--threshold-criterion', 'wilson', '--target-precision', '0.85'])
    assert cmd[cmd.index('--threshold-criterion') + 1] == 'wilson'
    assert cmd[cmd.index('--target-precision') + 1] == '0.85'


def test_main_rejects_unknown_threshold_criterion(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, ['--threshold-criterion', 'median'])
