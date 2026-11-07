import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import onyx_console as oc

ROOT = Path(__file__).resolve().parent.parent


class FakeTty(io.StringIO):
    encoding = 'utf-8'

    def isatty(self):
        return True


def make_ascii_stream():
    class AsciiTty(io.StringIO):
        encoding = 'cp1251'

        def isatty(self):
            return True

        def write(self, s):
            s.encode('cp1251')
            return super().write(s)

    return AsciiTty()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize('glyphs', [oc.UNICODE_GLYPHS, oc.ASCII_GLYPHS])
@pytest.mark.parametrize('width', [1, 5, 12, 20, 33, 40, 60, 79, 120, 200])
@pytest.mark.parametrize('color', [False, True])
def test_lines_never_exceed_width(glyphs, width, color):
    for done in (0, 7, 100, 246):
        l1, l2 = oc.render_lines(done, 600, done - 3 if done > 3 else done, 3 if done > 3 else 0, 'training: direction A and a very long status text', 8047, 11560, 5, width, glyphs, color=color)
        for line in (l1, l2):
            plain = line
            for code in ('\x1b[32m', '\x1b[31m', '\x1b[0m'):
                plain = plain.replace(code, '')
            assert len(plain) <= width


def test_percent_and_eta():
    clock = Clock()
    con = oc.Console(total=10, workers=2, stream=io.StringIO(), interactive=False, clock=clock, width=100)
    for _ in range(4):
        con.advance(ok=True, duration=10.0)
    assert con.eta() == 30
    clock.t += 3661
    l1, l2 = con.lines()
    assert '40%' in l1
    assert '4/10 stars' in l1
    assert 'ETA 0:00:30' in l2
    assert 'elapsed 1:01:01' in l2
    assert 'ok 4' in l2 and 'failed 0' in l2


def test_eta_unknown_before_first_star():
    con = oc.Console(total=5, stream=io.StringIO(), interactive=False, width=100)
    assert con.eta() is None
    assert 'ETA --:--:--' in con.lines()[1]


def test_zero_total():
    con = oc.Console(total=0, stream=io.StringIO(), interactive=False, width=80)
    l1, l2 = con.lines()
    assert '100%' in l1
    assert '0/0 stars' in l1
    assert con.eta() == 0


def test_full_completion():
    con = oc.Console(total=4, stream=io.StringIO(), interactive=False, width=80, glyphs=oc.ASCII_GLYPHS)
    for i in range(4):
        con.advance(ok=(i != 1), duration=2.0)
    l1, l2 = con.lines()
    assert '100%' in l1
    assert '4/4 stars' in l1
    assert '-' not in l1.split('%')[0].strip()
    assert 'ETA 0:00:00' in l2
    assert 'ok 3' in l2 and 'failed 1' in l2


def test_not_a_terminal_plain_lines():
    stream = io.StringIO()
    con = oc.Console(total=20, workers=2, stream=stream)
    assert not con.interactive
    con.start()
    con.set_status('extracting')
    for _ in range(20):
        con.advance(ok=True, duration=1.0)
    con.show('hello')
    con.finish(['summary'])
    out = stream.getvalue()
    assert '\r' not in out
    assert '\x1b' not in out
    lines = [x for x in out.split('\n') if x.startswith('[')]
    assert lines
    assert lines[-1].startswith('[20/20] 100%')
    assert len(lines) <= 11
    assert 'hello' in out
    assert out.rstrip().endswith('summary')


def test_ascii_fallback_for_cp1251():
    stream = make_ascii_stream()
    con = oc.Console(total=10, stream=stream, width=70)
    assert con.glyphs is oc.ASCII_GLYPHS
    con.start()
    con.set_status('extracting features')
    con.advance(ok=True, duration=1.0)
    con.show('message')
    con.finish(['done'])
    out = stream.getvalue()
    assert '#' in out
    l1, l2 = con.lines()
    for ch in l1 + l2:
        assert ord(ch) < 128


def test_unicode_stream_uses_blocks():
    stream = FakeTty()
    con = oc.Console(total=10, stream=stream, width=70)
    assert con.glyphs is oc.UNICODE_GLYPHS
    con.advance(ok=True, duration=1.0)
    assert '█' in con.lines()[0]
    assert '░' in con.lines()[0]


def test_interactive_redraw_sequences():
    stream = FakeTty()
    con = oc.Console(total=4, stream=stream, width=70)
    assert con.interactive
    con.start()
    con.advance(ok=True, duration=1.0)
    con.show('over the bar')
    con.finish(['end'])
    out = stream.getvalue()
    assert '\x1b[1A' in out
    assert '\x1b[2K' in out
    assert 'over the bar\n' in out
    assert out.rstrip().endswith('end')


def test_no_color(monkeypatch):
    monkeypatch.setenv('NO_COLOR', '1')
    con = oc.Console(total=4, stream=FakeTty(), width=70)
    assert not con.color
    con.advance(ok=False, duration=1.0)
    assert '\x1b[3' not in con.lines()[1]


def test_color_enabled_on_tty(monkeypatch):
    monkeypatch.delenv('NO_COLOR', raising=False)
    con = oc.Console(total=4, stream=FakeTty(), width=70)
    assert con.color
    con.advance(ok=False, duration=1.0)
    assert '\x1b[31m' in con.lines()[1]


def test_no_unicode_encode_error_on_strict_stream():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding='ascii', errors='strict', write_through=True)
    con = oc.Console(total=2, stream=stream, interactive=False)
    con.show('Привет')
    con.advance(ok=True, duration=1.0)
    con.finish(['Итог'])


def test_verbose_has_no_bar():
    stream = FakeTty()
    con = oc.Console(total=4, stream=stream, verbose=True)
    assert not con.interactive
    con.start()
    con.advance(ok=True, duration=1.0)
    con.show('line')
    con.finish([])
    out = stream.getvalue()
    assert '\x1b' not in out
    assert out == 'line\n'


def test_fmt_time():
    assert oc.fmt_time(8047) == '2:14:07'
    assert oc.fmt_time(0) == '0:00:00'
    assert oc.fmt_time(None) == '--:--:--'


def test_short_error():
    text = oc.short_error('line one\n' + 'x' * 500)
    assert '\n' not in text
    assert len(text) <= 110


def test_verbose_env(monkeypatch):
    monkeypatch.delenv('ONYX_VERBOSE', raising=False)
    assert not oc.is_verbose(False)
    assert oc.is_verbose(True)
    monkeypatch.setenv('ONYX_VERBOSE', '1')
    assert oc.is_verbose(False)


def test_extractor_import_has_no_tpfmodel_warning():
    code = (
        "import sys, importlib.util\n"
        "sys.path.insert(0, %r)\n"
        "spec = importlib.util.spec_from_file_location('onyx_extractor_probe', %r)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
    ) % (str(ROOT), str(ROOT / '1_data_extractor.py'))
    env = dict(os.environ)
    env.pop('PYTHONWARNINGS', None)
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=str(ROOT), env=env, encoding='utf-8', errors='replace')
    assert res.returncode == 0, res.stderr
    assert 'tpfmodel' not in res.stderr
    assert 'tpfmodel' not in res.stdout


def test_setup_warnings_sets_env_for_children(monkeypatch):
    monkeypatch.setenv('PYTHONWARNINGS', 'ignore::DeprecationWarning')
    oc.setup_warnings()
    value = os.environ['PYTHONWARNINGS']
    assert 'ignore::DeprecationWarning' in value
    assert oc.TPF_ENV_FILTER in value
    oc.setup_warnings()
    assert os.environ['PYTHONWARNINGS'].count(oc.TPF_ENV_FILTER) == 1
