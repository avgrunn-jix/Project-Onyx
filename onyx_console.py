import logging
import os
import shutil
import sys
import threading
import time
import warnings

TPF_ENV_FILTER = 'ignore::UserWarning:lightkurve.prf'
TPF_MESSAGE_REGEX = r'.*tpfmodel submodule is not available.*'

UNICODE_GLYPHS = {'fill': '█', 'empty': '░', 'spin': '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏', 'sep': '·', 'cut': '…'}
ASCII_GLYPHS = {'fill': '#', 'empty': '-', 'spin': '|/-\\', 'sep': '|', 'cut': '~'}

_ansi_enabled = False


def setup_warnings():
    current = os.environ.get('PYTHONWARNINGS', '')
    items = [x for x in current.split(',') if x]
    if TPF_ENV_FILTER not in items:
        items.append(TPF_ENV_FILTER)
    os.environ['PYTHONWARNINGS'] = ','.join(items)
    warnings.filterwarnings('ignore', message=TPF_MESSAGE_REGEX, category=UserWarning)


def is_verbose(flag=False):
    if flag:
        return True
    return os.environ.get('ONYX_VERBOSE', '').strip().lower() in ('1', 'true', 'yes', 'on')


def set_verbose_env(flag):
    if flag:
        os.environ['ONYX_VERBOSE'] = '1'


def enable_ansi():
    global _ansi_enabled
    if _ansi_enabled:
        return
    _ansi_enabled = True
    if os.name == 'nt':
        try:
            os.system('')
        except Exception:
            pass


def fmt_time(seconds):
    if seconds is None:
        return '--:--:--'
    s = max(0, int(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def stream_is_tty(stream):
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def pick_glyphs(stream):
    enc = getattr(stream, 'encoding', None) or 'ascii'
    probe = UNICODE_GLYPHS['fill'] + UNICODE_GLYPHS['empty'] + UNICODE_GLYPHS['spin'] + UNICODE_GLYPHS['sep'] + UNICODE_GLYPHS['cut']
    try:
        probe.encode(enc)
        return UNICODE_GLYPHS
    except (UnicodeEncodeError, LookupError):
        return ASCII_GLYPHS


def short_error(err, limit=110):
    text = ' '.join(str(err).split())
    if len(text) > limit:
        text = text[:limit - 3] + '...'
    return text


def _clip(text, avail, cut):
    if avail <= 0:
        return ''
    if len(text) <= avail:
        return text
    return text[:avail - 1] + cut


def render_lines(done, total, ok, failed, status, elapsed, eta, spin_idx, width, glyphs, color=False, unit='stars', show_ok=True, show_eta=True):
    width = max(1, int(width))
    pct = 100 if total <= 0 else min(100, int(done * 100 / total))
    tail = f"  {pct:3d}%  {done}/{total} {unit}"
    bar_w = min(30, width - 1 - len(tail))
    if bar_w >= 5:
        filled = bar_w * pct // 100
        line1 = ' ' + glyphs['fill'] * filled + glyphs['empty'] * (bar_w - filled) + tail
    else:
        line1 = f"{pct:3d}% {done}/{total}"
    line1 = line1[:width]

    spin = glyphs['spin'][spin_idx % len(glyphs['spin'])]
    sep = f"  {glyphs['sep']}  "
    ok_txt = f"ok {ok}"
    fail_txt = f"failed {failed}"
    counts = f"{ok_txt}  {fail_txt}" if show_ok else None
    tails = []
    if counts:
        tails.append(counts)
    tails.append(f"elapsed {fmt_time(elapsed)}")
    if show_eta:
        tails.append(f"ETA {fmt_time(eta)}")
    chosen = None
    for drop in range(len(tails) + 1):
        kept = tails[:len(tails) - drop]
        rest = ''.join(sep + t for t in kept)
        avail = width - len(f" {spin} ") - len(rest)
        if avail >= min(len(status), 12) or drop == len(tails):
            chosen = (kept, avail)
            break
    kept, avail = chosen
    if avail < 0:
        kept = []
        avail = width - len(f" {spin} ")
    status_txt = _clip(status, avail, glyphs['cut'])
    plain = f" {spin} " + status_txt + ''.join(sep + t for t in kept)
    plain = plain[:width]
    if color and len(plain) == len(f" {spin} " + status_txt + ''.join(sep + t for t in kept)):
        line2 = plain
        if counts and counts in plain:
            colored = ('\x1b[32m' + ok_txt + '\x1b[0m') + '  ' + (('\x1b[31m' + fail_txt + '\x1b[0m') if failed > 0 else fail_txt)
            line2 = plain.replace(counts, colored, 1)
        return line1, line2
    return line1, plain


class Console:
    def __init__(self, total=0, unit='stars', workers=1, stream=None, verbose=False, logger=None, color=None, glyphs=None, width=None, interactive=None, clock=time.time, show_ok=True, show_eta=True):
        self.stream = stream if stream is not None else sys.stdout
        self.total = int(total)
        self.unit = unit
        self.workers = max(1, int(workers))
        self.verbose = bool(verbose)
        self.logger = logger
        self.clock = clock
        self.show_ok = show_ok
        self.show_eta = show_eta
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.durations = []
        self.status = ''
        self.t0 = clock()
        tty = stream_is_tty(self.stream)
        self.interactive = (tty and not self.verbose) if interactive is None else bool(interactive)
        if color is None:
            color = tty and 'NO_COLOR' not in os.environ
        self.color = bool(color) and self.interactive
        self.glyphs = glyphs if glyphs is not None else pick_glyphs(self.stream)
        self._width = width
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._spin = 0
        self._drawn = False
        self._active = False
        self._paused = False
        self._last_report = 0
        if self.interactive:
            enable_ansi()

    def width(self):
        if self._width is not None:
            return self._width
        return max(1, shutil.get_terminal_size((80, 24)).columns - 1)

    def _write(self, text):
        try:
            self.stream.write(text)
            self.stream.flush()
        except UnicodeEncodeError:
            enc = getattr(self.stream, 'encoding', None) or 'ascii'
            try:
                self.stream.write(text.encode(enc, errors='replace').decode(enc, errors='replace'))
                self.stream.flush()
            except Exception:
                pass
        except (OSError, ValueError):
            pass

    def eta(self):
        if self.total > 0 and self.done >= self.total:
            return 0
        if not self.durations:
            return None
        avg = sum(self.durations) / len(self.durations)
        return int(max(0, self.total - self.done) * avg / self.workers)

    def elapsed(self):
        return self.clock() - self.t0

    def lines(self):
        return render_lines(self.done, self.total, self.ok, self.failed, self.status, self.elapsed(), self.eta(), self._spin, self.width(), self.glyphs, color=self.color, unit=self.unit, show_ok=self.show_ok, show_eta=self.show_eta)

    def _draw(self):
        if not (self.interactive and self._active and not self._paused):
            return
        l1, l2 = self.lines()
        prefix = '\x1b[1A\r' if self._drawn else ''
        self._write(prefix + l1 + '\x1b[K\n' + '\r' + l2 + '\x1b[K')
        self._drawn = True

    def _erase(self):
        if self._drawn:
            self._write('\r\x1b[2K\x1b[1A\r\x1b[2K')
            self._drawn = False

    def _tick(self):
        while not self._stop.wait(1.0):
            with self._lock:
                self._spin += 1
                self._draw()

    def start(self):
        with self._lock:
            self.t0 = self.clock()
            self._active = True
            if self.interactive:
                self._draw()
                self._stop.clear()
                self._thread = threading.Thread(target=self._tick, daemon=True)
                self._thread.start()

    def set_total(self, total):
        with self._lock:
            self.total = int(total)
            self._draw()

    def set_status(self, text):
        with self._lock:
            self.status = str(text)
            self._draw()

    def advance(self, ok=True, duration=None, count=1):
        with self._lock:
            self.done += count
            if ok:
                self.ok += count
            else:
                self.failed += count
            if duration is not None:
                self.durations.append(float(duration))
            if self.verbose:
                return
            if self.interactive:
                self._draw()
            else:
                step = max(1, self.total // 10)
                if self.done % step == 0 or self.done >= self.total:
                    if self.done != self._last_report:
                        self._last_report = self.done
                        self._write(self.plain_progress() + '\n')

    def plain_progress(self):
        pct = 100 if self.total <= 0 else min(100, int(self.done * 100 / self.total))
        parts = f"[{self.done}/{self.total}] {pct}%"
        if self.show_ok:
            parts += f" | ok {self.ok} failed {self.failed}"
        parts += f" | elapsed {fmt_time(self.elapsed())}"
        if self.show_eta:
            parts += f" | ETA {fmt_time(self.eta())}"
        return parts

    def show(self, text):
        with self._lock:
            if self.interactive and self._drawn:
                self._erase()
                self._write(text + '\n')
                self._draw()
            else:
                self._write(text + '\n')

    def note(self, text, level='info'):
        if self.logger is not None:
            getattr(self.logger, level)(text)

    def say(self, text, level='info'):
        self.note(text, level)
        self.show(text)

    def pause(self):
        with self._lock:
            self._erase()
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False
            self._draw()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    def finish(self, lines=()):
        self.stop()
        with self._lock:
            if self.interactive and self._active and not self._paused:
                self._draw()
                self._write('\n')
                self._drawn = False
            elif self.interactive and self._drawn:
                self._erase()
            self._active = False
            for line in lines:
                self._write(line + '\n')


def build_logger(name, log_path, verbose=False, stream=None, level=logging.INFO):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    if verbose:
        console_handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
