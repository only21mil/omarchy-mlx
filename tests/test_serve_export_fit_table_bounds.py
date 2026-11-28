"""Atomic output, target-type refusal, and path-validation tests for export_fit_table."""

from __future__ import annotations

import errno
import io
import os
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import export_fit_table  # noqa: E402
from mlx_omarchy_serve.export_fit_table import _write_text_atomic, main  # noqa: E402


def _assert_single_error(rc: int, out: str, err: str) -> str:
    lines = err.splitlines()
    assert rc == 1
    assert out == ""
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert "Traceback" not in err
    return lines[0]


def _leftovers(directory: Path, expected: set[str]) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name not in expected)


@pytest.mark.parametrize(
    "bad_out",
    [
        "",
        "bad\x00.json",
        "bad\n\x00.json",
        "bad\ud800.json",
        "/",
        "//",
        ".",
        "..",
        "./",
        "../",
        "out.json/",
        "sub/out.json/",
        "sub/.",
        "sub/..",
        "/.",
        "/..",
    ],
)
def test_main_rejects_invalid_raw_out_paths(tmp_path, monkeypatch, capsys, bad_out):
    monkeypatch.chdir(tmp_path)
    open_calls: list[str] = []
    real_open = os.open

    def spy_open(path, flags, *args, **kwargs):
        open_calls.append(os.fsdecode(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)
    with pytest.raises(OSError) as exc_info:
        _write_text_atomic(bad_out, "{}\n")
    assert exc_info.value.errno == errno.EINVAL
    assert open_calls == []

    rc = main(["-o", bad_out])
    cap = capsys.readouterr()
    line = _assert_single_error(rc, cap.out, cap.err)
    assert "\x00" not in cap.err
    assert list(tmp_path.iterdir()) == []
    if bad_out == "":
        assert "empty" in line.lower()


def test_write_text_atomic_rejects_existing_dir_with_trailing_slash_before_open(
    tmp_path, monkeypatch
):
    subdir = tmp_path / "adir"
    subdir.mkdir()

    def forbid_open(*_args, **_kwargs):
        raise AssertionError("os.open must not be called for invalid raw path")

    monkeypatch.setattr(export_fit_table.os, "open", forbid_open)
    for raw in (f"{subdir}/", f"{subdir}/.", f"{subdir}/.."):
        with pytest.raises(OSError) as exc_info:
            _write_text_atomic(raw, "{}\n")
        assert exc_info.value.errno == errno.EINVAL


def test_write_text_atomic_refuses_existing_directory_with_eisdir(
    tmp_path, monkeypatch, capsys
):
    subdir = tmp_path / "adir"
    subdir.mkdir()

    real_open = os.open
    opened_creates: list[str] = []

    def spy_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            opened_creates.append(os.fsdecode(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)
    with pytest.raises(IsADirectoryError) as exc_info:
        _write_text_atomic(subdir, "{}\n")
    assert exc_info.value.errno == errno.EISDIR
    assert str(subdir) in str(exc_info.value)
    assert opened_creates == []

    rc = main(["-o", str(subdir)])
    cap = capsys.readouterr()
    line = _assert_single_error(rc, cap.out, cap.err)
    assert os.strerror(errno.EISDIR) in line
    assert list(subdir.iterdir()) == []


@pytest.mark.parametrize("suffix", ["/", "/.", "/.."])
def test_main_rejects_trailing_dir_spellings_without_overwriting_existing_file(
    tmp_path, capsys, suffix
):
    existing = tmp_path / "report.json"
    existing.write_text("original\n", encoding="utf-8")
    old_ino = os.lstat(existing).st_ino

    rc = main(["-o", f"{existing}{suffix}"])
    cap = capsys.readouterr()
    _assert_single_error(rc, cap.out, cap.err)
    assert existing.read_text(encoding="utf-8") == "original\n"
    assert os.lstat(existing).st_ino == old_ino
    assert _leftovers(tmp_path, {"report.json"}) == []


@pytest.mark.parametrize(
    ("sep", "escaped"),
    [
        pytest.param("\n", r"\n", id="lf"),
        pytest.param("\r", r"\r", id="cr"),
        pytest.param("\u2028", r"\u2028", id="ls"),
    ],
)
def test_main_escapes_line_breaks_in_invalid_raw_out_paths(
    tmp_path, capsys, sep, escaped
):
    raw = str(tmp_path / f"bad{sep}name") + "/"
    rc = main(["-o", raw])
    cap = capsys.readouterr()
    line = _assert_single_error(rc, cap.out, cap.err)
    assert f"bad{escaped}name/" in line


def test_main_refuses_file_as_parent_directory(tmp_path, capsys):
    blocker = tmp_path / "regular"
    blocker.write_text("keep\n", encoding="utf-8")

    rc = main(["-o", str(blocker / "out.json")])
    cap = capsys.readouterr()
    _assert_single_error(rc, cap.out, cap.err)
    assert blocker.read_text(encoding="utf-8") == "keep\n"
    assert _leftovers(tmp_path, {"regular"}) == []


@pytest.mark.parametrize("kind", ["regular_target", "dangling", "directory_target", "self_loop"])
def test_main_refuses_symlinks_before_staging(tmp_path, monkeypatch, capsys, kind):
    link = tmp_path / "out.json"
    expected = {"out.json"}
    victim = tmp_path / "victim.txt"
    subdir = tmp_path / "subdir"
    if kind == "regular_target":
        victim.write_text("keep\n", encoding="utf-8")
        link.symlink_to(victim)
        expected.add("victim.txt")
    elif kind == "dangling":
        link.symlink_to(tmp_path / "never_created.json")
    elif kind == "directory_target":
        subdir.mkdir()
        link.symlink_to(subdir)
        expected.add("subdir")
    else:
        link.symlink_to(link)

    real_open = os.open
    opened_creates: list[str] = []

    def spy_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            opened_creates.append(os.fsdecode(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)

    rc = main(["-o", str(link)])
    cap = capsys.readouterr()
    line = _assert_single_error(rc, cap.out, cap.err)
    assert "symlink" in line.lower()
    assert str(link) in line
    assert opened_creates == []
    assert link.is_symlink()
    if kind == "regular_target":
        assert victim.read_text(encoding="utf-8") == "keep\n"
    elif kind == "dangling":
        assert not (tmp_path / "never_created.json").exists()
    assert _leftovers(tmp_path, expected) == []


def test_main_refuses_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "out.fifo"
    os.mkfifo(fifo)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_omarchy_serve.export_fit_table", "-o", str(fifo)],
        cwd=REPO_ROOT / "serve",
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    line = _assert_single_error(proc.returncode, proc.stdout, proc.stderr)
    assert "non-regular" in line.lower()
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)
    assert _leftovers(tmp_path, {"out.fifo"}) == []


def test_main_refuses_fifo_as_parent_without_blocking(tmp_path):
    fifo = tmp_path / "parent.fifo"
    os.mkfifo(fifo)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_omarchy_serve.export_fit_table",
            "-o",
            str(fifo / "out.json"),
        ],
        cwd=REPO_ROOT / "serve",
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    _assert_single_error(proc.returncode, proc.stdout, proc.stderr)
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)
    assert _leftovers(tmp_path, {"parent.fifo"}) == []


def test_main_refuses_unix_socket(capsys):
    # Use a short /tmp dir to stay well under the 108-byte AF_UNIX path limit.
    with tempfile.TemporaryDirectory(prefix="eft-sock-", dir="/tmp") as td:
        d = Path(td)
        sock_path = d / "out.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(sock_path))
            rc = main(["-o", str(sock_path)])
            cap = capsys.readouterr()
            line = _assert_single_error(rc, cap.out, cap.err)
            assert "non-regular" in line.lower()
            assert stat.S_ISSOCK(os.lstat(sock_path).st_mode)
            assert _leftovers(d, {"out.sock"}) == []
        finally:
            srv.close()


def test_main_refuses_character_device(tmp_path, monkeypatch, capsys):
    # Stage under tmp_path and return os.stat(os.devnull) from lstat so even a
    # regression on a privileged runner can never overwrite live /dev nodes.
    dev_st = os.stat(os.devnull)
    assert stat.S_ISCHR(dev_st.st_mode)
    target = tmp_path / "chrdev.json"
    real_lstat = os.lstat

    def fake_lstat(name, *, dir_fd=None):
        if name == target.name:
            return dev_st
        return real_lstat(name, dir_fd=dir_fd)

    monkeypatch.setattr(export_fit_table.os, "lstat", fake_lstat)

    with pytest.raises(OSError) as exc_info:
        _write_text_atomic(target, "{}\n")
    assert exc_info.value.errno == errno.EINVAL
    assert not isinstance(exc_info.value, PermissionError)
    assert list(tmp_path.iterdir()) == []

    rc = main(["-o", str(target)])
    cap = capsys.readouterr()
    line = _assert_single_error(rc, cap.out, cap.err)
    assert "non-regular" in line.lower()
    assert str(target) in line
    assert list(tmp_path.iterdir()) == []


def test_main_replaces_existing_regular_file_atomically(tmp_path, capsys):
    assert main([]) == 0
    expected_bytes = capsys.readouterr().out.encode("utf-8")

    out_path = tmp_path / "fit-table.json"
    out_path.write_text('{"stale": true}\n', encoding="utf-8")
    old_ino = os.lstat(out_path).st_ino

    rc = main(["-o", str(out_path)])
    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out == ""
    assert cap.err == ""
    assert out_path.read_bytes() == expected_bytes
    assert os.lstat(out_path).st_ino != old_ino
    assert _leftovers(tmp_path, {"fit-table.json"}) == []


def test_main_writes_relative_name_symlinked_parent_and_long_filename(
    tmp_path, monkeypatch, capsys
):
    assert main([]) == 0
    expected_bytes = capsys.readouterr().out.encode("utf-8")

    # Relative filename in cwd (exercises parent or ".")
    rel_dir = tmp_path / "rel"
    rel_dir.mkdir()
    monkeypatch.chdir(rel_dir)
    assert main(["-o", "table.json"]) == 0
    assert capsys.readouterr() == ("", "")
    assert (rel_dir / "table.json").read_bytes() == expected_bytes
    assert _leftovers(rel_dir, {"table.json"}) == []

    # Dot-prefixed filenames are valid regular files, not "." or ".."
    for dot_name in (".fit-table.json", "..fit-table.json"):
        assert main(["-o", str(rel_dir / dot_name)]) == 0
        assert capsys.readouterr() == ("", "")
        assert (rel_dir / dot_name).read_bytes() == expected_bytes

    # Symlinked parent directory is allowed; final regular file is staged inside it
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    dir_link = tmp_path / "dir_link"
    dir_link.symlink_to(real_dir)
    assert main(["-o", str(dir_link / "fit.json")]) == 0
    assert capsys.readouterr() == ("", "")
    assert (real_dir / "fit.json").read_bytes() == expected_bytes
    assert _leftovers(real_dir, {"fit.json"}) == []

    # 250-byte filename succeeds because sibling temp name has fixed short length
    long_name = "a" * 245 + ".json"
    long_dir = tmp_path / "long"
    long_dir.mkdir()
    assert main(["-o", str(long_dir / long_name)]) == 0
    assert capsys.readouterr() == ("", "")
    assert (long_dir / long_name).read_bytes() == expected_bytes
    assert _leftovers(long_dir, {long_name}) == []


@pytest.mark.parametrize(("umask_val", "expected_mode"), [(0o022, 0o644), (0o077, 0o600)])
def test_main_respects_process_umask(tmp_path, umask_val, expected_mode):
    out_path = tmp_path / f"mode-{umask_val:03o}.json"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    snippet = (
        f"import os, sys; os.umask({umask_val:#o}); "
        "from mlx_omarchy_serve.export_fit_table import main; "
        "raise SystemExit(main(sys.argv[1:]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet, "-o", str(out_path)],
        cwd=REPO_ROOT / "serve",
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""
    assert stat.S_IMODE(os.lstat(out_path).st_mode) == expected_mode


def test_main_fsize_failure_preserves_existing_file_and_cleans_temp(tmp_path):
    existing = tmp_path / "fit-table.json"
    existing.write_text('{"preserved": true}\n', encoding="utf-8")
    old_ino = os.lstat(existing).st_ino

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    snippet = (
        "import resource, signal, sys; "
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN); "
        "resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64)); "
        "from mlx_omarchy_serve.export_fit_table import main; "
        "raise SystemExit(main(sys.argv[1:]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet, "-o", str(existing)],
        cwd=REPO_ROOT / "serve",
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    line = _assert_single_error(proc.returncode, proc.stdout, proc.stderr)
    assert os.strerror(errno.EFBIG) in line
    assert str(existing) in line
    assert existing.read_text(encoding="utf-8") == '{"preserved": true}\n'
    assert os.lstat(existing).st_ino == old_ino
    assert _leftovers(tmp_path, {"fit-table.json"}) == []


def test_write_text_atomic_handles_short_writes_and_checks_syscalls(
    tmp_path, monkeypatch
):
    target = tmp_path / "out.json"
    payload = "hello atomic world\n" * 4
    expected_bytes = payload.encode("utf-8")

    real_open = os.open
    real_lstat = os.lstat
    real_write = os.write
    real_fsync = os.fsync
    real_close = os.close
    real_replace = os.replace

    events: list[tuple] = []
    dir_fds: list[int] = []
    tmp_fds: list[int] = []
    write_calls = 0

    def spy_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is None:
            dir_fds.append(fd)
            events.append(("open_dir", flags))
        else:
            tmp_fds.append(fd)
            events.append(("open_tmp", flags, args[0], kwargs["dir_fd"]))
        return fd

    def spy_lstat(path, *, dir_fd=None):
        events.append(("lstat", path, dir_fd))
        return real_lstat(path, dir_fd=dir_fd)

    def spy_write(fd, buf):
        nonlocal write_calls
        write_calls += 1
        chunk = bytes(buf[:7])
        return real_write(fd, chunk)

    def spy_fsync(fd):
        events.append(("fsync", fd))
        return real_fsync(fd)

    def spy_close(fd):
        events.append(("close", fd))
        return real_close(fd)

    def spy_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        events.append(("replace", src, dst, src_dir_fd, dst_dir_fd))
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)
    monkeypatch.setattr(export_fit_table.os, "lstat", spy_lstat)
    monkeypatch.setattr(export_fit_table.os, "write", spy_write)
    monkeypatch.setattr(export_fit_table.os, "fsync", spy_fsync)
    monkeypatch.setattr(export_fit_table.os, "close", spy_close)
    monkeypatch.setattr(export_fit_table.os, "replace", spy_replace)

    _write_text_atomic(target, payload)

    assert target.read_bytes() == expected_bytes
    assert write_calls > 1
    assert len(dir_fds) == 1
    assert len(tmp_fds) == 1
    dfd, tfd = dir_fds[0], tmp_fds[0]

    expected_dir_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    expected_tmp_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    assert events[0] == ("open_dir", expected_dir_flags)
    assert events[1] == ("lstat", "out.json", dfd)
    assert events[2] == ("open_tmp", expected_tmp_flags, 0o666, dfd)
    assert events[3] == ("fsync", tfd)
    assert events[4] == ("close", tfd)
    assert events[5][0] == "replace"
    assert events[5][2:] == ("out.json", dfd, dfd)
    assert events[6] == ("close", dfd)
    assert _leftovers(tmp_path, {"out.json"}) == []


@pytest.mark.parametrize(
    "fail_point",
    [
        "zero_write",
        "write_enospc",
        "write_interrupt",
        "fsync_eio",
        "close_eio",
        "replace_eacces",
        "fstat_non_regular",
    ],
)
def test_write_text_atomic_cleanup_on_errors(tmp_path, monkeypatch, fail_point):
    target = tmp_path / "fit.json"
    target.write_text("keep\n", encoding="utf-8")
    old_ino = os.lstat(target).st_ino

    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    real_unlink = os.unlink

    opened_fds: list[int] = []
    closed_fds: list[int] = []
    unlinked_calls: list[tuple[str, int | None]] = []
    replace_called = False

    def spy_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened_fds.append(fd)
        return fd

    def spy_close(fd):
        closed_fds.append(fd)
        real_close(fd)
        if fail_point == "close_eio" and len(closed_fds) == 1:
            raise OSError(errno.EIO, "delayed close I/O error")

    def spy_unlink(path, *, dir_fd=None):
        unlinked_calls.append((os.fsdecode(path), dir_fd))
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)
    monkeypatch.setattr(export_fit_table.os, "close", spy_close)
    monkeypatch.setattr(export_fit_table.os, "unlink", spy_unlink)

    if fail_point == "zero_write":
        monkeypatch.setattr(export_fit_table.os, "write", lambda _fd, _b: 0)
    elif fail_point == "write_enospc":
        def boom_write(_fd, _b):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        monkeypatch.setattr(export_fit_table.os, "write", boom_write)
    elif fail_point == "write_interrupt":
        def interrupt_write(_fd, _b):
            raise KeyboardInterrupt("stop")
        monkeypatch.setattr(export_fit_table.os, "write", interrupt_write)
    elif fail_point == "fsync_eio":
        def boom_fsync(_fd):
            raise OSError(errno.EIO, "fsync failed")
        monkeypatch.setattr(export_fit_table.os, "fsync", boom_fsync)
    elif fail_point == "fstat_non_regular":
        def fake_fstat(fd):
            st = real_fstat(fd)
            return os.stat_result((stat.S_IFIFO | 0o600,) + tuple(st)[1:])
        monkeypatch.setattr(export_fit_table.os, "fstat", fake_fstat)

    real_replace = os.replace

    def spy_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal replace_called
        replace_called = True
        if fail_point == "replace_eacces":
            raise PermissionError(errno.EACCES, "Permission denied", src, None, dst)
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(export_fit_table.os, "replace", spy_replace)

    expected_exc = KeyboardInterrupt if fail_point == "write_interrupt" else OSError
    with pytest.raises(expected_exc) as exc_info:
        _write_text_atomic(target, '{"new": 1}\n')

    if expected_exc is OSError:
        assert str(target) in str(exc_info.value)
        assert ".export_fit_table." not in str(exc_info.value)
    if fail_point != "replace_eacces":
        assert not replace_called
    assert target.read_text(encoding="utf-8") == "keep\n"
    assert os.lstat(target).st_ino == old_ino
    assert _leftovers(tmp_path, {"fit.json"}) == []
    assert len(unlinked_calls) == 1
    assert unlinked_calls[0][1] == opened_fds[0]
    assert sorted(closed_fds) == sorted(opened_fds)
    assert len(closed_fds) == len(opened_fds)


def test_write_text_atomic_retries_temp_collision_and_exhausts(tmp_path, monkeypatch):
    target = tmp_path / "fit.json"
    real_open = os.open
    collisions = 0

    def flaky_open(path, flags, *args, **kwargs):
        nonlocal collisions
        if (flags & os.O_CREAT) and collisions < 2:
            collisions += 1
            raise FileExistsError(errno.EEXIST, "File exists", path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(export_fit_table.os, "open", flaky_open)
    _write_text_atomic(target, '{"ok": true}\n')
    assert collisions == 2
    assert target.read_text(encoding="utf-8") == '{"ok": true}\n'

    # Exhaust all attempts
    dir_fds: list[int] = []
    closed_fds: list[int] = []
    real_close = os.close

    def always_collide(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            raise FileExistsError(errno.EEXIST, "File exists", path)
        fd = real_open(path, flags, *args, **kwargs)
        dir_fds.append(fd)
        return fd

    def spy_close(fd):
        closed_fds.append(fd)
        real_close(fd)

    monkeypatch.setattr(export_fit_table.os, "open", always_collide)
    monkeypatch.setattr(export_fit_table.os, "close", spy_close)
    with pytest.raises(FileExistsError) as exc_info:
        _write_text_atomic(target, '{"again": true}\n')
    assert str(target) in str(exc_info.value)
    assert closed_fds == dir_fds


def test_write_text_atomic_propagates_non_enoent_lstat_error(tmp_path, monkeypatch):
    target = tmp_path / "fit.json"
    real_open = os.open
    real_close = os.close
    dir_fds: list[int] = []
    closed_fds: list[int] = []

    def spy_open(path, flags, *args, **kwargs):
        assert not (flags & os.O_CREAT)
        fd = real_open(path, flags, *args, **kwargs)
        dir_fds.append(fd)
        return fd

    def spy_close(fd):
        closed_fds.append(fd)
        real_close(fd)

    def boom_lstat(name, *, dir_fd=None):
        raise PermissionError(errno.EACCES, "Permission denied", name)

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)
    monkeypatch.setattr(export_fit_table.os, "close", spy_close)
    monkeypatch.setattr(export_fit_table.os, "lstat", boom_lstat)

    with pytest.raises(PermissionError) as exc_info:
        _write_text_atomic(target, "{}\n")
    assert str(target) in str(exc_info.value)
    assert closed_fds == dir_fds


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory write permissions")
def test_main_reports_error_on_readonly_parent_dir(tmp_path, capsys):
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_dir.chmod(0o555)
    try:
        target = ro_dir / "fit.json"
        rc = main(["-o", str(target)])
        cap = capsys.readouterr()
        line = _assert_single_error(rc, cap.out, cap.err)
        assert str(target) in line
        assert ".export_fit_table." not in line
        assert list(ro_dir.iterdir()) == []
    finally:
        ro_dir.chmod(0o755)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("to_file", [False, True])
def test_main_rejects_non_finite_json_values(
    tmp_path, monkeypatch, capsys, bad_val, to_file
):
    monkeypatch.setattr(
        export_fit_table, "build_fit_table", lambda _cat=None: {"bad": bad_val}
    )
    out_path = tmp_path / "fit.json"
    argv = ["-o", str(out_path)] if to_file else []
    rc = main(argv)
    cap = capsys.readouterr()
    line = _assert_single_error(rc, cap.out, cap.err)
    assert "non-json-compliant" in line.lower()
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stderr_kind", ["closed_stream", "broken_pipe"])
def test_main_handles_unwritable_stderr_stream_in_process(
    monkeypatch, capsys, stderr_kind
):
    if stderr_kind == "closed_stream":
        closed = io.StringIO()
        closed.close()
        monkeypatch.setattr(sys, "stderr", closed)
    else:
        class _BrokenStderr:
            def write(self, _text: str) -> int:
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")

        monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    rc = main(["-o", ""])
    assert rc == 1
    assert capsys.readouterr().out == ""


def test_main_module_broken_stderr_pipe_exits_1():
    r_fd, w_fd = os.pipe()
    os.close(r_fd)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    env.pop("PYTHONUNBUFFERED", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mlx_omarchy_serve.export_fit_table", "-o", ""],
            cwd=REPO_ROOT / "serve",
            env=env,
            stdout=subprocess.PIPE,
            stderr=w_fd,
            text=True,
            timeout=15,
        )
    finally:
        os.close(w_fd)
    assert proc.returncode == 1
    assert proc.stdout == ""


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="/dev/full not available")
def test_main_module_full_stderr_exits_1():
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    env.pop("PYTHONUNBUFFERED", None)
    with open("/dev/full", "wb") as full:
        proc = subprocess.run(
            [sys.executable, "-m", "mlx_omarchy_serve.export_fit_table", "-o", ""],
            cwd=REPO_ROOT / "serve",
            env=env,
            stdout=subprocess.PIPE,
            stderr=full,
            text=True,
            timeout=15,
        )
    assert proc.returncode == 1
    assert proc.stdout == ""


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="/bin/sh not available")
@pytest.mark.parametrize("case", ["empty", "symlink", "trailing_slash"])
def test_main_closed_stderr_keeps_stdout_empty_on_new_refusals(tmp_path, case):
    if case == "empty":
        arg = ""
    elif case == "symlink":
        victim = tmp_path / "victim.txt"
        victim.write_text("keep\n", encoding="utf-8")
        link = tmp_path / "out.json"
        link.symlink_to(victim)
        arg = str(link)
    else:
        existing = tmp_path / "out.json"
        existing.write_text("keep\n", encoding="utf-8")
        arg = f"{existing}/"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    proc = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'exec "$0" "$@" 2>&-',
            sys.executable,
            "-m",
            "mlx_omarchy_serve.export_fit_table",
            "-o",
            arg,
        ],
        cwd=REPO_ROOT / "serve",
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    if case == "symlink":
        assert (tmp_path / "victim.txt").read_text(encoding="utf-8") == "keep\n"
    elif case == "trailing_slash":
        assert (tmp_path / "out.json").read_text(encoding="utf-8") == "keep\n"


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="/bin/sh not available")
def test_main_closed_stdout_reports_symlink_refusal_on_stderr(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "out.json"
    link.symlink_to(victim)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "serve"))
    proc = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'exec "$0" "$@" >&-',
            sys.executable,
            "-m",
            "mlx_omarchy_serve.export_fit_table",
            "-o",
            str(link),
        ],
        cwd=REPO_ROOT / "serve",
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    line = _assert_single_error(proc.returncode, "", proc.stderr)
    assert "symlink" in line.lower()
    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_write_text_atomic_preserves_original_error_when_cleanup_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "fit.json"
    real_open = os.open
    real_close = os.close
    real_unlink = os.unlink
    opened_fds: list[int] = []
    closed_fds: list[int] = []
    tmp_names: list[str] = []

    def spy_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened_fds.append(fd)
        if kwargs.get("dir_fd") is not None:
            tmp_names.append(os.fsdecode(path))
        return fd

    def spy_close(fd):
        closed_fds.append(fd)
        real_close(fd)
        if len(closed_fds) == 1:
            raise OSError(errno.EBADF, "secondary close failure")

    def boom_write(_fd, _buf):
        raise OSError("custom errno-less write error")

    def boom_unlink(_path, *, dir_fd=None):
        raise PermissionError(errno.EPERM, "secondary unlink failure")

    monkeypatch.setattr(export_fit_table.os, "open", spy_open)
    monkeypatch.setattr(export_fit_table.os, "close", spy_close)
    monkeypatch.setattr(export_fit_table.os, "write", boom_write)
    monkeypatch.setattr(export_fit_table.os, "unlink", boom_unlink)

    try:
        with pytest.raises(OSError) as exc_info:
            _write_text_atomic(target, "{}\n")
        assert "custom errno-less write error" in str(exc_info.value)
        assert str(target) in str(exc_info.value)
        assert sorted(closed_fds) == sorted(opened_fds)
    finally:
        for name in tmp_names:
            p = tmp_path / name
            if p.exists():
                real_unlink(p)


def test_write_text_atomic_accepts_pathlike_and_raw_bytes(tmp_path):
    class _CustomPath:
        def __init__(self, value: str) -> None:
            self._value = value

        def __fspath__(self) -> str:
            return self._value

    p1 = tmp_path / "pathlike.json"
    _write_text_atomic(_CustomPath(str(p1)), '{"ok": 1}\n')
    assert p1.read_text(encoding="utf-8") == '{"ok": 1}\n'

    utf8_bytes_path = os.path.join(os.fsencode(tmp_path), b"raw-bytes.json")
    _write_text_atomic(utf8_bytes_path, '{"ok": 2}\n')
    with open(utf8_bytes_path, "rb") as fh:
        assert fh.read() == b'{"ok": 2}\n'


def test_write_text_atomic_accepts_non_utf8_raw_bytes(tmp_path):
    probe = os.path.join(os.fsencode(tmp_path), b"probe-\xff.tmp")
    try:
        with open(probe, "wb") as fh:
            fh.write(b"x")
    except OSError as exc:
        pytest.skip(f"filesystem refused non-UTF-8 file name: {exc}")
    os.unlink(probe)

    raw_bytes_path = os.path.join(os.fsencode(tmp_path), b"raw-\xff.json")
    _write_text_atomic(raw_bytes_path, '{"ok": 3}\n')
    with open(raw_bytes_path, "rb") as fh:
        assert fh.read() == b'{"ok": 3}\n'


def test_write_text_atomic_cleans_up_if_target_becomes_directory_before_replace(
    tmp_path, monkeypatch
):
    target = tmp_path / "raced.json"
    real_replace = os.replace

    def race_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        target.mkdir()
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(export_fit_table.os, "replace", race_replace)
    with pytest.raises(IsADirectoryError) as exc_info:
        _write_text_atomic(target, '{"ok": true}\n')
    assert str(target) in str(exc_info.value)
    assert ".export_fit_table." not in str(exc_info.value)
    assert target.is_dir()
    assert _leftovers(tmp_path, {"raced.json"}) == []
