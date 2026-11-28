# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Bounded file verification, download staging/locking, and receipt tests for gpu_stt."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import signal
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import gpu_stt, recognition, synthesis  # noqa: E402


PASSING = dict(
    wer_by_subset={
        "test-clean": 0.031,
        "test-other": 0.034,
        "accented": 0.048,
        "mixed_0dB": 0.227,
    },
    empty_rate_by_subset={"silence": 1.0, "noise": 1.0},
    latency_ms={"p50": 457.1, "p95": 468.1},
    cpu_tensor_events=0,
    receipt_source="eval-results.json",
    host="test",
)


def _make_pack():
    content = {
        "config.json": b'{"model_type": "parakeet"}',
        "model.safetensors": b"weights" * 64,
    }
    files = [
        {
            "name": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in content.items()
    ]
    return files, content


class _Loopback:
    """Loopback HTTP server serving finite responses and counting sent body bytes."""

    def __init__(self, case: unittest.TestCase):
        self.routes: dict = {}
        self.sent: dict[str, int] = {}
        self.done: dict[str, threading.Event] = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path
                sent = 0
                try:
                    if path not in outer.routes:
                        self.send_error(404)
                        return
                    status, headers, chunks = outer.routes[path]
                    self.send_response(status)
                    for key, value in headers:
                        self.send_header(key, value)
                    self.end_headers()
                    for chunk in chunks():
                        self.wfile.write(chunk)
                        sent += len(chunk)
                except OSError:
                    self.close_connection = True
                finally:
                    outer.sent[path] = outer.sent.get(path, 0) + sent
                    outer.done.setdefault(path, threading.Event()).set()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        case.addCleanup(thread.join, 10)
        case.addCleanup(self.server.server_close)
        case.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def url(self, path: str) -> str:
        return self.base + path

    def serve(self, path: str, body: bytes, *, status: int = 200,
              length=None, headers=()):
        head = list(headers)
        if length is False:
            head.append(("Connection", "close"))
        else:
            head.append((
                "Content-Length",
                str(len(body) if length is None else length),
            ))
            if length is not None and length != len(body):
                head.append(("Connection", "close"))
        self.routes[path] = (status, head, lambda: [body])

    def wait(self, case: unittest.TestCase, path: str, timeout: float = 30) -> int:
        case.assertTrue(
            self.done.setdefault(path, threading.Event()).wait(timeout),
            f"server never finished {path}",
        )
        return self.sent[path]


class _WatchdogExpired(BaseException):
    """Non-OSError sentinel raised by SIGALRM so caught OSError cannot mask a hang."""


def _run_with_alarm(seconds: int, fn):
    """Run ``fn`` under a POSIX SIGALRM watchdog so a blocking FIFO fails fast."""
    def _timeout(_signum, _frame):
        raise _WatchdogExpired(f"operation blocked for more than {seconds}s")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class PrepareStagingAndCapTests(unittest.TestCase):
    """gpu_stt.prepare() locking, safe staging, and pinned download caps."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.files, self.content = _make_pack()
        self.enterContext(mock.patch.dict(gpu_stt.GPU_STT_MODEL, {"files": self.files}))
        gpu_stt._verified.cache_clear()
        self.addCleanup(gpu_stt._verified.cache_clear)
        self.directory = gpu_stt.model_dir(self.home)
        self.enterContext(mock.patch.dict(os.environ, {"no_proxy": "*", "NO_PROXY": "*"}))
        self.enterContext(mock.patch.object(synthesis, "_URL_TIMEOUT", 10))
        self.http = _Loopback(self)
        self.enterContext(mock.patch.object(
            synthesis, "_URL_TEMPLATE", self.http.base + "/{name}",
        ))

    def _serve_pack(self):
        for name, data in self.content.items():
            self.http.serve("/" + name, data)

    def _entry(self, name: str) -> dict:
        return next(e for e in self.files if e["name"] == name)

    def _parts(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.name for p in self.directory.rglob("*") if p.name.endswith(".part"))

    def _assert_installed(self):
        for name, data in self.content.items():
            dest = self.directory / name
            self.assertFalse(dest.is_symlink(), name)
            self.assertEqual(dest.read_bytes(), data, name)

    def _static_fetch(self, url: str, dest: Path):
        dest.write_bytes(self.content[url.removeprefix(self.http.base + "/")])

    def test_default_download_installs_files_and_caps_each_entry(self):
        self._serve_pack()
        with mock.patch.object(
            synthesis, "_default_fetch", wraps=synthesis._default_fetch,
        ) as fetch:
            result = gpu_stt.prepare(self.home, approve_download=True)
        self.assertTrue(result["verified"])
        self._assert_installed()
        self.assertEqual(self._parts(), [])
        caps = sorted(c.kwargs["max_bytes"] for c in fetch.call_args_list)
        self.assertEqual(caps, sorted(e["bytes"] for e in self.files))
        for name, data in self.content.items():
            self.assertEqual(self.http.wait(self, "/" + name), len(data), name)

    def test_declared_length_over_cap_is_refused_before_writing(self):
        entry = self._entry("config.json")
        block = b"a" * (1 << 20)
        self.http.routes["/config.json"] = (
            200,
            [("Content-Length", str(64 << 20))],
            lambda: [block] * 64,
        )
        self.http.serve("/model.safetensors", self.content["model.safetensors"])
        with self.assertRaisesRegex(
            recognition.RecognitionUnavailable,
            rf"config\.json is {64 << 20} bytes, over the {entry['bytes']}-byte limit",
        ):
            gpu_stt.prepare(self.home, approve_download=True)
        self.assertEqual(self._parts(), [])
        self.assertFalse((self.directory / "config.json").exists())
        self.assertLess(self.http.wait(self, "/config.json"), 32 << 20)

    def test_undeclared_body_over_cap_stops_past_cap(self):
        entry = self._entry("config.json")
        block = b"b" * (1 << 20)
        self.http.routes["/config.json"] = (
            200,
            [("Connection", "close")],
            lambda: [block] * 64,
        )
        with self.assertRaisesRegex(
            recognition.RecognitionUnavailable,
            rf"sent more than {entry['bytes']} bytes",
        ):
            gpu_stt.prepare(self.home, approve_download=True)
        self.assertEqual(self._parts(), [])
        self.assertFalse((self.directory / "config.json").exists())
        self.assertLess(self.http.wait(self, "/config.json"), 32 << 20)

    def test_redirect_body_is_closed_unread(self):
        self.http.serve("/real/config.json", self.content["config.json"])
        self.http.serve("/model.safetensors", self.content["model.safetensors"])
        block = b"r" * (1 << 20)
        self.http.routes["/config.json"] = (
            302,
            [
                ("Location", self.http.url("/real/config.json")),
                ("Content-Length", str(64 << 20)),
            ],
            lambda: [block] * 64,
        )
        result = gpu_stt.prepare(self.home, approve_download=True)
        self.assertTrue(result["verified"])
        self._assert_installed()
        self.assertLess(self.http.wait(self, "/config.json"), 32 << 20)

    def test_non_200_and_short_body_are_refused(self):
        self.http.serve("/config.json", self.content["config.json"], status=206)
        with self.assertRaisesRegex(
            recognition.RecognitionUnavailable, r"/config\.json returned HTTP 206",
        ):
            gpu_stt.prepare(self.home, approve_download=True)
        self.assertEqual(self._parts(), [])
        self.assertFalse((self.directory / "config.json").exists())

        self.http.serve("/config.json", b"short", length=len(self.content["config.json"]))
        with self.assertRaisesRegex(
            recognition.RecognitionUnavailable, r"ended after 5 of",
        ):
            gpu_stt.prepare(self.home, approve_download=True)
        self.assertEqual(self._parts(), [])
        self.assertFalse((self.directory / "config.json").exists())

    def test_part_symlink_not_followed_by_default_or_injected_fetch(self):
        for mode in ("default", "injected"):
            with self.subTest(mode):
                home = self.home / f"part-link-{mode}"
                directory = gpu_stt.model_dir(home)
                directory.mkdir(parents=True)
                victim = home / "victim.txt"
                victim.write_bytes(b"keep me")
                planted = directory / "config.json.part"
                os.symlink(victim, planted)
                seen: list[tuple[Path, bool]] = []

                def injected(url, dest):
                    seen.append((dest, dest.exists() or dest.is_symlink()))
                    self._static_fetch(url, dest)

                if mode == "default":
                    self._serve_pack()
                    result = gpu_stt.prepare(home, approve_download=True)
                else:
                    result = gpu_stt.prepare(home, approve_download=True, fetch=injected)
                    self.assertTrue(all(not existed for _, existed in seen))
                self.assertTrue(result["verified"])
                self.assertEqual(victim.read_bytes(), b"keep me")
                self.assertFalse(planted.exists() or planted.is_symlink())
                self.assertEqual(
                    (directory / "config.json").read_bytes(),
                    self.content["config.json"],
                )

    def test_stale_part_cleaned_and_directory_at_part_refused(self):
        self.directory.mkdir(parents=True)
        stale = self.directory / "config.json.part"
        stale.write_bytes(b"stale partial")
        self._serve_pack()
        result = gpu_stt.prepare(self.home, approve_download=True)
        self.assertTrue(result["verified"])
        self.assertEqual(self._parts(), [])
        self._assert_installed()

        home2 = self.home / "part-dir"
        d2 = gpu_stt.model_dir(home2)
        (d2 / "config.json.part").mkdir(parents=True)
        calls: list[str] = []

        def fetch(url, dest):
            calls.append(url)
            self._static_fetch(url, dest)

        with self.assertRaisesRegex(
            recognition.RecognitionUnavailable,
            r"speech recognition model download failed:",
        ):
            gpu_stt.prepare(home2, approve_download=True, fetch=fetch)
        self.assertEqual(calls, [])

    def test_concurrent_prepares_serialize_on_pack_lock(self):
        started = threading.Event()
        release = threading.Event()
        fetched: list[tuple[str, str]] = []
        fetch_lock = threading.Lock()
        results: list[dict] = []
        errors: list[BaseException] = []

        def fetch_a(url, dest):
            with fetch_lock:
                fetched.append(("a", url))
            if url.endswith("/config.json"):
                started.set()
                if not release.wait(10):
                    raise TimeoutError("release event never set")
            self._static_fetch(url, dest)

        def fetch_b(url, dest):
            with fetch_lock:
                fetched.append(("b", url))
            self._static_fetch(url, dest)

        def run(fn):
            try:
                results.append(gpu_stt.prepare(self.home, approve_download=True, fetch=fn))
            except BaseException as exc:
                errors.append(exc)

        thread_a = threading.Thread(target=run, args=(fetch_a,))
        thread_b = threading.Thread(target=run, args=(fetch_b,))
        thread_a.start()
        try:
            self.assertTrue(started.wait(10), "first prepare never entered fetch")
            thread_b.start()
            thread_b.join(0.5)
            self.assertTrue(thread_b.is_alive(), "second prepare did not wait on pack lock")
        finally:
            release.set()
            thread_a.join(30)
            if thread_b.ident is not None:
                thread_b.join(30)

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(thread_b.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["verified"] for r in results))
        self.assertEqual([who for who, _ in fetched], ["a"] * len(self.files))
        lock_path = self.home / "voice" / f".{gpu_stt.GPU_STT_MODEL['id']}.lock"
        self.assertTrue(lock_path.is_file())
        self.assertEqual(list(self.directory.rglob("*.lock")), [])
        self._assert_installed()

    def test_waiting_prepare_reuses_cached_verified_identities_without_rehashing(self):
        # First prepare populates verified files and cache without re-hashing
        # freshly staged files during final verification.
        with mock.patch.object(
            synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
        ) as fresh_sha:
            gpu_stt.prepare(self.home, approve_download=True, fetch=self._static_fetch)
        self.assertEqual(fresh_sha.call_count, len(self.files))
        # Simulate a second prepare that enters the lock after the pack is verified
        # (e.g., it raced the first prepare before the lock).
        with mock.patch.object(gpu_stt, "model_verified", side_effect=[False, True]), \
                mock.patch.object(
                    synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
                ) as sha_stream:
            result = gpu_stt.prepare(
                self.home, approve_download=True, fetch=self._static_fetch,
            )
        self.assertTrue(result["verified"])
        self.assertEqual(sha_stream.call_count, 0)

    def test_repair_of_same_size_corrupt_weights_reuses_cached_mismatch(self):
        # Valid config.json + same-size bit-flipped model.safetensors.
        self.directory.mkdir(parents=True)
        (self.directory / "config.json").write_bytes(self.content["config.json"])
        bad = bytearray(self.content["model.safetensors"])
        bad[0] ^= 1
        (self.directory / "model.safetensors").write_bytes(bytes(bad))

        hashed_sizes: list[int] = []
        real_sha = synthesis._sha256_stream

        def counting_sha(fh, expected_bytes):
            hashed_sizes.append(expected_bytes)
            return real_sha(fh, expected_bytes)

        with mock.patch.object(synthesis, "_sha256_stream", side_effect=counting_sha):
            result = gpu_stt.prepare(
                self.home, approve_download=True, fetch=self._static_fetch,
            )
        self.assertTrue(result["verified"])
        weights_size = self._entry("model.safetensors")["bytes"]
        # 1 hash in precheck (detects mismatch + caches it) + 1 hash of staged
        # replacement; in-lock skip check and final verification do 0 re-hashes.
        self.assertEqual(hashed_sizes.count(weights_size), 2)
        self._assert_installed()

    def test_unopenable_pack_lock_refuses_before_fetch(self):
        for kind in ("directory", "symlink"):
            with self.subTest(kind):
                home = self.home / f"lock-{kind}"
                lock_path = home / "voice" / f".{gpu_stt.GPU_STT_MODEL['id']}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                victim = home / "victim.txt"
                if kind == "directory":
                    lock_path.mkdir()
                else:
                    victim.write_bytes(b"keep me")
                    os.symlink(victim, lock_path)
                calls: list[str] = []

                def fetch(url, dest):
                    calls.append(url)
                    self._static_fetch(url, dest)

                with self.assertRaisesRegex(
                    recognition.RecognitionUnavailable,
                    r"speech recognition model download failed:",
                ):
                    gpu_stt.prepare(home, approve_download=True, fetch=fetch)
                self.assertEqual(calls, [])
                if kind == "symlink":
                    self.assertEqual(victim.read_bytes(), b"keep me")

    def test_staging_cleanup_on_errors_mismatch_fifo_sparse_and_interrupt(self):
        def fetch_same_size_bad(url, dest):
            data = bytearray(self.content[url.removeprefix(self.http.base + "/")])
            data[0] ^= 1
            dest.write_bytes(bytes(data))

        def fetch_sparse_3gib(_url, dest):
            with open(dest, "wb") as fh:
                fh.truncate(3 << 30)

        def fetch_fifo(_url, dest):
            os.mkfifo(dest)

        def fetch_raising(exc):
            def fn(url, dest):
                self._static_fetch(url, dest)
                raise exc
            return fn

        cases = [
            ("same-size-bad", fetch_same_size_bad, recognition.RecognitionUnavailable, r"sha256 mismatch"),
            ("sparse-3gib", fetch_sparse_3gib, recognition.RecognitionUnavailable, r"wrong size"),
            ("fifo", fetch_fifo, recognition.RecognitionUnavailable, r"failed verification"),
            ("oserror", fetch_raising(OSError("network down")), recognition.RecognitionUnavailable, r"download failed"),
            ("interrupt", fetch_raising(KeyboardInterrupt()), KeyboardInterrupt, None),
        ]
        for label, fetch, exc_type, pattern in cases:
            with self.subTest(label):
                def run():
                    if pattern is None:
                        with self.assertRaises(exc_type):
                            gpu_stt.prepare(self.home, approve_download=True, fetch=fetch)
                    else:
                        with self.assertRaisesRegex(exc_type, pattern):
                            gpu_stt.prepare(self.home, approve_download=True, fetch=fetch)

                _run_with_alarm(5, run)
                self.assertEqual(self._parts(), [])
                self.assertFalse((self.directory / "config.json").exists())

    def test_failed_part_cleanup_preserves_original_error(self):
        real_unlink = Path.unlink

        def failing_unlink(path, *args, **kwargs):
            if path.name.endswith(".part"):
                raise PermissionError("read-only")
            return real_unlink(path, *args, **kwargs)

        def fetch(url, dest):
            self._static_fetch(url, dest)
            raise KeyboardInterrupt

        with mock.patch.object(Path, "unlink", failing_unlink), \
                self.assertRaises(KeyboardInterrupt):
            gpu_stt.prepare(self.home, approve_download=True, fetch=fetch)

    def test_symlinked_fifo_and_unreadable_dest_replaced_not_followed(self):
        self._serve_pack()
        self.directory.mkdir(parents=True)
        victim = self.home / "victim.txt"
        victim.write_bytes(b"keep me")
        os.symlink(victim, self.directory / "config.json")
        os.mkfifo(self.directory / "model.safetensors")

        result = _run_with_alarm(5, lambda: gpu_stt.prepare(self.home, approve_download=True))
        self.assertTrue(result["verified"])
        self.assertEqual(victim.read_bytes(), b"keep me")
        self._assert_installed()

        # Mode-000 pinned-size dest is also replaced cleanly instead of crashing.
        home2 = self.home / "unreadable-dest"
        d2 = gpu_stt.model_dir(home2)
        d2.mkdir(parents=True)
        cfg = d2 / "config.json"
        cfg.write_bytes(self.content["config.json"])
        os.chmod(cfg, 0o000)
        try:
            result2 = gpu_stt.prepare(home2, approve_download=True)
            self.assertTrue(result2["verified"])
            self.assertEqual(cfg.read_bytes(), self.content["config.json"])
        finally:
            if cfg.exists():
                os.chmod(cfg, 0o600)

    def test_prepare_raises_if_post_download_verification_fails(self):
        self._serve_pack()
        with mock.patch.object(gpu_stt, "model_verified", side_effect=[False, False]), \
                self.assertRaisesRegex(
                    recognition.RecognitionUnavailable,
                    r"speech recognition model failed verification",
                ):
            gpu_stt.prepare(self.home, approve_download=True)


class RegularFileVerificationTests(unittest.TestCase):
    """_file_status(), _verified(), model_verified(), and probe_runtime()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.files, self.content = _make_pack()
        self.enterContext(mock.patch.dict(gpu_stt.GPU_STT_MODEL, {"files": self.files}))
        gpu_stt._verified.cache_clear()
        self.addCleanup(gpu_stt._verified.cache_clear)
        self.directory = gpu_stt.model_dir(self.home)
        self.directory.mkdir(parents=True)

    def _populate_valid(self):
        for name, data in self.content.items():
            (self.directory / name).write_bytes(data)

    def test_fifo_and_symlink_to_fifo_rejected_without_blocking(self):
        self._populate_valid()
        cfg = self.directory / "config.json"
        for kind in ("fifo", "symlink-to-fifo"):
            with self.subTest(kind):
                cfg.unlink(missing_ok=True)
                if kind == "fifo":
                    os.mkfifo(cfg)
                else:
                    fifo = self.home / "target.fifo"
                    if not fifo.exists():
                        os.mkfifo(fifo)
                    os.symlink(fifo, cfg)

                def check():
                    ok, bad = gpu_stt._file_status(self.directory)
                    self.assertFalse(ok)
                    self.assertEqual(bad, ["config.json"])
                    self.assertFalse(gpu_stt._verified(self.directory))
                    self.assertFalse(gpu_stt.model_verified(self.home))
                    probe = gpu_stt.probe_runtime(self.home, verify_cache=True)
                    self.assertEqual(probe["facts"]["cache"], "missing")

                _run_with_alarm(5, check)

    def test_unreadable_directory_dangling_loop_and_sparse_rejected(self):
        self._populate_valid()
        cfg = self.directory / "config.json"

        cases = ("mode-000", "directory", "dangling", "loop", "sparse-3gib")
        for kind in cases:
            if kind == "mode-000" and os.geteuid() == 0:
                continue
            with self.subTest(kind):
                if cfg.is_dir() and not cfg.is_symlink():
                    cfg.rmdir()
                elif cfg.exists() or cfg.is_symlink():
                    try:
                        os.chmod(cfg, 0o600)
                    except OSError:
                        pass
                    cfg.unlink()
                if kind == "mode-000":
                    cfg.write_bytes(self.content["config.json"])
                    os.chmod(cfg, 0o000)
                elif kind == "directory":
                    cfg.mkdir()
                elif kind == "dangling":
                    os.symlink(self.home / "does-not-exist", cfg)
                elif kind == "loop":
                    os.symlink(cfg, cfg)
                elif kind == "sparse-3gib":
                    with open(cfg, "wb") as fh:
                        fh.truncate(3 << 30)

                try:
                    ok, bad = gpu_stt._file_status(self.directory)
                    self.assertFalse(ok)
                    self.assertEqual(bad, ["config.json"])
                    self.assertFalse(gpu_stt.model_verified(self.home))
                    probe = gpu_stt.probe_runtime(self.home, verify_cache=True)
                    self.assertEqual(probe["facts"]["cache"], "missing")
                finally:
                    if kind == "mode-000":
                        os.chmod(cfg, 0o600)

    def test_valid_regular_and_symlink_verify_and_cache_hits_on_repeat(self):
        self._populate_valid()
        real_cfg = self.home / "real_config.json"
        real_cfg.write_bytes(self.content["config.json"])
        (self.directory / "config.json").unlink()
        os.symlink(real_cfg, self.directory / "config.json")

        with mock.patch.object(
            synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
        ) as sha_stream:
            self.assertTrue(gpu_stt.model_verified(self.home))
            self.assertEqual(sha_stream.call_count, len(self.files))
            # Repeat call skips re-hashing while identities match.
            self.assertTrue(gpu_stt.model_verified(self.home))
            self.assertEqual(sha_stream.call_count, len(self.files))
            # Clearing cache forces re-hash.
            gpu_stt._verified.cache_clear()
            self.assertTrue(gpu_stt.model_verified(self.home))
            self.assertEqual(sha_stream.call_count, 2 * len(self.files))

    def test_failed_verification_is_not_cached_and_pin_change_invalidates_cache(self):
        self._populate_valid()
        with mock.patch.object(
            synthesis, "_verify_pinned", side_effect=OSError("transient"),
        ):
            self.assertFalse(gpu_stt.model_verified(self.home))
        # Recovery succeeds without needing cache_clear().
        self.assertTrue(gpu_stt.model_verified(self.home))

        # Changing the pinned sha256 invalidates the cached identity.
        altered = [dict(e) for e in self.files]
        altered[0]["sha256"] = "0" * 64
        with mock.patch.dict(gpu_stt.GPU_STT_MODEL, {"files": altered}):
            self.assertFalse(gpu_stt.model_verified(self.home))

    def test_broken_pack_avoids_redundant_hashing_of_large_weights(self):
        # Put the large file before a small trailing file in pin order, matching
        # GPU_STT_MODEL where model.safetensors precedes tokenizer/vocab files.
        ordered = [
            next(e for e in self.files if e["name"] == "model.safetensors"),
            next(e for e in self.files if e["name"] == "config.json"),
        ]
        with mock.patch.dict(gpu_stt.GPU_STT_MODEL, {"files": ordered}):
            gpu_stt._verified.cache_clear()
            (self.directory / "model.safetensors").write_bytes(
                self.content["model.safetensors"],
            )
            # Trailing config.json is missing: size/presence gate stops before hashing.
            with mock.patch.object(
                synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
            ) as sha_stream:
                self.assertFalse(gpu_stt.model_verified(self.home))
                self.assertFalse(gpu_stt.model_verified(self.home))
            self.assertEqual(sha_stream.call_count, 0)

            # Wrong-sized large file stops in the pre-hash size gate before
            # hashing even smaller valid files.
            (self.directory / "config.json").write_bytes(self.content["config.json"])
            (self.directory / "model.safetensors").write_bytes(
                self.content["model.safetensors"][:-1],
            )
            with mock.patch.object(
                synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
            ) as sha_stream:
                self.assertFalse(gpu_stt.model_verified(self.home))
            self.assertEqual(sha_stream.call_count, 0)

            # Both files present, but model.safetensors is same-size corrupt:
            bad = bytearray(self.content["model.safetensors"])
            bad[0] ^= 1
            weights = self.directory / "model.safetensors"
            weights.write_bytes(bytes(bad))
            with mock.patch.object(
                synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
            ) as sha_stream:
                self.assertFalse(gpu_stt.model_verified(self.home))
                first_calls = sha_stream.call_count
                self.assertGreaterEqual(first_calls, 1)
                self.assertFalse(gpu_stt.model_verified(self.home))
                self.assertFalse(gpu_stt.model_verified(self.home))
                self.assertEqual(sha_stream.call_count, first_calls)

            # Replacing the corrupt weights with valid bytes invalidates the
            # cached mismatch immediately via the new file identity.
            swap = weights.with_name("weights.good")
            swap.write_bytes(self.content["model.safetensors"])
            os.replace(swap, weights)
            self.assertTrue(gpu_stt.model_verified(self.home))

    def test_growing_file_mid_hash_fails_verification(self):
        self._populate_valid()
        cfg = self.directory / "config.json"
        cfg_entry = next(e for e in self.files if e["name"] == "config.json")
        real_open_regular = synthesis._open_regular
        cfg_opens = 0

        def open_then_grow_on_verify(path: Path):
            nonlocal cfg_opens
            fh, st = real_open_regular(path)
            if path == cfg:
                cfg_opens += 1
                if cfg_opens == 2:
                    # Second open is inside _verify_pinned after its fstat
                    # passes; append so _sha256_stream reads past pinned bytes.
                    with open(cfg, "ab") as writer:
                        writer.write(b"extra")
            return fh, st

        with mock.patch.object(
            synthesis, "_open_regular", side_effect=open_then_grow_on_verify,
        ), mock.patch.object(
            synthesis, "_sha256_stream", wraps=synthesis._sha256_stream,
        ) as sha_stream:
            self.assertFalse(gpu_stt.model_verified(self.home))
        self.assertEqual(cfg_opens, 2)
        self.assertEqual(sha_stream.call_count, 1)
        self.assertNotIn(
            gpu_stt._cache_key(self.directory, cfg_entry),
            gpu_stt._verified_identities,
        )


class AcceptanceReceiptBoundsTests(unittest.TestCase):
    """read_acceptance_receipt() and write_acceptance_receipt() bounds and staging."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.directory = gpu_stt.model_dir(self.home)
        self.directory.mkdir(parents=True)
        self.identity = {
            "mlx_backend": "backend-a",
            "mlx_backend_detail": None,
            "model_sha256": "model-a",
        }
        self.identity_mock = self.enterContext(mock.patch.object(
            gpu_stt, "_runtime_identity",
            side_effect=lambda _home: dict(self.identity),
        ))

    def test_read_receipt_refuses_fifo_and_symlink_to_fifo_without_blocking(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        for kind in ("fifo", "symlink-to-fifo"):
            with self.subTest(kind):
                receipt_path.unlink(missing_ok=True)
                if kind == "fifo":
                    os.mkfifo(receipt_path)
                else:
                    fifo = self.home / "receipt.fifo"
                    if not fifo.exists():
                        os.mkfifo(fifo)
                    os.symlink(fifo, receipt_path)
                result = _run_with_alarm(
                    5, lambda: gpu_stt.read_acceptance_receipt(self.home),
                )
                self.assertIsNone(result)

    def test_watchdog_sentinel_escapes_read_acceptance_receipt(self):
        with mock.patch.object(
            synthesis, "_read_state_json", side_effect=_WatchdogExpired("hung"),
        ), self.assertRaises(_WatchdogExpired):
            gpu_stt.read_acceptance_receipt(self.home)

    def test_read_receipt_refuses_directory_unreadable_and_dangling_symlink(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        for kind in ("directory", "mode-000", "dangling"):
            if kind == "mode-000" and os.geteuid() == 0:
                continue
            with self.subTest(kind):
                if receipt_path.is_dir() and not receipt_path.is_symlink():
                    receipt_path.rmdir()
                elif receipt_path.exists() or receipt_path.is_symlink():
                    try:
                        os.chmod(receipt_path, 0o600)
                    except OSError:
                        pass
                    receipt_path.unlink()
                if kind == "directory":
                    receipt_path.mkdir()
                elif kind == "mode-000":
                    receipt_path.write_text('{"mlx_backend":"backend-a"}')
                    os.chmod(receipt_path, 0o000)
                else:
                    os.symlink(self.home / "missing.json", receipt_path)
                try:
                    self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))
                finally:
                    if kind == "mode-000":
                        os.chmod(receipt_path, 0o600)

    def test_read_receipt_refuses_oversized_sparse_and_cap_plus_one(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        for label, size in (
            ("sparse-3gib", 3 << 30),
            ("cap-plus-one", synthesis._STATE_FILE_MAX_BYTES + 1),
        ):
            with self.subTest(label):
                with open(receipt_path, "wb") as fh:
                    fh.truncate(size)
                self.identity_mock.reset_mock()
                self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))
                self.identity_mock.assert_not_called()

    def test_read_receipt_refuses_deep_nesting_and_recursion_error(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        valid = gpu_stt.write_acceptance_receipt(self.home, **PASSING)

        # Past the 32-level depth cap inside an otherwise valid receipt dict
        # (so rejection depends on the depth check, not the dict check).
        deep = {}
        cur = deep
        for _ in range(synthesis._STATE_FILE_MAX_DEPTH + 4):
            cur["a"] = {}
            cur = cur["a"]
        receipt_path.write_text(json.dumps({**valid, "extra": deep}))
        self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))

        # Deep JSON fixture (triggers RecursionError or depth cap depending on C stack).
        receipt_path.write_bytes(b"[" * 200_000 + b"]" * 200_000)
        self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))

        # Deterministic RecursionError regardless of Python runner C stack size.
        receipt_path.write_text(json.dumps(valid))
        with mock.patch.object(
            synthesis.json, "loads", side_effect=RecursionError("deep"),
        ):
            self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))

    def test_read_receipt_refuses_non_dict_and_bad_utf8_without_identity_check(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        for label, raw in (
            ("json-array", b"[1, 2, 3]"),
            ("json-string", b'"ok"'),
            ("invalid-utf8", b"\xff\xfe"),
        ):
            with self.subTest(label):
                receipt_path.write_bytes(raw)
                self.identity_mock.reset_mock()
                self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))
                self.identity_mock.assert_not_called()

    def test_read_receipt_accepts_valid_regular_and_symlinked_receipt(self):
        written = gpu_stt.write_acceptance_receipt(self.home, **PASSING)
        self.assertEqual(gpu_stt.read_acceptance_receipt(self.home), written)

        receipt_path = gpu_stt._receipt_path(self.home)
        real_copy = self.home / "real_receipt.json"
        real_copy.write_bytes(receipt_path.read_bytes())
        receipt_path.unlink()
        os.symlink(real_copy, receipt_path)
        self.assertEqual(gpu_stt.read_acceptance_receipt(self.home), written)

    def test_write_receipt_ignores_planted_tmp_symlink_and_sets_mode_0600(self):
        victim = self.home / "victim.txt"
        victim.write_bytes(b"keep me")
        os.chmod(victim, 0o644)
        planted = gpu_stt._receipt_path(self.home).with_suffix(".tmp")
        os.symlink(victim, planted)

        written = gpu_stt.write_acceptance_receipt(self.home, **PASSING)
        self.assertEqual(victim.read_bytes(), b"keep me")
        self.assertEqual(victim.stat().st_mode & 0o777, 0o644)
        receipt_path = gpu_stt._receipt_path(self.home)
        self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(receipt_path.read_text()), written)
        # Only the untouched planted symlink matches *.tmp; no leaked staging file.
        self.assertEqual(
            [p.name for p in self.directory.glob("*.tmp")],
            [planted.name],
        )

    def test_write_receipt_replaces_symlinked_receipt_without_touching_target(self):
        victim = self.home / "victim.txt"
        victim.write_bytes(b"keep me")
        receipt_path = gpu_stt._receipt_path(self.home)
        os.symlink(victim, receipt_path)

        written = gpu_stt.write_acceptance_receipt(self.home, **PASSING)
        self.assertEqual(victim.read_bytes(), b"keep me")
        self.assertFalse(receipt_path.is_symlink())
        self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(gpu_stt.read_acceptance_receipt(self.home), written)

    def test_write_receipt_cleans_unique_tmp_on_replace_failure_or_interrupt(self):
        prior = gpu_stt.write_acceptance_receipt(self.home, **PASSING)
        for exc in (OSError("disk full"), KeyboardInterrupt()):
            with self.subTest(type(exc).__name__):
                with mock.patch.object(synthesis.os, "replace", side_effect=exc), \
                        self.assertRaises(type(exc)):
                    gpu_stt.write_acceptance_receipt(self.home, **PASSING)
                self.assertEqual(list(self.directory.glob("*.tmp")), [])
                self.assertEqual(gpu_stt.read_acceptance_receipt(self.home), prior)

    def test_write_receipt_refuses_oversized_or_deeply_nested_payload(self):
        receipt_path = gpu_stt._receipt_path(self.home)

        oversized = {
            **PASSING,
            "receipt_source": "x" * (synthesis._STATE_FILE_MAX_BYTES + 1),
        }
        with self.assertRaisesRegex(ValueError, r"exceeds"):
            gpu_stt.write_acceptance_receipt(self.home, **oversized)
        self.assertFalse(receipt_path.exists())

        deep = {}
        cur = deep
        for _ in range(synthesis._STATE_FILE_MAX_DEPTH + 2):
            cur["a"] = {}
            cur = cur["a"]
        nested = {
            **PASSING,
            "latency_ms": {**PASSING["latency_ms"], "nested": deep},
        }
        with self.assertRaisesRegex(ValueError, r"nesting depth"):
            gpu_stt.write_acceptance_receipt(self.home, **nested)
        self.assertFalse(receipt_path.exists())

    def test_read_receipt_refuses_legacy_invalid_or_nan_metrics(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        valid = gpu_stt.write_acceptance_receipt(self.home, **PASSING)
        # Valid legacy schema-v2 receipt continues to qualify (with or without p50).
        self.assertEqual(gpu_stt.read_acceptance_receipt(self.home), valid)
        no_p50 = {**valid, "latency_ms": {"p95": 468.1}}
        receipt_path.write_text(json.dumps(no_p50))
        self.assertEqual(gpu_stt.read_acceptance_receipt(self.home), no_p50)

        bad_payloads = [
            ("wer-nan", json.dumps({
                **valid,
                "wer_by_subset": {**valid["wer_by_subset"], "test-clean": float("nan")},
            }, allow_nan=True)),
            ("wer-over-limit", json.dumps({
                **valid,
                "wer_by_subset": {**valid["wer_by_subset"], "test-clean": 0.5},
            })),
            ("wer-negative", json.dumps({
                **valid,
                "wer_by_subset": {**valid["wer_by_subset"], "test-clean": -0.01},
            })),
            ("wer-huge-int", json.dumps({
                **valid,
                "wer_by_subset": {**valid["wer_by_subset"], "test-clean": 10 ** 400},
            })),
            ("empty-rate-low", json.dumps({
                **valid,
                "empty_rate_by_subset": {"silence": 0.5, "noise": 1.0},
            })),
            ("empty-rate-above-one", json.dumps({
                **valid,
                "empty_rate_by_subset": {"silence": 1.01, "noise": 1.0},
            })),
            ("p95-nan", json.dumps({
                **valid,
                "latency_ms": {"p50": 100.0, "p95": float("nan")},
            }, allow_nan=True)),
            ("p95-negative", json.dumps({
                **valid,
                "latency_ms": {"p95": -1.0},
            })),
            ("p95-huge-int", json.dumps({
                **valid,
                "latency_ms": {"p50": 100.0, "p95": 10 ** 400},
            })),
            ("p50-nan", json.dumps({
                **valid,
                "latency_ms": {"p50": float("nan"), "p95": 468.1},
            }, allow_nan=True)),
            ("p50-negative", json.dumps({
                **valid,
                "latency_ms": {"p50": -1.0, "p95": 468.1},
            })),
            ("p50-exceeds-p95", json.dumps({
                **valid,
                "latency_ms": {"p50": 3000.0, "p95": 450.0},
            })),
            ("p99-nan-extra", json.dumps({
                **valid,
                "latency_ms": {"p50": 100.0, "p95": 450.0, "p99": float("nan")},
            }, allow_nan=True)),
            ("recorded-at-inf", json.dumps({
                **valid,
                "recorded_at": float("inf"),
            }, allow_nan=True)),
            ("nested-list-nan", json.dumps({
                **valid,
                "extra": [1.0, float("nan")],
            }, allow_nan=True)),
            ("cpu-events-nonzero", json.dumps({**valid, "cpu_tensor_events": 1})),
            ("cpu-events-false", json.dumps({**valid, "cpu_tensor_events": False})),
            ("cpu-events-float", json.dumps({**valid, "cpu_tensor_events": 0.0})),
            ("missing-metrics", json.dumps({
                "mlx_backend": "backend-a",
                "model_sha256": "model-a",
            })),
        ]
        for label, text in bad_payloads:
            with self.subTest(label):
                receipt_path.write_text(text)
                self.identity_mock.reset_mock()
                self.assertIsNone(gpu_stt.read_acceptance_receipt(self.home))
                self.identity_mock.assert_not_called()

    def test_write_receipt_refuses_non_finite_and_bool_metrics(self):
        receipt_path = gpu_stt._receipt_path(self.home)
        # Omitting optional p50 remains accepted.
        written = gpu_stt.write_acceptance_receipt(
            self.home, **{**PASSING, "latency_ms": {"p95": 468.1}},
        )
        self.assertEqual(written["latency_ms"], {"p95": 468.1})
        receipt_path.unlink()

        threshold_cases = [
            ("wer-nan", {"wer_by_subset": {**PASSING["wer_by_subset"], "test-clean": float("nan")}}),
            ("wer-inf", {"wer_by_subset": {**PASSING["wer_by_subset"], "test-clean": float("inf")}}),
            ("wer-negative", {"wer_by_subset": {**PASSING["wer_by_subset"], "test-clean": -0.01}}),
            ("wer-huge-int", {"wer_by_subset": {**PASSING["wer_by_subset"], "test-clean": 10 ** 400}}),
            ("wer-bool", {"wer_by_subset": {**PASSING["wer_by_subset"], "test-clean": False}}),
            ("empty-nan", {"empty_rate_by_subset": {**PASSING["empty_rate_by_subset"], "silence": float("nan")}}),
            ("empty-above-one", {"empty_rate_by_subset": {**PASSING["empty_rate_by_subset"], "silence": 1.01}}),
            ("empty-bool", {"empty_rate_by_subset": {**PASSING["empty_rate_by_subset"], "silence": True}}),
            ("p95-nan", {"latency_ms": {**PASSING["latency_ms"], "p95": float("nan")}}),
            ("p95-neg-inf", {"latency_ms": {**PASSING["latency_ms"], "p95": float("-inf")}}),
            ("p95-negative", {"latency_ms": {"p95": -1.0}}),
            ("p95-huge-int", {"latency_ms": {**PASSING["latency_ms"], "p95": 10 ** 400}}),
            ("p50-nan", {"latency_ms": {"p50": float("nan"), "p95": 468.1}}),
            ("p50-negative", {"latency_ms": {"p50": -1.0, "p95": 468.1}}),
            ("p50-exceeds-p95", {"latency_ms": {"p50": 3000.0, "p95": 450.0}}),
            ("cpu-bool-false", {"cpu_tensor_events": False}),
            ("cpu-float-zero", {"cpu_tensor_events": 0.0}),
        ]
        for label, overrides in threshold_cases:
            with self.subTest(label):
                self.identity_mock.reset_mock()
                with self.assertRaisesRegex(ValueError, r"acceptance refused"):
                    gpu_stt.write_acceptance_receipt(self.home, **{**PASSING, **overrides})
                self.identity_mock.assert_not_called()
                self.assertFalse(receipt_path.exists())

        with self.subTest("extra-latency-nan"):
            with self.assertRaisesRegex(ValueError, r"acceptance refused"):
                gpu_stt.write_acceptance_receipt(
                    self.home,
                    **{**PASSING, "latency_ms": {**PASSING["latency_ms"], "p99": float("nan")}},
                )
            self.assertFalse(receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
