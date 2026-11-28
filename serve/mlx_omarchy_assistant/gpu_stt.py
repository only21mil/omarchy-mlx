# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU speech recognition for the offline assistant (mlx Vulkan backend).

The recognition backend for Apple Silicon Linux hosts without a live
ANE. An owned worker subprocess (``gpu_stt_worker``) keeps the pinned
Parakeet-TDT model resident on the mlx device, resamples to 16 kHz there
and runs mlx-audio's ``generate``; cancellation and wedges end with a
confirmed process-group kill.

The model is a pinned file set downloaded approve-first into
``<home>/voice/<id>/`` and sha256-verified, like the TTS pack.
``ready`` needs an acceptance receipt that names this process's mlx
backend binary and the verified model hash, and records every frozen
threshold in ACCEPTANCE_THRESHOLDS as passed.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import json
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .recognition import (
    RecognitionCancelled,
    RecognitionInputError,
    RecognitionTimedOut,
    RecognitionUnavailable,
)

TARGET_SAMPLE_RATE = 16_000
CANCEL_GRACE_SECONDS = 5.0
BACKEND_KIND = "gpu"
MLX_AUDIO_VERSION = "0.5.6"

GPU_STT_MODEL = {
    "id": "parakeet-tdt-0.6b-v3",
    "repo": "mlx-community/parakeet-tdt-0.6b-v3",
    "revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    "license": "cc-by-4.0",
    "weights_bytes": 2_508_288_736,
    "files": [
        {"name": "config.json", "bytes": 244_093,
         "sha256": "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c"},
        {"name": "model.safetensors", "bytes": 2_508_288_736,
         "sha256": "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592"},
        {"name": "tokenizer.model", "bytes": 360_916,
         "sha256": "eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e"},
        {"name": "tokenizer.vocab", "bytes": 101_024,
         "sha256": "41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17"},
        {"name": "vocab.txt", "bytes": 46_772,
         "sha256": "3cde1409fd78783a79b29ed4d32da57c746993856f7c8263bcb905d2e5839db7"},
    ],
}


def model_dir(home: Path) -> Path:
    return Path(home) / "voice" / GPU_STT_MODEL["id"]


_verified_lock = threading.Lock()
# Maps (dir, name, pinned_bytes, pinned_sha256) -> (stat_identity, verified_ok).
# Caches both verified hits and same-size digest mismatches; never caches OSError.
_verified_identities: dict[tuple[str, str, int, str], tuple[tuple, bool]] = {}


def _cache_key(directory: Path, entry: dict) -> tuple[str, str, int, str]:
    return (str(directory), entry["name"], entry["bytes"], entry["sha256"])


def _file_status(directory: Path) -> tuple[bool, list[str]]:
    """(all present as readable regular files with pinned sizes, names missing or wrong-sized)."""
    from .synthesis import _open_regular

    bad: list[str] = []
    for entry in GPU_STT_MODEL["files"]:
        try:
            fh, st = _open_regular(directory / entry["name"])
        except OSError:
            bad.append(entry["name"])
            continue
        with fh:
            if st.st_size != entry["bytes"]:
                bad.append(entry["name"])
    return not bad, bad


def _verified(directory: Path) -> bool:
    """Verify every pinned file via non-blocking regular-file opens.

    Checks presence and pinned size for all files before hashing any file so
    a missing or wrong-sized trailing file never triggers hashing multi-GB
    weights. Caches verified hits and same-size digest mismatches in memory
    by (dir, name, size, sha256) and stat identity; transient OSErrors are
    never cached.
    """
    from .synthesis import _identity, _open_regular, _verify_pinned

    entries: list[tuple[dict, tuple[str, str, int, str], tuple]] = []
    for entry in GPU_STT_MODEL["files"]:
        try:
            fh, st = _open_regular(directory / entry["name"])
        except OSError:
            return False
        with fh:
            if st.st_size != entry["bytes"]:
                return False
            ident = _identity(st)
        entries.append((entry, _cache_key(directory, entry), ident))

    for entry, key, pre_ident in sorted(entries, key=lambda item: item[0]["bytes"]):
        with _verified_lock:
            cached = _verified_identities.get(key)
        if cached is not None and cached[0] == pre_ident:
            if not cached[1]:
                return False
            continue
        path = directory / entry["name"]
        try:
            ident, actual = _verify_pinned(path, entry)
        except OSError:
            return False
        if ident is not None:
            with _verified_lock:
                _verified_identities[key] = (ident, True)
            continue
        if actual is not None:
            # Record same-size digest mismatch only when the file identity
            # did not change across the hash.
            try:
                fh, st = _open_regular(path)
                with fh:
                    post_ident = _identity(st)
            except OSError:
                post_ident = None
            if post_ident == pre_ident:
                with _verified_lock:
                    _verified_identities[key] = (pre_ident, False)
        return False
    return True


def _clear_verified_cache() -> None:
    with _verified_lock:
        _verified_identities.clear()


_verified.cache_clear = _clear_verified_cache  # type: ignore[attr-defined]


def model_verified(home: Path) -> bool:
    return _verified(model_dir(home))


def prepare(home: Path, approve_download: bool, *, fetch=None) -> dict:
    """Download the pinned model with explicit approval; verify every file.

    ``fetch(url, path)`` replaces the network for tests; the default download
    is capped at each file's pinned size. Concurrent prepares serialize on
    ``voice/.<model>.lock``; defending against a hostile co-writer who already
    has write permission on ``home`` is out of scope.
    """
    from .synthesis import (
        _URL_TEMPLATE,
        _default_fetch,
        _error_reason,
        _identity,
        _open_regular,
        _pack_lock,
        _verify_pinned,
    )

    directory = model_dir(home)
    if model_verified(home):
        return {"downloaded": True, "verified": True, "path": str(directory)}
    if not approve_download:
        return {"downloaded": False, "verified": False, "path": str(directory),
                "reason": ("speech recognition model download requires explicit "
                           f"approval ({GPU_STT_MODEL['weights_bytes']} bytes, "
                           f"{GPU_STT_MODEL['license']} license, "
                           f"{GPU_STT_MODEL['repo']} revision {GPU_STT_MODEL['revision']})")}
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _pack_lock(directory):
            for entry in sorted(GPU_STT_MODEL["files"], key=lambda e: e["bytes"]):
                dest = directory / entry["name"]
                key = _cache_key(directory, entry)
                with _verified_lock:
                    cached_entry = _verified_identities.get(key)
                skip_verify = False
                if cached_entry is not None and not cached_entry[1]:
                    # Known same-size digest mismatch: skip re-hashing if the
                    # on-disk file still has that exact identity.
                    try:
                        fh, st = _open_regular(dest)
                        with fh:
                            current_ident = (
                                _identity(st)
                                if st.st_size == entry["bytes"]
                                else None
                            )
                    except OSError:
                        current_ident = None
                    skip_verify = (
                        current_ident is not None
                        and current_ident == cached_entry[0]
                    )
                if not skip_verify:
                    cached_ok = (
                        cached_entry[0]
                        if cached_entry is not None and cached_entry[1]
                        else None
                    )
                    try:
                        ident, _actual = _verify_pinned(
                            dest, entry, cached=cached_ok,
                        )
                    except OSError:
                        ident = None
                    if ident is not None:
                        if cached_ok != ident:
                            with _verified_lock:
                                _verified_identities[key] = (ident, True)
                        continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Fixed staging path under the pack lock: one prepare owns it
                # at a time, and any leftover file or planted symlink is
                # removed unfollowed before fetching.
                tmp = dest.with_name(dest.name + ".part")
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                url = _URL_TEMPLATE.format(
                    repo=GPU_STT_MODEL["repo"],
                    revision=GPU_STT_MODEL["revision"],
                    name=entry["name"],
                )
                try:
                    if fetch is None:
                        _default_fetch(url, tmp, max_bytes=entry["bytes"])
                    else:
                        fetch(url, tmp)
                    try:
                        ident, actual = _verify_pinned(tmp, entry)
                    except OSError as exc:
                        raise RecognitionUnavailable(
                            f"speech model file {entry['name']} failed "
                            f"verification: {_error_reason(exc)}"
                        ) from exc
                    if ident is None:
                        got = actual or "a file of the wrong size"
                        raise RecognitionUnavailable(
                            f"speech model file {entry['name']}: sha256 mismatch "
                            f"(expected {entry['sha256']}, got {got})"
                        )
                    os.replace(tmp, dest)
                    try:
                        fh, st = _open_regular(dest)
                        with fh:
                            # synthesis._identity tuple is (size, mtime_ns, ctime_ns, ino);
                            # rename(2) can bump ctime_ns while preserving size, mtime, and inode.
                            if (st.st_size == ident[0]
                                    and st.st_mtime_ns == ident[1]
                                    and st.st_ino == ident[3]):
                                with _verified_lock:
                                    _verified_identities[key] = (_identity(st), True)
                    except OSError:
                        pass
                finally:
                    with contextlib.suppress(OSError):
                        tmp.unlink(missing_ok=True)
            if not model_verified(home):
                raise RecognitionUnavailable(
                    "speech recognition model failed verification"
                )
    except RecognitionUnavailable:
        raise
    except Exception as exc:
        raise RecognitionUnavailable(
            f"speech recognition model download failed: {exc}"
        ) from exc
    return {"downloaded": True, "verified": True, "path": str(directory)}


def probe_runtime(home: Path, verify_cache: bool = False) -> dict:
    """Named preflight facts. Never promotes to ``ready``: only a receipt does."""
    reasons: list[str] = []
    facts: dict = {"kind": "gpu_stt",
                   "model": {"repo": GPU_STT_MODEL["repo"],
                             "revision": GPU_STT_MODEL["revision"]},
                   "asset_bytes": {"weights_bytes": GPU_STT_MODEL["weights_bytes"]}}
    try:
        version = importlib.metadata.version("mlx-audio")
    except importlib.metadata.PackageNotFoundError:
        version = None
    facts["mlx_audio"] = version
    if version != MLX_AUDIO_VERSION:
        reasons.append(f"mlx-audio {MLX_AUDIO_VERSION} is required (found {version}); "
                       "install with bash install.sh --voice")
    try:
        import mlx.core as mx
        device = str(mx.default_device())
    except Exception as exc:
        device = None
        reasons.append(f"mlx is not importable: {exc}")
    facts["device"] = device
    if device is not None and "gpu" not in device:
        reasons.append(f"GPU unavailable: default device is {device}")
    present, missing = _file_status(model_dir(home))
    if not present:
        facts["cache"] = "missing"
        reasons.append("speech recognition model not downloaded (missing: "
                       + ", ".join(missing) + "); enable voice in model setup")
    elif verify_cache and not model_verified(home):
        facts["cache"] = "invalid"
        reasons.append("speech recognition model failed sha256 verification")
    else:
        facts["cache"] = "verified" if verify_cache else "present"
    return {"ok": not reasons, "reasons": reasons, "facts": facts}


def resample_to_16k(samples, rate: int):
    """Resample a mono mx.array to 16 kHz on the GPU; returns an mx.array.

    Kaiser-windowed sinc polyphase filter (the same design the ANE path
    uses). The taps are host-designed constants; every sample operation
    runs on the mlx device.
    """
    import math

    import mlx.core as mx

    if rate == TARGET_SAMPLE_RATE:
        return samples
    if rate <= 0:
        raise RecognitionInputError(f"invalid source rate {rate}")
    divisor = math.gcd(int(rate), TARGET_SAMPLE_RATE)
    up, down = TARGET_SAMPLE_RATE // divisor, int(rate) // divisor
    half = 16 * up
    n = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.sinc(n / max(up, down)) * np.kaiser(n.size, 8.0)
    taps = (taps / taps.sum()).astype(np.float32)
    rows = -(-taps.size // up)
    padded = np.zeros(rows * up, dtype=np.float32)
    padded[: taps.size] = taps
    centre = (taps.size - 1) // 2
    count = samples.shape[0]
    n_out = -(-count * up // down)
    signal = mx.pad(samples, [(rows, rows)])
    m = mx.arange(n_out) * down + centre
    phase, base = m % up, m // up
    offsets = mx.arange(rows)
    gathered = mx.take(signal, (base[:, None] - offsets[None, :] + rows).reshape(-1))
    weights = mx.take(mx.array(padded), (offsets[None, :] * up + phase[:, None]).reshape(-1))
    return (gathered.reshape(n_out, rows) * weights.reshape(n_out, rows)).sum(axis=1)


def _receipt_path(home: Path) -> Path:
    return model_dir(home) / "qualification.json"


ACCEPTANCE_THRESHOLDS = {
    "wer": {"test-clean": 0.06, "test-other": 0.14, "accented": 0.20,
            "mixed_0dB": 0.30},
    "empty_rate_min": {"silence": 0.95, "noise": 0.95},
    "latency_p95_ms_max": 2000.0,
}


@functools.cache
def _backend_identity() -> tuple[str | None, str | None]:
    """(identity, detail) of the loaded mlx extension + libmlx.so; fixed per process."""
    from .synthesis import _backend_provenance
    backend = _backend_provenance()
    return (backend.get("identity") if backend.get("verified") == "match" else None,
            backend.get("detail"))


def _runtime_identity(home: Path) -> dict:
    backend, detail = _backend_identity()
    weights = next(e for e in GPU_STT_MODEL["files"] if e["name"] == "model.safetensors")
    return {"mlx_backend": backend, "mlx_backend_detail": detail,
            "model_sha256": weights["sha256"] if model_verified(home) else None}


def _is_finite_number(val) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    try:
        return math.isfinite(val)
    except OverflowError:
        return False


def _reject_non_finite_floats(value) -> None:
    """Reject NaN/Infinity anywhere in a receipt iteratively so server JSON
    encoders (allow_nan=False) cannot fail on forwarded receipt fields."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError(f"acceptance refused: non-finite float {node!r}")
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def _validate_acceptance_metrics(*, wer_by_subset, empty_rate_by_subset,
                                 latency_ms, cpu_tensor_events) -> None:
    if (not isinstance(cpu_tensor_events, int)
            or isinstance(cpu_tensor_events, bool)
            or cpu_tensor_events != 0):
        raise ValueError(f"acceptance refused: cpu_tensor_events={cpu_tensor_events}")
    if (not isinstance(wer_by_subset, dict)
            or not isinstance(empty_rate_by_subset, dict)
            or not isinstance(latency_ms, dict)):
        raise ValueError("acceptance refused: metric sections must be dicts")
    failures = []
    for subset, limit in ACCEPTANCE_THRESHOLDS["wer"].items():
        value = wer_by_subset.get(subset)
        if not _is_finite_number(value) or value < 0 or value > limit:
            failures.append(f"{subset} WER {value} not in [0, {limit}]")
    for subset, floor in ACCEPTANCE_THRESHOLDS["empty_rate_min"].items():
        value = empty_rate_by_subset.get(subset)
        if not _is_finite_number(value) or value < floor or value > 1.0:
            failures.append(f"{subset} empty rate {value} not in [{floor}, 1.0]")
    p95 = latency_ms.get("p95")
    p95_max = ACCEPTANCE_THRESHOLDS["latency_p95_ms_max"]
    p95_ok = _is_finite_number(p95) and 0 <= p95 <= p95_max
    if not p95_ok:
        failures.append(f"latency p95 {p95} ms not in [0, {p95_max}]")
    if "p50" in latency_ms:
        p50 = latency_ms["p50"]
        if not _is_finite_number(p50) or p50 < 0:
            failures.append(f"latency p50 {p50} ms is not a finite nonnegative number")
        elif p95_ok and p50 > p95:
            failures.append(f"latency p50 {p50} ms > p95 {p95} ms")
    if failures:
        raise ValueError("acceptance refused: " + "; ".join(failures))


def read_acceptance_receipt(home: Path) -> dict | None:
    """The receipt, only while it names THIS runtime's backend and model
    and satisfies every frozen acceptance threshold."""
    from .synthesis import _read_state_json

    try:
        receipt = _read_state_json(_receipt_path(home))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(receipt, dict):
        return None
    try:
        _reject_non_finite_floats(receipt)
        _validate_acceptance_metrics(
            wer_by_subset=receipt.get("wer_by_subset"),
            empty_rate_by_subset=receipt.get("empty_rate_by_subset"),
            latency_ms=receipt.get("latency_ms"),
            cpu_tensor_events=receipt.get("cpu_tensor_events"),
        )
    except ValueError:
        return None
    identity = _runtime_identity(home)
    if (identity["mlx_backend"] is None
            or identity["model_sha256"] is None
            or receipt.get("mlx_backend") != identity["mlx_backend"]
            or receipt.get("model_sha256") != identity["model_sha256"]):
        return None
    return receipt


def write_acceptance_receipt(home: Path, *, wer_by_subset: dict,
                             empty_rate_by_subset: dict, latency_ms: dict,
                             cpu_tensor_events: int, receipt_source: str,
                             host: str) -> dict:
    """Record a passed hardware acceptance run for the running runtime.

    Refuses unless every frozen threshold in ACCEPTANCE_THRESHOLDS passes
    and the CPU tensor event count is zero. The backend identity and model
    hash are computed here, never taken from the caller. Stages through a
    unique sibling temp file (0600) and atomically replaces the receipt;
    defending against a hostile co-writer who already has write permission on
    ``home`` is out of scope.
    """
    from .synthesis import _atomic_write_bytes, _check_state_depth, _check_state_size

    _validate_acceptance_metrics(
        wer_by_subset=wer_by_subset,
        empty_rate_by_subset=empty_rate_by_subset,
        latency_ms=latency_ms,
        cpu_tensor_events=cpu_tensor_events,
    )
    identity = _runtime_identity(home)
    if identity["mlx_backend"] is None or identity["model_sha256"] is None:
        raise ValueError("acceptance refused: runtime identity unverified "
                         f"({identity['mlx_backend_detail']})")
    receipt = {
        "schema": "mlx-omarchy/gpu-stt-acceptance/2",
        "recorded_at": time.time(),
        "host": host,
        "model": GPU_STT_MODEL["id"],
        "model_sha256": identity["model_sha256"],
        "mlx_backend": identity["mlx_backend"],
        "wer_by_subset": wer_by_subset,
        "empty_rate_by_subset": empty_rate_by_subset,
        "latency_ms": latency_ms,
        "cpu_tensor_events": cpu_tensor_events,
        "receipt_source": receipt_source,
    }
    _check_state_depth(receipt)
    _reject_non_finite_floats(receipt)
    payload = json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False,
    ).encode("utf-8")
    _check_state_size(len(payload))
    path = _receipt_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, payload)
    return receipt


def _worker_error(response: dict) -> RuntimeError:
    message = response.get("error", "gpu-stt worker failed without a reason")
    if response.get("kind") == "input":
        return RecognitionInputError(message)
    return RecognitionUnavailable(message)


class _WorkerHandle:
    """Owned GPU STT subprocess; model resident, IPC-framed, cancel confirms."""

    def __init__(self, model_dir: Path):
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        # ``python -m mlx_omarchy_assistant.gpu_stt_worker`` needs the directory
        # that holds this package, wherever the assistant was installed.
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (package_root, env.get("PYTHONPATH", "")) if p)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "mlx_omarchy_assistant.gpu_stt_worker",
             "--model-dir", str(model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._pgid = os.getpgid(self._process.pid)
        self._send_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ready_lock = threading.Lock()
        self._ready_seen = False
        self._request_id = 0

    def _drain_stderr(self) -> None:
        proc = self._process
        assert proc.stderr is not None
        try:
            for line in iter(proc.stderr.readline, b""):
                sys.stderr.write("[gpu-stt] " + line.decode("utf-8", "replace"))
        except Exception:
            return

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def _send(self, header: dict, payload: bytes = b"") -> None:
        header = dict(header)
        header["payload_bytes"] = len(payload)
        encoded = json.dumps(header).encode("utf-8")
        with self._send_lock:
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(
                    struct.pack("<Q", len(encoded)) + encoded + payload
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise RecognitionUnavailable(
                    "the gpu-stt worker exited unexpectedly (exit code "
                    f"{self._process.poll()}; its stderr is prefixed [gpu-stt])"
                ) from exc

    def request(self, header: dict, payload: bytes = b"", *,
                timeout: float, cancel=None) -> dict:
        """One framed round trip. Any cancel outcome raises RecognitionCancelled.

        generate() cannot be interrupted, so a cancel kills the worker's
        process group at once and confirms it is gone; the next request
        starts a fresh worker.
        """
        self._request_id += 1
        done = threading.Event()

        def watch_cancel():
            while not done.is_set():
                if cancel.wait(0.05):
                    if not done.is_set():
                        self.confirm_stopped()
                    return

        watcher = None
        if cancel is not None:
            watcher = threading.Thread(target=watch_cancel, daemon=True)
            watcher.start()
        try:
            with self._ready_lock:
                # The boot "ready" frame must be consumed before the first
                # request; reading it as a response shifted every transcript.
                if not self._ready_seen:
                    ready = self._read_response(max(timeout, 120.0))
                    if ready.get("event") != "ready":
                        raise _worker_error(ready)
                    self._ready_seen = True
            self._send(dict(header, id=self._request_id), payload)
            response = self._read_response(timeout)
        except (RecognitionUnavailable, RecognitionTimedOut) as exc:
            if cancel is not None and cancel.is_set():
                raise RecognitionCancelled("cancelled during transcription") from exc
            raise
        finally:
            done.set()
            if watcher is not None:
                watcher.join(timeout=CANCEL_GRACE_SECONDS + 1.0)
        if cancel is not None and cancel.is_set():
            raise RecognitionCancelled("cancelled during transcription")
        if not response.get("ok"):
            raise _worker_error(response)
        return response

    def _read_response(self, timeout: float) -> dict:
        process = self._process
        assert process.stdout is not None
        deadline = time.monotonic() + timeout
        raw_header = b""
        while len(raw_header) < 8:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.alive:
                break
            byte = process.stdout.read(8 - len(raw_header))
            if byte:
                raw_header += byte
        if len(raw_header) < 8:
            self.confirm_stopped()
            if process.poll() not in (None, 0):
                raise RecognitionUnavailable(
                    "the gpu-stt worker exited while a request was in "
                    f"flight (exit code {process.poll()})"
                )
            raise RecognitionTimedOut(
                f"the gpu-stt worker did not answer within {timeout:.0f} s "
                "and was stopped (exit confirmed)"
            )
        (header_size,) = struct.unpack("<Q", raw_header)
        header_bytes = process.stdout.read(header_size)
        response = json.loads(header_bytes.decode("utf-8"))
        payload_size = int(response.get("payload_bytes", 0))
        if payload_size:
            process.stdout.read(payload_size)
        return response

    def _group_alive(self) -> bool:
        try:
            os.killpg(self._pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _wait_group(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._process.poll()
            if not self._group_alive():
                return True
            time.sleep(0.05)
        self._process.poll()
        return not self._group_alive()

    def confirm_stopped(self, grace: float = 0.0) -> None:
        with self._stop_lock:
            process = self._process
            if process.poll() is None and grace > 0:
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
            if self._group_alive():
                try:
                    os.killpg(self._pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                self._wait_group(2.0)
            if self._group_alive():
                try:
                    os.killpg(self._pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._wait_group(5.0)
            if process.poll() is None:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None or self._group_alive():
                raise RecognitionUnavailable(
                    "the gpu-stt worker process group could not be "
                    "confirmed stopped"
                )
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def shutdown(self) -> None:
        if self.alive:
            try:
                self._send({"op": "close"})
            except RecognitionUnavailable:
                pass
        self.confirm_stopped(grace=CANCEL_GRACE_SECONDS)

    def kill(self) -> None:
        self.confirm_stopped()