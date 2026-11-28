"""Export a precomputed model-fit table for static sites (joshuawarren.com/models).

Every number here comes from budget.estimate_required() or catalog fields —
this module adds no new budgeting math. For each catalog model and each
common Apple Silicon unified-memory size it records:

  max_context          the largest context (<= the model's max_tokens) whose
                       estimate_required total fits in N GiB minus the
                       2 GiB safety reserve
  fits                 the same gates the serve CLI applies at the model's
                       default context (resolve_context(ctx, None)):
                       default-context estimate fits in N GiB minus the
                       2 GiB safety reserve, AND capability.min_mem_gib <= N
  peak_estimate_bytes  estimate_required total at max_context when the model
                       fits; otherwise the total at its default context,
                       i.e. what running it out of the box would need

Binary search is valid because estimate_required is monotonic non-decreasing
in context_tokens.

Usage (from serve/):  python3 -m mlx_omarchy_serve.export_fit_table [-o out.json]
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import secrets
import stat
import sys
from pathlib import Path

from . import budget
from .catalog import CatalogError, bundled_path, validate_file

MEMORY_SIZES_GIB = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192]
_LINE_BREAK_ESCAPES = str.maketrans(
    {c: repr(c)[1:-1] for c in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"}
)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_PATH_SEPS = tuple(s for s in (os.sep, os.altsep) if s)
_TMP_ATTEMPTS = 16


def _max_fitting_context(memory: dict, max_tokens: int, available_bytes: int) -> int:
    if budget.estimate_required(memory, max_tokens).total <= available_bytes:
        return max_tokens
    if budget.estimate_required(memory, 0).total > available_bytes:
        return 0
    lo, hi = 0, max_tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if budget.estimate_required(memory, mid).total <= available_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _fit_cell(entry: dict, gib: int) -> dict:
    available = gib * budget.GiB - budget.SAFETY_RESERVE_BYTES
    max_tokens = entry["context"]["max_tokens"] or budget.DEFAULT_CONTEXT_TOKENS
    default_context = budget.resolve_context(entry["context"], None)
    max_context = _max_fitting_context(entry["memory"], max_tokens, available)
    floor_gib = entry["capability"]["min_mem_gib"]
    fits = (
        max_context >= default_context
        and (floor_gib is None or floor_gib <= gib)
    )
    peak_context = max_context if fits else default_context
    peak = budget.estimate_required(entry["memory"], peak_context).total
    return {"fits": fits, "max_context": max_context, "peak_estimate_bytes": peak}


def _model_row(entry: dict) -> dict:
    quant = entry["quant"]
    return {
        "id": entry["id"],
        "label": entry["repo"].split("/", 1)[1],
        "family": entry["family"],
        "kind": entry["kind"],
        "license": entry["license"],
        "repo": entry["repo"],
        "revision": entry["revision"],
        "recommended": entry["recommended"],
        "trained_tokens": entry["context"]["trained_tokens"],  # the horizon: null means unknown, said out loud
        "generation_status": entry["qualification"]["generation"]["status"],
        "quant": None if quant is None else {
            "bits": quant["bits"], "mode": quant["mode"], "group_size": quant["group_size"],
        },
        "weights_bytes": entry["memory"]["weights_bytes"],
        "kv_bytes_per_token": entry["memory"]["kv_bytes_per_token"],
        "max_tokens": entry["context"]["max_tokens"],
        "min_mem_gib": entry["capability"]["min_mem_gib"],
        "fits": {str(gib): _fit_cell(entry, gib) for gib in MEMORY_SIZES_GIB},
    }


def build_fit_table(catalog_path: Path | None = None) -> dict:
    path = catalog_path or bundled_path()
    try:
        cat = validate_file(path)
    except OSError as exc:
        raise CatalogError(f"{path}: {exc.strerror or exc}") from exc
    # No generated timestamp here on purpose: the site's sync test re-runs
    # this export and deep-compares it, so it must be byte-reproducible.
    return {
        "catalog": {"version": cat["version"], "generated_at": cat["generated_at"], "source": cat["source"]},
        "budget": {
            "safety_reserve_bytes": budget.SAFETY_RESERVE_BYTES,
            "min_usable_context_tokens": budget.DEFAULT_CONTEXT_TOKENS,
            "workspace_fraction": budget.WORKSPACE_FRACTION,
            "workspace_min_bytes": budget.WORKSPACE_MIN_BYTES,
            "unknown_kv_margin_bytes": budget.UNKNOWN_KV_MARGIN,
        },
        "memory_sizes_gib": MEMORY_SIZES_GIB,
        "models": [_model_row(e) for e in cat["models"]],
    }


_PathInput = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def _io_error(exc: OSError, path: str) -> OSError:
    if exc.filename == path and exc.filename2 is None:
        return exc
    if exc.errno is not None:
        return OSError(exc.errno, exc.strerror or str(exc), path)
    return OSError(f"{path}: {exc}")


def _validate_out_path(out: _PathInput) -> tuple[str, str, str]:
    raw = os.fsdecode(out)
    if not raw:
        raise OSError(errno.EINVAL, "empty output path", raw)
    if "\x00" in raw:
        raise OSError(errno.EINVAL, "output path contains NUL byte", raw)
    try:
        os.fsencode(raw)
    except UnicodeEncodeError as exc:
        raise OSError(
            errno.EINVAL, "output path is not filesystem-encodable", raw
        ) from exc
    if raw.endswith(_PATH_SEPS):
        raise OSError(
            errno.EINVAL, "output path must name a file, not a directory", raw
        )
    parent, name = os.path.split(raw)
    if name in (".", ".."):
        raise OSError(
            errno.EINVAL, "output path must name a file, not a directory", raw
        )
    return raw, parent or ".", name


def _write_text_atomic(out: _PathInput, text: str) -> None:
    """Write UTF-8 text to `out` via a unique sibling temp file in a pinned dir_fd.

    Validates the raw path before opening the parent directory, refuses any
    existing non-regular destination (symlinks, directories, FIFOs, sockets,
    and devices) via lstat(dir_fd=...) before staging, stages all bytes into
    a fresh O_CREAT | O_EXCL | O_NOFOLLOW sibling temp file (mode 0o666 masked
    by the process umask, matching Path.write_text), fsyncs and closes the
    descriptor before atomic os.replace(..., src_dir_fd, dst_dir_fd), and
    performs best-effort cleanup of only the temp file it created on error.

    Behavioral differences from in-place Path.write_text:
    - Opening the parent directory and staging a sibling temp file requires
      read, write, and execute permission on the parent directory (a writable
      file inside a read-only directory or owned by another user in a sticky
      directory cannot be replaced atomically).
    - Special paths such as /dev/stdout or /dev/fd/1 are refused as symlinks
      or non-regular targets; omit -o/--out to write to stdout.
    - Replacing an existing regular file does not preserve its prior inode,
      permissions, owner, or hard links, and the parent directory is not
      fsynced after os.replace.
    - Non-goal: defending against a hostile co-writer who already has write
      permission on the output directory (such a writer can replace or unlink
      the output file directly after os.replace).
    """
    raw, parent, name = _validate_out_path(out)
    data = text.encode("utf-8")
    try:
        dir_fd = os.open(
            parent, os.O_RDONLY | _O_DIRECTORY | _O_NONBLOCK | _O_CLOEXEC
        )
    except OSError as exc:
        raise _io_error(exc, raw) from exc

    try:
        try:
            dst_st = os.lstat(name, dir_fd=dir_fd)
        except FileNotFoundError:
            dst_st = None
        except OSError as exc:
            raise _io_error(exc, raw) from exc

        if dst_st is not None:
            if stat.S_ISDIR(dst_st.st_mode):
                raise IsADirectoryError(
                    errno.EISDIR, os.strerror(errno.EISDIR), raw
                )
            if stat.S_ISLNK(dst_st.st_mode):
                raise OSError(errno.ELOOP, "refusing to write to symlink", raw)
            if not stat.S_ISREG(dst_st.st_mode):
                raise OSError(
                    errno.EINVAL, "refusing to write to non-regular file", raw
                )

        open_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
        )
        fd: int | None = None
        tmp_name = ""
        for _ in range(_TMP_ATTEMPTS):
            candidate = f".export_fit_table.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(candidate, open_flags, 0o666, dir_fd=dir_fd)
            except FileExistsError:
                continue
            except OSError as exc:
                raise _io_error(exc, raw) from exc
            tmp_name = candidate
            break
        if fd is None:
            raise OSError(
                errno.EEXIST,
                "could not create unique temporary output file",
                raw,
            )

        try:
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError(
                        errno.EINVAL, "staged temporary file is not regular", raw
                    )
                view = memoryview(data)
                written = 0
                while written < len(view):
                    n = os.write(fd, view[written:])
                    if n <= 0:
                        raise OSError(errno.EIO, "short write", raw)
                    written += n
                os.fsync(fd)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            os.close(fd)
            os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException as exc:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name, dir_fd=dir_fd)
            if isinstance(exc, OSError):
                mapped = _io_error(exc, raw)
                if mapped is not exc:
                    raise mapped from exc
            raise
    finally:
        with contextlib.suppress(OSError):
            os.close(dir_fd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", type=Path, default=None, help="catalog.json path (default: bundled)")
    ap.add_argument("-o", "--out", type=str, default=None, help="write JSON here (default: stdout)")
    args = ap.parse_args(argv)
    try:
        table = build_fit_table(args.catalog)
        try:
            text = json.dumps(table, indent=2, sort_keys=False, allow_nan=False) + "\n"
        except ValueError as exc:
            raise CatalogError(f"non-JSON-compliant fit table: {exc}") from exc
        if args.out is not None:
            _write_text_atomic(args.out, text)
        elif sys.stdout is None:
            raise OSError(errno.EBADF, "stdout is closed")
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
    except (CatalogError, OSError) as exc:
        if sys.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                print(
                    f"error: {str(exc).translate(_LINE_BREAK_ESCAPES)}",
                    file=sys.stderr,
                )
        return 1
    return 0


if __name__ == "__main__":
    status = main()
    if status:
        # Redirect stdout/stderr to devnull only when a stream still holds a
        # failed write; exit-time flush would retry it and turn exit 1 into 120
        # (Python docs: signal module, "Note on SIGPIPE").
        for stream in (sys.stdout, sys.stderr):
            if stream is None:
                continue
            try:
                stream.flush()
            except OSError:
                devnull = os.open(os.devnull, os.O_WRONLY)
                try:
                    with contextlib.suppress(OSError, ValueError):
                        os.dup2(devnull, stream.fileno())
                finally:
                    os.close(devnull)
            except ValueError:
                pass
    raise SystemExit(status)
