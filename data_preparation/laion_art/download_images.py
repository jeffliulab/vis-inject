"""
Robust LAION-Art image downloader -- no img2dataset dependency.

Downloads images from LAION-Art parquet metadata (128 shards) using
only standard Python libraries + pyarrow + Pillow. Designed for HPC
environments where installing img2dataset is problematic.

Key design decisions based on Tufts HPC constraints:
  - Parquet metadata must be pre-downloaded on login node (compute
    nodes cannot resolve cdn-lfs.huggingface.co)
  - Supports 128-shard parquet layout from laion/laion-art
  - Image URLs point to external hosts (not HF CDN), so compute
    nodes CAN download them directly

Usage:
    python download_images.py --test-run                       # 100 images, verify setup
    python download_images.py --workers 32                     # full download
    python download_images.py --resume --workers 32            # resume interrupted download
    python download_images.py --output-format folder           # save as individual files
    python download_images.py --start-shard 50 --end-shard 100 # specific parquet shard range
"""

import argparse
import datetime
import io
import json
import logging
import os
import platform
import shutil
import signal
import socket
import sys
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """
    Logger writing to:
      1. Console -- INFO level, compact timestamps
      2. Main log file -- DEBUG level, full context
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("laion_downloader")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-5s  %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    main_log = os.path.join(log_dir, f"download_{run_id}.log")
    fh = logging.FileHandler(main_log, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(funcName)-22s | %(message)s"
    ))
    logger.addHandler(fh)

    logger.info(f"Log file: {main_log}")
    return logger


def setup_fail_log(log_dir: str, run_id: str) -> logging.Logger:
    """Separate logger that records every failed URL with its error."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fail_logger = logging.getLogger("laion_fail")
    fail_logger.setLevel(logging.DEBUG)
    fail_logger.handlers.clear()
    fail_logger.propagate = False

    fail_log = os.path.join(log_dir, f"failed_urls_{run_id}.log")
    fh = logging.FileHandler(fail_log, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    fail_logger.addHandler(fh)

    logging.getLogger("laion_downloader").info(f"Failed-URL log: {fail_log}")
    return fail_logger

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorStats:
    """Categorizes and counts download errors for diagnostics."""

    CATEGORIES = [
        "timeout", "dns_failure", "connection_refused", "connection_reset",
        "ssl_error", "http_404", "http_403", "http_other",
        "invalid_url", "empty_response", "image_corrupt", "image_too_small",
        "pillow_error", "unknown",
    ]

    def __init__(self):
        self.counts = {cat: 0 for cat in self.CATEGORIES}
        self.recent_errors: list = []
        self._max_recent = 50

    def record(self, category: str, url: str = "", detail: str = ""):
        if category not in self.counts:
            category = "unknown"
        self.counts[category] += 1
        entry = {
            "time": datetime.datetime.now().isoformat(),
            "category": category,
            "url": url[:200],
            "detail": detail[:300],
        }
        self.recent_errors.append(entry)
        if len(self.recent_errors) > self._max_recent:
            self.recent_errors.pop(0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary_lines(self) -> list:
        lines = ["Error breakdown:"]
        for cat in self.CATEGORIES:
            cnt = self.counts[cat]
            if cnt > 0:
                lines.append(f"  {cat:25s}: {cnt:>8,}")
        lines.append(f"  {'TOTAL':25s}: {self.total:>8,}")
        return lines

    def to_dict(self) -> dict:
        return {"counts": self.counts, "recent_errors": self.recent_errors}

# ---------------------------------------------------------------------------
# System diagnostics
# ---------------------------------------------------------------------------

def log_system_info(logger: logging.Logger, output_dir: str):
    logger.info("=" * 60)
    logger.info("SYSTEM DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"Hostname   : {socket.gethostname()}")
    logger.info(f"Platform   : {platform.platform()}")
    logger.info(f"Python     : {sys.version}")
    logger.info(f"Executable : {sys.executable}")
    logger.info(f"PID        : {os.getpid()}")
    logger.info(f"CWD        : {os.getcwd()}")
    logger.info(f"Timestamp  : {datetime.datetime.now().isoformat()}")

    slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM")}
    if slurm_vars:
        logger.info("SLURM environment:")
        for k in sorted(slurm_vars):
            logger.info(f"  {k} = {slurm_vars[k]}")
    else:
        logger.info("SLURM: not detected")

    try:
        target = output_dir if os.path.exists(output_dir) else os.path.expanduser("~")
        usage = shutil.disk_usage(target)
        logger.info(f"Disk total : {usage.total / (1024**3):.1f} GB")
        logger.info(f"Disk free  : {usage.free / (1024**3):.1f} GB")
        logger.info(f"Disk used  : {usage.used / (1024**3):.1f} GB "
                     f"({usage.used / usage.total * 100:.1f}%)")
    except Exception as e:
        logger.warning(f"Disk check failed: {e}")

    logger.info("Network tests:")
    for host, port in [("huggingface.co", 443), ("cdn-lfs.huggingface.co", 443),
                       ("8.8.8.8", 53)]:
        try:
            t0 = time.time()
            s = socket.create_connection((host, port), timeout=5)
            s.close()
            logger.info(f"  {host}:{port}  OK  ({(time.time()-t0)*1000:.0f}ms)")
        except Exception as e:
            logger.warning(f"  {host}:{port}  FAIL  ({type(e).__name__}: {e})")

    logger.info("Python packages:")
    for pkg in ["pyarrow", "PIL", "pandas", "numpy"]:
        try:
            mod = __import__(pkg)
            logger.info(f"  {pkg:12s}: {getattr(mod, '__version__', '?')}")
        except ImportError:
            logger.warning(f"  {pkg:12s}: NOT INSTALLED")
    logger.info("=" * 60)


def log_config(logger: logging.Logger, args):
    logger.info("CONFIGURATION")
    logger.info("-" * 40)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k:20s}: {v}")
    logger.info("-" * 40)

# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def classify_error(e: Exception) -> str:
    msg = str(e).lower()
    if isinstance(e, HTTPError):
        code = e.code
        if code == 404: return "http_404"
        if code == 403: return "http_403"
        return "http_other"
    if isinstance(e, TimeoutError) or "timed out" in msg:
        return "timeout"
    if "ssl" in msg or "certificate" in msg:
        return "ssl_error"
    if "name or service not known" in msg or "getaddrinfo" in msg:
        return "dns_failure"
    if "connection refused" in msg:
        return "connection_refused"
    if "connection reset" in msg:
        return "connection_reset"
    return "unknown"


def download_single_image(url: str, timeout: int = 5,
                          max_retries: int = 1) -> Tuple[Optional[bytes], str]:
    """Returns (image_bytes, error_category). error_category is "" on success."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LAION-Downloader/2.0)"}

    if not url or not url.startswith("http"):
        return None, "invalid_url"

    last_err = "unknown"
    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None, "http_other"
                data = resp.read()
                if len(data) < 100:
                    return None, "empty_response"
                return data, ""
        except (URLError, HTTPError, TimeoutError, OSError,
                ConnectionError, ValueError) as e:
            last_err = classify_error(e)
            if last_err in ("http_404", "http_403", "dns_failure",
                            "connection_refused", "invalid_url"):
                return None, last_err
            if attempt < max_retries:
                time.sleep(0.3)
        except Exception:
            return None, "unknown"
    return None, last_err


def resize_and_crop(image_bytes: bytes, target_size: int,
                    quality: int = 95) -> Tuple[Optional[bytes], str]:
    """Returns (jpeg_bytes, error_category)."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        if w < 10 or h < 10:
            return None, "image_too_small"

        scale = target_size / min(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - target_size) // 2
        top = (nh - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), ""
    except Exception as e:
        msg = str(e).lower()
        if "cannot identify" in msg or "truncated" in msg:
            return None, "image_corrupt"
        return None, "pillow_error"


def process_row(row: dict, target_size: int, timeout: int) -> dict:
    """Download and process one row. Always returns a result dict."""
    url = row.get("URL", "")
    image_bytes, dl_err = download_single_image(url, timeout=timeout)
    if image_bytes is None:
        return {"success": False, "url": url, "error": dl_err}

    processed, proc_err = resize_and_crop(image_bytes, target_size)
    if processed is None:
        return {"success": False, "url": url, "error": proc_err,
                "raw_size": len(image_bytes)}

    return {
        "success": True, "url": url, "error": "",
        "image_bytes": processed,
        "caption": row.get("TEXT", ""),
        "metadata": {
            "similarity": row.get("similarity"),
            "aesthetic": row.get("aesthetic"),
            "punsafe": row.get("punsafe"),
            "pwatermark": row.get("pwatermark"),
            "hash": row.get("hash"),
        },
    }

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes downloaded images into WebDataset .tar shards."""

    def __init__(self, output_dir: str, shard_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.current_shard_id = -1
        self.current_count = 0
        self.current_tar = None
        self.total_written = 0

    def _open_shard(self, shard_id: int):
        if self.current_tar is not None:
            self.current_tar.close()
        path = self.output_dir / f"{shard_id:05d}.tar"

        # Resume protection: skip shards that are already complete.
        # Without this, tarfile.open(path, "w") would truncate existing
        # shards, destroying all previously downloaded images in that shard.
        if path.exists() and path.stat().st_size > 1_000_000:
            try:
                with tarfile.open(str(path), "r") as existing:
                    member_count = len(existing.getmembers())
                expected = self.shard_size * 3  # jpg + txt + json per image
                if member_count >= expected * 0.9:
                    logging.getLogger("laion_downloader").info(
                        f"Shard {shard_id:05d} already complete "
                        f"({member_count} members), skipping"
                    )
                    self.current_tar = None
                    self.current_shard_id = shard_id
                    self.current_count = self.shard_size
                    return
            except tarfile.TarError:
                pass  # corrupted shard, overwrite it

        self.current_tar = tarfile.open(str(path), "w")
        self.current_shard_id = shard_id
        self.current_count = 0

    def _add_bytes(self, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        self.current_tar.addfile(info, io.BytesIO(data))

    def write(self, sample: dict, global_idx: int):
        target_shard = global_idx // self.shard_size
        if self.current_tar is None or target_shard != self.current_shard_id:
            self._open_shard(target_shard)

        # Skip writes for already-complete shards
        if self.current_tar is None:
            return

        prefix = f"{global_idx:09d}"
        self._add_bytes(f"{prefix}.jpg", sample["image_bytes"])
        self._add_bytes(f"{prefix}.txt", sample["caption"].encode("utf-8"))
        meta_json = json.dumps(sample["metadata"], ensure_ascii=False)
        self._add_bytes(f"{prefix}.json", meta_json.encode("utf-8"))
        self.current_count += 1
        self.total_written += 1

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None


class FolderWriter:
    """Writes downloaded images as individual files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "captions").mkdir(parents=True, exist_ok=True)
        self.total_written = 0

    def write(self, sample: dict, global_idx: int):
        prefix = f"{global_idx:09d}"
        with open(self.output_dir / "images" / f"{prefix}.jpg", "wb") as f:
            f.write(sample["image_bytes"])
        with open(self.output_dir / "captions" / f"{prefix}.txt", "w",
                  encoding="utf-8") as f:
            f.write(sample["caption"])
        self.total_written += 1

    def close(self):
        pass

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks download progress with resume support."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = {
            "completed_rows": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "session_history": [],
        }
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.state.update(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass

    def update(self, success: bool, image_size: int = 0):
        self.state["completed_rows"] += 1
        if success:
            self.state["successful_downloads"] += 1
            self.state["total_bytes"] += image_size
        else:
            self.state["failed_downloads"] += 1
        self.state["last_update"] = time.time()

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_session(self, info: dict):
        self.state["session_history"].append({
            "started": datetime.datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            **info,
        })

    @property
    def completed(self) -> int:
        return self.state["completed_rows"]

    @property
    def successful(self) -> int:
        return self.state["successful_downloads"]

    @property
    def failed(self) -> int:
        return self.state["failed_downloads"]

    def format_progress(self, total_rows: int) -> str:
        elapsed = time.time() - self.state["start_time"]
        rate = self.completed / max(elapsed, 1)
        remaining = (total_rows - self.completed) / max(rate, 0.01)
        pct = self.completed / max(total_rows, 1) * 100
        succ_rate = self.successful / max(self.completed, 1) * 100
        size_gb = self.state["total_bytes"] / (1024 ** 3)
        return (
            f"[{pct:5.1f}%] {self.completed:,}/{total_rows:,} | "
            f"OK {self.successful:,} | Fail {self.failed:,} | "
            f"Rate {succ_rate:.0f}% | "
            f"{size_gb:.2f} GB | {rate:.1f} img/s | "
            f"ETA {remaining/3600:.1f}h"
        )

# ---------------------------------------------------------------------------
# Parquet loading  (multi-shard)
# ---------------------------------------------------------------------------

def find_parquet_files(parquet_dir: str, logger: logging.Logger) -> List[str]:
    """
    Find all parquet shard files in a directory.
    Supports the LAION-Art naming: part-XXXXX-*.snappy.parquet
    """
    pdir = Path(parquet_dir)
    if not pdir.is_dir():
        logger.error(f"Parquet directory does not exist: {pdir}")
        sys.exit(1)

    patterns = ["*.parquet", "*.snappy.parquet"]
    files = set()
    for pat in patterns:
        for p in pdir.glob(pat):
            if not p.name.startswith("."):
                files.add(p)

    sorted_files = sorted(files, key=lambda p: p.name)
    logger.info(f"Found {len(sorted_files)} parquet files in {pdir}")
    if sorted_files:
        logger.info(f"  First: {sorted_files[0].name}")
        logger.info(f"  Last : {sorted_files[-1].name}")
    return [str(f) for f in sorted_files]


def load_parquet_rows(parquet_files: List[str],
                      start_idx: int, end_idx: Optional[int],
                      logger: logging.Logger) -> list:
    """Load rows from multiple parquet files, returning a combined list of dicts."""
    import pyarrow.parquet as pq

    all_rows = []
    for i, pf in enumerate(parquet_files):
        table = pq.read_table(pf)
        n = table.num_rows
        logger.debug(f"  Shard {i:03d}: {Path(pf).name}  ({n:,} rows)")
        df = table.to_pandas()
        all_rows.extend(df.to_dict("records"))

    total = len(all_rows)
    logger.info(f"Total rows across all shards: {total:,}")

    if "URL" in (all_rows[0] if all_rows else {}):
        logger.debug(f"  Sample URL: {all_rows[0]['URL'][:120]}")

    # Apply range selection
    effective_end = end_idx if end_idx is not None else total
    effective_end = min(effective_end, total)
    effective_start = min(start_idx, total)
    selected = all_rows[effective_start:effective_end]

    logger.info(f"Selected rows [{effective_start}:{effective_end}] = {len(selected):,}")
    return selected

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = False

def _sig_handler(signum, frame):
    global _shutdown
    name = signal.Signals(signum).name
    logging.getLogger("laion_downloader").warning(
        f"Received {name} -- finishing current batch then exiting"
    )
    _shutdown = True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    global _shutdown

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")

    logger = setup_logging(log_dir, run_id)
    fail_logger = setup_fail_log(log_dir, run_id)
    error_stats = ErrorStats()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    logger.info("=" * 60)
    logger.info("LAION-Art Image Downloader v2.1")
    logger.info("=" * 60)

    log_system_info(logger, args.output_dir)
    log_config(logger, args)

    # --- Find and load parquet files ---
    parquet_files = find_parquet_files(args.parquet_dir, logger)
    if not parquet_files:
        logger.critical(f"No parquet files found in {args.parquet_dir}")
        logger.critical("Download them first on the login node. See README.md.")
        sys.exit(1)

    # Optionally restrict to a range of parquet shards
    if args.start_shard is not None or args.end_shard is not None:
        s = args.start_shard or 0
        e = args.end_shard or len(parquet_files)
        parquet_files = parquet_files[s:e]
        logger.info(f"Using parquet shards [{s}:{e}] ({len(parquet_files)} files)")

    # Determine row range
    start_row = 0
    end_row = None
    if args.test_run:
        end_row = args.test_count
        logger.info(f"[TEST RUN] Limiting to {args.test_count} rows")

    logger.info("Loading parquet data into memory...")
    rows = load_parquet_rows(parquet_files, start_row, end_row, logger)
    total = len(rows)

    if total == 0:
        logger.error("No rows to process.")
        return

    # --- Writer ---
    if args.output_format == "webdataset":
        writer = ShardWriter(args.output_dir, shard_size=args.shard_size)
    else:
        writer = FolderWriter(args.output_dir)

    # --- Progress tracker ---
    state_file = os.path.join(args.output_dir, ".download_state.json")
    tracker = ProgressTracker(state_file)

    skip = 0
    if args.resume and tracker.completed > 0:
        skip = tracker.completed
        logger.info(f"[RESUME] Skipping {skip:,} already-processed rows")
        rows = rows[skip:]
        total = len(rows)
        logger.info(f"  Remaining: {total:,}")

    tracker.record_session({
        "mode": "resume" if args.resume else ("test" if args.test_run else "full"),
        "total_rows": total, "skip": skip,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
    })
    tracker.save()

    logger.info("")
    logger.info(f"Starting download: {total:,} rows, {args.workers} workers")
    logger.info("")

    t0 = time.time()
    batch_size = args.workers * 8
    global_idx = skip
    batches_done = 0
    last_snapshot = t0

    for batch_start in range(0, total, batch_size):
        if _shutdown:
            logger.warning("Shutdown requested. Saving state and exiting.")
            break

        batch = rows[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, row in enumerate(batch):
                idx = global_idx + i
                fut = executor.submit(process_row, row, args.image_size, args.timeout)
                futures[fut] = (idx, row)

            for fut in as_completed(futures):
                idx, row = futures[fut]
                try:
                    result = fut.result()
                    if result["success"]:
                        writer.write(result, idx)
                        tracker.update(True, len(result["image_bytes"]))
                    else:
                        tracker.update(False)
                        error_stats.record(
                            result["error"], url=result.get("url", ""),
                            detail=f"raw_size={result.get('raw_size', 'N/A')}"
                        )
                        fail_logger.info(
                            f"{result['error']:20s} | idx={idx} | "
                            f"{result.get('url', '')}"
                        )
                except Exception as e:
                    tracker.update(False)
                    error_stats.record("unknown", url=row.get("URL", ""),
                                       detail=f"{type(e).__name__}: {e}")
                    fail_logger.info(
                        f"{'exception':20s} | idx={idx} | "
                        f"{row.get('URL', '')} | {type(e).__name__}: {e}"
                    )
                    logger.debug(f"Exception at idx={idx}: {traceback.format_exc()}")

        global_idx += len(batch)
        batches_done += 1

        if batches_done % 3 == 0 or batch_start + batch_size >= total:
            logger.info(tracker.format_progress(total + skip))

        now = time.time()
        if now - last_snapshot > 60:
            tracker.save()
            _save_snapshot(log_dir, run_id, error_stats, tracker)
            last_snapshot = now

    writer.close()
    tracker.save()
    _save_snapshot(log_dir, run_id, error_stats, tracker)

    elapsed = time.time() - t0

    # --- Final report ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD REPORT")
    logger.info("=" * 60)
    logger.info(f"  Finished       : {datetime.datetime.now().isoformat()}")
    logger.info(f"  Elapsed        : {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    logger.info(f"  Rows processed : {tracker.completed:,}")
    logger.info(f"  Images saved   : {tracker.successful:,}")
    logger.info(f"  Failed         : {tracker.failed:,}")
    sr = tracker.successful / max(tracker.completed, 1) * 100
    logger.info(f"  Success rate   : {sr:.1f}%")
    logger.info(f"  Total size     : {tracker.state['total_bytes']/(1024**3):.2f} GB")
    spd = tracker.completed / max(elapsed, 1)
    logger.info(f"  Avg speed      : {spd:.1f} img/s")
    logger.info(f"  Output         : {args.output_dir}")
    if _shutdown:
        logger.info("  ** Stopped early (signal). Use --resume to continue. **")
    logger.info("")
    for line in error_stats.summary_lines():
        logger.info(line)

    try:
        u = shutil.disk_usage(args.output_dir)
        logger.info(f"\n  Disk free now : {u.free/(1024**3):.1f} GB")
    except Exception:
        pass

    logger.info("")
    logger.info("FILES FOR DEBUGGING (share these if you need help):")
    logger.info(f"  Main log       : {log_dir}/download_{run_id}.log")
    logger.info(f"  Failed URLs    : {log_dir}/failed_urls_{run_id}.log")
    logger.info(f"  Error snapshot : {log_dir}/error_snapshot_{run_id}.json")
    logger.info(f"  State file     : {state_file}")
    logger.info("=" * 60)


def _save_snapshot(log_dir: str, run_id: str,
                   error_stats: ErrorStats, tracker: ProgressTracker):
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "progress": {
            "completed": tracker.completed,
            "successful": tracker.successful,
            "failed": tracker.failed,
            "total_bytes": tracker.state["total_bytes"],
        },
        "errors": error_stats.to_dict(),
    }
    path = os.path.join(log_dir, f"error_snapshot_{run_id}.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust LAION-Art image downloader (no img2dataset)"
    )

    parser.add_argument("--parquet-dir", type=str,
        default="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/metadata",
        help="Directory containing parquet shard files (pre-downloaded)")
    parser.add_argument("--output-dir", type=str,
        default="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset")
    parser.add_argument("--output-format", type=str,
        choices=["webdataset", "folder"], default="webdataset")
    parser.add_argument("--log-dir", type=str, default=None,
        help="Log directory (defaults to <output-dir>/logs)")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encode-quality", type=int, default=95)

    parser.add_argument("--workers", type=int, default=64,
        help="Parallel download threads")
    parser.add_argument("--timeout", type=int, default=5,
        help="Per-image download timeout (seconds)")
    parser.add_argument("--shard-size", type=int, default=10000,
        help="Images per WebDataset .tar shard")

    parser.add_argument("--resume", action="store_true",
        help="Resume from last checkpoint")
    parser.add_argument("--start-shard", type=int, default=None,
        help="Start from this parquet shard index (0-127)")
    parser.add_argument("--end-shard", type=int, default=None,
        help="End at this parquet shard index")

    parser.add_argument("--test-run", action="store_true",
        help="Download only a small batch to verify setup")
    parser.add_argument("--test-count", type=int, default=100,
        help="Number of images in test mode")

    main(parser.parse_args())
