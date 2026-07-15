"""
Extract ADNI_definitivo_2_CN_.zip to D:\ADNI_definitivo_2_CN\
- Skips already-extracted files (safe to resume after interruption)
- Shows live progress: entries, %, speed, ETA, free space
- After full extraction, deletes the source zip to free C: space
"""

import zipfile
import os
import sys
import time
import shutil
from pathlib import Path

ZIP_PATH  = r"C:\Users\user\Desktop\2026.AD_MotionCorrection\ADNI_definitivo_2_CN_.zip"
OUT_DIR   = r"D:\ADNI_definitivo_2_CN"
LOG_PATH  = r"C:\Users\user\Desktop\2026.AD_MotionCorrection\extract_ADNI_CN_progress.log"

REPORT_EVERY = 5000   # print a status line every N entries
FLUSH_EVERY  = 50000  # flush log every N entries


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def fmt_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds)//60}m{int(seconds)%60:02d}s"
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    return f"{h}h{m:02d}m"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Opening zip …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        all_entries = zf.infolist()

    total = len(all_entries)
    print(f"Total entries : {total:,}")
    print(f"Destination   : {OUT_DIR}")
    print(f"Free on D:    : {free_gb('D:'):.1f} GB")
    print(f"Free on C:    : {free_gb('C:'):.1f} GB")
    print()

    # Count already-extracted files for resume
    already = 0
    for info in all_entries:
        dest = os.path.join(OUT_DIR, info.filename.replace("/", os.sep))
        if not info.filename.endswith("/") and os.path.exists(dest):
            already += 1

    if already:
        print(f"Resuming: {already:,} files already extracted, skipping them.\n")

    log = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    log.write(f"\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    done      = 0
    skipped   = 0
    errors    = 0
    t_start   = time.time()
    t_last    = t_start

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for i, info in enumerate(all_entries):
            dest = os.path.join(OUT_DIR, info.filename.replace("/", os.sep))

            # Skip directories
            if info.filename.endswith("/"):
                os.makedirs(dest, exist_ok=True)
                continue

            # Resume: skip already done
            if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
                skipped += 1
                done    += 1
                if (i + 1) % REPORT_EVERY == 0:
                    _report(i+1, total, done, skipped, errors, t_start, log)
                continue

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            # Extract
            try:
                with zf.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                done += 1
            except Exception as e:
                errors += 1
                msg = f"ERROR extracting {info.filename}: {e}"
                print(msg)
                log.write(msg + "\n")

            if (i + 1) % REPORT_EVERY == 0:
                _report(i+1, total, done, skipped, errors, t_start, log)

            if (i + 1) % FLUSH_EVERY == 0:
                log.flush()

    # Final report
    elapsed = time.time() - t_start
    summary = (
        f"\nExtraction complete in {fmt_time(elapsed)}.\n"
        f"  Extracted : {done - skipped:,}\n"
        f"  Skipped   : {skipped:,}\n"
        f"  Errors    : {errors}\n"
        f"  Free on D:: {free_gb('D:'):.1f} GB\n"
        f"  Free on C:: {free_gb('C:'):.1f} GB\n"
    )
    print(summary)
    log.write(summary)
    log.close()

    if errors == 0:
        print(f"All files extracted successfully.")
        answer = input(f"\nDelete source zip ({os.path.getsize(ZIP_PATH)/1e9:.1f} GB) from C: to free space? [y/N] ").strip().lower()
        if answer == "y":
            os.remove(ZIP_PATH)
            print(f"Zip deleted. Free on C: now {free_gb('C:'):.1f} GB")
        else:
            print("Zip kept.")
    else:
        print(f"WARNING: {errors} errors occurred. Zip NOT deleted. Check {LOG_PATH} for details.")


def _report(i, total, done, skipped, errors, t_start, log):
    elapsed  = time.time() - t_start
    pct      = 100.0 * i / total
    rate     = done / elapsed if elapsed > 0 else 0
    remaining = (total - i) / rate if rate > 0 else float("inf")
    line = (
        f"[{i:>9,}/{total:,}] {pct:5.1f}%  "
        f"rate={rate:.0f} f/s  ETA={fmt_time(remaining)}  "
        f"errors={errors}  D_free={free_gb('D:'):.1f}GB  C_free={free_gb('C:'):.1f}GB"
    )
    print(line)
    log.write(line + "\n")


if __name__ == "__main__":
    main()
