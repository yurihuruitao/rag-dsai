#!/usr/bin/env python3
"""
Download all PDFs listed in pdf_urls.json to bench_pdf/ directory.
Uses concurrent downloads with retry logic and progress tracking.
"""

import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- Configuration ---
JSON_PATH = "/workspace/rag/openragbench/pdf/arxiv/pdf_urls.json"
OUTPUT_DIR = "/workspace/rag/bench_pdf"
MAX_WORKERS = 8  # number of concurrent downloads
MAX_RETRIES = 3  # retry count per file
RETRY_DELAY = 2  # seconds between retries
TIMEOUT = 60  # request timeout in seconds


def download_one(paper_id: str, url: str, output_dir: str) -> tuple[str, bool, str]:
    """Download a single PDF. Returns (paper_id, success, message)."""
    filepath = os.path.join(output_dir, f"{paper_id}.pdf")

    # Skip if already downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return (paper_id, True, "already exists")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return (paper_id, True, "ok")
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                # Remove partial file
                if os.path.exists(filepath):
                    os.remove(filepath)
                return (paper_id, False, str(e))


def main():
    # Load URLs
    with open(JSON_PATH, "r") as f:
        pdf_urls: dict[str, str] = json.load(f)

    total = len(pdf_urls)
    print(f"Total PDFs to download: {total}")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    failed_papers = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one, pid, url, OUTPUT_DIR): pid
            for pid, url in pdf_urls.items()
        }

        for i, future in enumerate(as_completed(futures), 1):
            paper_id, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_papers.append((paper_id, msg))

            # Print progress every 50 files or on failure
            if i % 50 == 0 or not ok:
                status = "✓" if ok else "✗"
                print(f"[{i}/{total}] {status} {paper_id} - {msg}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Success: {success_count}/{total}")
    print(f"  Failed:  {fail_count}/{total}")

    if failed_papers:
        print(f"\nFailed papers:")
        for pid, msg in failed_papers:
            print(f"  {pid}: {msg}")

        # Save failed list for re-downloading
        fail_path = os.path.join(OUTPUT_DIR, "_failed.json")
        with open(fail_path, "w") as f:
            json.dump(dict(failed_papers), f, indent=2)
        print(f"\nFailed list saved to {fail_path}")


if __name__ == "__main__":
    main()
