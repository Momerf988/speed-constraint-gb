"""Download raw source files and record them in a manifest.

Files land in data/raw/ unmodified. The manifest records where each file came
from, when it was retrieved, its size and its SHA-256 checksum, so every later
result can be traced to an exact copy of the source.

Usage:
    python ingest/download.py            # download anything not already present
    python ingest/download.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "ingest" / "manifest.json"

STATS19 = "https://data.dft.gov.uk/road-accidents-safety-data"
GOVUK = "https://assets.publishing.service.gov.uk/media"

# Source files, keyed by a short description. Local file name = URL basename.
SOURCES = {
    "STATS19 collisions, 1979 to latest year": f"{STATS19}/dft-road-casualty-statistics-collision-1979-latest-published-year.csv",
    "STATS19 vehicles, 1979 to latest year": f"{STATS19}/dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv",
    "STATS19 casualties, 1979 to latest year": f"{STATS19}/dft-road-casualty-statistics-casualty-1979-latest-published-year.csv",
    "STATS19 data guide (code lookups), 2025": f"{GOVUK}/6a63900b2dc18ebe4c3b2bc8/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx",
    "Severity adjustment guidance": f"{GOVUK}/691c644021ef5aaa6543eef0/dft-road-casualty-statistics-severity-adjustment-figure-guidance.docx",
    "DfT TRA0202 traffic, billion vehicle-km by road class, 1993 onwards": f"{GOVUK}/6a0b7cb4c510c3913d8267cc/tra0202-km-by-road-class.ods",


}

CHUNK = 1 << 20  # 1 MiB
REPORT_EVERY = 100 * CHUNK


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def download(url: str, dest: Path) -> dict:
    """Stream url to dest, hashing as it goes. Writes to a .part file first so an
    interrupted download never looks complete."""
    part = dest.with_name(dest.name + ".part")
    sha = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "speed-constraint-gb/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        expected = int(response.headers.get("Content-Length") or 0)
        while chunk := response.read(CHUNK):
            out.write(chunk)
            sha.update(chunk)
            size += len(chunk)
            if size % REPORT_EVERY < CHUNK:
                total = f" of {expected / 1e6:,.0f}" if expected else ""
                print(f"    {size / 1e6:,.0f}{total} MB", flush=True)
    if expected and size != expected:
        raise IOError(f"{dest.name}: got {size} bytes, server said {expected}")
    part.rename(dest)
    return {
        "url": url,
        "file": str(dest.relative_to(ROOT)),
        "bytes": size,
        "sha256": sha.hexdigest(),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    for label, url in SOURCES.items():
        dest = RAW / url.rsplit("/", 1)[1]
        entry = manifest.get(label)
        if not args.force and entry and dest.exists() and dest.stat().st_size == entry["bytes"]:
            print(f"skip  {label} (already downloaded)")
            continue
        print(f"get   {label}\n      {url}")
        manifest[label] = download(url, dest)
        save_manifest(manifest)  # save after each file so a later failure keeps earlier work
        print(f"done  {manifest[label]['bytes'] / 1e6:,.1f} MB  sha256 {manifest[label]['sha256'][:12]}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())
