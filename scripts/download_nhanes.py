#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE = "https://wwwn.cdc.gov"
DEFAULT_CYCLES = ["2021-2023", "2017-2018"]
REQUESTED_MODULES = [
    "DEMO", "BMX", "BPX", "BPXO", "TCHOL", "HDL", "TRIGLY", "GLU", "GHB",
    "BIOPRO", "CBC", "CRP", "DIQ", "BPQ", "MCQ", "SMQ", "ALQ", "PAQ",
    "SLQ", "DPQ", "HSQ", "RXQ_RX",
]


def cycle_suffix(cycle):
    return {"2017-2018": "J", "2021-2023": "L", "2017-March 2020": "P"}.get(cycle)


def data_page(cycle):
    return f"{BASE}/Nchs/Nhanes/Search/DataPage.aspx?Cycle={cycle}"


def discover_links(cycle):
    html = requests.get(data_page(cycle), timeout=45).text
    soup = BeautifulSoup(html, "lxml")
    links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".xpt"):
            continue
        name = Path(href).name
        links[name.upper()] = urljoin(BASE, href)
    return links


def module_to_file(module, suffix, links):
    candidates = []
    if suffix:
        candidates.append(f"{module}_{suffix}.XPT")
    candidates.append(f"{module}.XPT")
    for name in candidates:
        if name.upper() in links:
            return name.upper(), links[name.upper()]
    return None, None


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def clean_frame(df, module):
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    if "SEQN" not in df.columns:
        return None
    keep = ["SEQN"]
    for col in df.columns:
        if col == "SEQN":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            keep.append(col)
    df = df[keep]
    renamed = {"SEQN": "SEQN"}
    for col in df.columns:
        if col != "SEQN":
            renamed[col] = f"{module}__{col}"
    return df.rename(columns=renamed)


def merge_cycle(raw_dir, files):
    merged = None
    used = []
    for module, path in files:
        df = pd.read_sas(path, format="xport")
        df = clean_frame(df, module)
        if df is None:
            continue
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="SEQN", how="outer")
        used.append({"module": module, "path": str(path), "rows": int(len(df)), "cols": int(df.shape[1])})
    return merged, used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", nargs="+", default=DEFAULT_CYCLES)
    p.add_argument("--modules", nargs="+", default=REQUESTED_MODULES)
    p.add_argument("--out-dir", default="data/nhanes")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    manifest = {"cycles": [], "modules_requested": args.modules}

    for cycle in args.cycles:
        suffix = cycle_suffix(cycle)
        links = discover_links(cycle)
        files = []
        cycle_manifest = {"cycle": cycle, "downloaded": [], "missing": []}
        for module in args.modules:
            name, url = module_to_file(module, suffix, links)
            if not url:
                cycle_manifest["missing"].append(module)
                continue
            dest = raw_dir / re.sub(r"[^0-9A-Za-z_-]+", "_", cycle) / name
            download(url, dest)
            files.append((module, dest))
            cycle_manifest["downloaded"].append({"module": module, "file": name, "url": url, "bytes": dest.stat().st_size})
        merged, used = merge_cycle(raw_dir, files)
        if merged is not None:
            merged["CYCLE"] = cycle
            all_frames.append(merged)
        cycle_manifest["used"] = used
        manifest["cycles"].append(cycle_manifest)

    combined = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)
    combined.to_parquet(processed_dir / "nhanes_phenome_raw.parquet", index=False)
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "rows": int(combined.shape[0]),
        "cols": int(combined.shape[1]),
        "parquet": str(processed_dir / "nhanes_phenome_raw.parquet"),
        "manifest": str(processed_dir / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
