"""Pre-publication audit (no git needed): which files would be committed under .gitignore rules,
their total size, large files, licence-restricted extensions, and a regex secret scan.
Writes results/repo_audit.md. Run: .venv/Scripts/python.exe scripts/repo_audit.py"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rules = [l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def ignored(rel: str) -> bool:
    parts = rel.split("/")
    for r in rules:
        r0 = r.rstrip("/")
        if r.endswith("/"):
            if any(fnmatch.fnmatch("/".join(parts[:i + 1]), r0) or fnmatch.fnmatch(parts[i], r0) for i in range(len(parts) - 1)):
                return True
        elif "**" in r:
            if fnmatch.fnmatch(rel, r.replace("**/", "*")) or fnmatch.fnmatch(rel, r):
                return True
        elif "/" in r0:
            if fnmatch.fnmatch(rel, r0):
                return True
        elif fnmatch.fnmatch(parts[-1], r0):
            return True
    return False


SECRET = [re.compile(p) for p in (r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{36}", r"github_pat_[A-Za-z0-9_]{40,}", r"sk-[A-Za-z0-9]{32,}",
                                  r"hf_[A-Za-z0-9]{30,}", r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}")]
keep, total, big, hits, restricted = [], 0, [], [], []
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(ROOT).as_posix()
    if rel.startswith(".git/") or ignored(rel):
        continue
    sz = p.stat().st_size; keep.append((rel, sz)); total += sz
    if sz > 5e6:
        big.append((rel, sz))
    if p.suffix.lower() in (".pkl", ".pt", ".pth", ".h5", ".hdf5", ".safetensors", ".npz", ".npy", ".csv") and "data/" in rel:
        restricted.append(rel)
    if sz < 3e6 and p.suffix.lower() in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".tex", ".csv", ".toml", ".cfg", ".ini", ""):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for rx in SECRET:
            for m in rx.finditer(txt):
                hits.append((rel, rx.pattern[:30], m.group(0)[:12] + "..."))
by_dir = {}
for rel, sz in keep:
    d = rel.split("/")[0] if "/" in rel else "."
    by_dir[d] = by_dir.get(d, 0) + sz
md = ["# Repository audit (pre-publication)", "", f"Files that would be committed: {len(keep)}, total {total/1e6:.1f} MB", "",
      "## Size by top-level folder", "", "| folder | MB |", "|---|---|"] + [f"| {d} | {s/1e6:.2f} |" for d, s in sorted(by_dir.items(), key=lambda x: -x[1])]
md += ["", "## Files > 5 MB", ""] + ([f"- {r} ({s/1e6:.1f} MB)" for r, s in big] or ["none"])
md += ["", "## Data-like files outside ignored folders", ""] + ([f"- {r}" for r in restricted] or ["none"])
md += ["", "## Secret scan (regex: cloud keys, GitHub/HF tokens, private keys, key=value secrets)", ""] + ([f"- {r}: `{p}` -> `{m}`" for r, p, m in hits] or ["no matches"])
md += ["", "## Licence-restricted material (must stay out of git)", "",
       "- data/raw/ (CARE-PD, gaitpdb, drawings), models/raw/ (Sabo checkpoints, SMPL, Parkonix/Swin/fastai weights), models/safe/ (our trained weights derived from CARE-PD)",
       "- results/ keeps only small derived numbers/figures; per-walk CSVs contain CARE-PD walk IDs and labels (derived data) -> check CARE-PD terms before publishing them.",
       "", "## Largest committed files", ""] + [f"- {r} ({s/1e6:.2f} MB)" for r, s in sorted(keep, key=lambda x: -x[1])[:15]]
(ROOT / "results" / "repo_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md[:30]))
