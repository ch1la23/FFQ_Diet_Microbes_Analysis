"""
Run the whole pipeline (D1-D4) with one command:

    python run_pipeline.py                      # Python engine (default)
    python run_pipeline.py --engine maaslin2    # MaAsLin2 in R (needs R + Maaslin2)
    python run_pipeline.py --skip 02            # skip a step (see STEPS below)
    python run_pipeline.py --skip-check         # don't run the input check

All settings are in config.py. A record of the run (software versions,
checksums of the input files, the settings used, timings) is written to
results/D4_Reproducible_Pipeline/run_report.json
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as C

# (step number, script, does it take --engine?)
STEPS = [
    ("00", "scripts/00_check_inputs.py", False),
    ("01", "scripts/01_preprocess.py", False),
    ("02", "scripts/02_d1_dietary_atlas.py", False),
    ("03", "scripts/03_d2_ghost_biomarkers.py", True),
    ("04", "scripts/04_d3_food_microbiome.py", True),
    ("05", "scripts/05_figures_and_tables.py", False),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_versions():
    versions = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "openpyxl"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = "not installed"
    return versions


parser = argparse.ArgumentParser(description="Run the HMP2 diet / microbiome pipeline.")
parser.add_argument("--engine", default="python", choices=["python", "maaslin2"])
parser.add_argument("--skip", nargs="*", default=[], help="step numbers to skip, e.g. --skip 02 05")
parser.add_argument("--skip-check", action="store_true", help="same as --skip 00")
args = parser.parse_args()

skip = set(args.skip) | ({"00"} if args.skip_check else set())

problems = C.check_config()
if problems:
    sys.exit("Problems in config.py:\n  " + "\n  ".join(problems))

C.make_output_dirs()
timings = {}

for number, script, takes_engine in STEPS:
    if number in skip:
        print(f"\n--- skipping step {number}: {script}")
        continue

    command = [sys.executable, str(ROOT / script)]
    if takes_engine:
        command += ["--engine", args.engine]

    print(f"\n=== step {number}: {script} ===", flush=True)
    start = time.time()
    subprocess.run(command, check=True, cwd=ROOT)
    timings[number] = round(time.time() - start, 1)

# ---- run report ------------------------------------------------------
inputs = [C.DATA_DIR / C.METADATA_FILE, C.DATA_DIR / C.ABUNDANCE_FILE]
if C.TAXONOMY_FILE:
    inputs.append(C.DATA_DIR / C.TAXONOMY_FILE)

report = {
    "engine": args.engine,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": package_versions(),
    "seed": C.SEED,
    "input_files_sha256": {p.name: sha256(p) for p in inputs if p.exists()},
    "config_sha256": sha256(ROOT / "config.py"),
    "foods": list(C.FOODS),
    "model_foods": C.MODEL_FOODS,
    "ffq_scores": C.FFQ_SCORES,
    "models": list(C.SPECS),
    "timings_s": timings,
}

for name, folder in (("d1", C.D1_DIR), ("d2", C.D2_DIR), ("d3", C.D3_DIR)):
    summary_file = folder / f"{name}_summary.json"
    if summary_file.exists():
        report[name] = json.loads(summary_file.read_text())

report_file = C.D4_DIR / "run_report.json"
report_file.write_text(json.dumps(report, indent=2, default=str))

print(f"\nPipeline finished. Results are in {C.RESULTS_DIR}")
print(f"Run report: {report_file}")
