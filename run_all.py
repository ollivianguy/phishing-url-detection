r"""
run_all.py
==========
Reproduce every number, table and figure in the README with one command:

    .\.venv\Scripts\python.exe run_all.py

It runs the five stages in order (a few minutes; the dataset is downloaded
once and cached in data/). Each stage can also be run on its own.
"""
import time

import attack
import defend
import diagnose
import failure_analysis
import phishing_baseline

STAGES = [
    ("Data audit -- why the full dataset is too easy", diagnose.main),
    ("Baselines -- all features vs URL-only", phishing_baseline.main),
    ("Failure analysis -- why the URL-only model misses what it misses", failure_analysis.main),
    ("Attack -- feature-space vs problem-space evasion", attack.main),
    ("Defences -- what each one buys, and what it costs", defend.main),
]


def main():
    start = time.time()
    for i, (name, stage) in enumerate(STAGES, 1):
        print("\n" + "=" * 78)
        print(f"STAGE {i}/{len(STAGES)}: {name}")
        print("=" * 78)
        t = time.time()
        stage()
        print(f"[stage {i} finished in {time.time() - t:.0f}s]")
    print(f"\nAll stages done in {(time.time() - start) / 60:.1f} minutes. "
          "Figures are in figures/, numbers in results/.")


if __name__ == "__main__":
    main()
