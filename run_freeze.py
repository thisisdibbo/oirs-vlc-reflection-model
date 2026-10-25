"""Regenerate every figure at the frozen settings, then the diagnostics.

Replaces run_freeze.sh, which hardcoded a Linux path. Works on Windows,
macOS and Linux. Run it from the repository folder:

    python run_freeze.py

Output goes to freeze_run.log, overwriting whatever is there.
"""

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "freeze_run.log"

FROZEN = [
    "compare_models.py",
    "validate_optimizers.py",
    "sweep_panel.py",
    "sweep_power.py",
    "day4_orientation.py",
    "day5_mobility.py",
    "day6_blockage.py",
    "day7_noise.py",
    "day8_proposed.py",
    "day9_ber.py",
]

DIAGNOSTIC = [
    "physical_c.py",
    "noise_regime.py",
]

BAR = "#" * 64


def run(script, log, tag):
    log.write(f"\n{BAR}\n### {tag}{script}   started {time.strftime('%H:%M:%S')}\n{BAR}\n")
    log.flush()
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE,
                       stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    log.write(f"### {script} finished in {dt:.0f} s (exit {r.returncode})\n")
    log.flush()
    print(f"  {'FAILED ' if r.returncode else 'done   '}{script}  {dt:.0f}s")
    return r.returncode


def main():
    missing = [s for s in FROZEN + DIAGNOSTIC if not (HERE / s).exists()]
    if missing:
        print("Missing from this folder:", ", ".join(missing))
        return 1

    failed = []
    with open(LOG, "w", encoding="utf-8") as log:
        print(f"Frozen run -> {LOG.name}\n")
        for s in FROZEN:
            if run(s, log, ""):
                failed.append(s)
        print("\nDiagnostics (separate settings, cited as such in the paper)\n")
        for s in DIAGNOSTIC:
            if run(s, log, "DIAGNOSTIC "):
                failed.append(s)
        log.write("\nALL_FREEZE_RUNS_COMPLETE\n" if not failed
                  else f"\nFAILED: {', '.join(failed)}\n")

    print()
    if failed:
        print("FAILED:", ", ".join(failed))
        print("Open freeze_run.log and search for the script name.")
        return 1
    print("All runs complete. freeze_run.log is current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
