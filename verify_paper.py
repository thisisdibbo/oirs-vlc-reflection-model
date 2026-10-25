"""
Regenerate every figure and diagnostic, then check every number the paper cites.

    python verify_paper.py            run everything, then check      (~25 min)
    python verify_paper.py --check    check the existing freeze_run.log,
                                      running only diagnostics it lacks (~1 min)

Writes freeze_run.log and verification_report.txt in this folder.
"""

import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "freeze_run.log"
REPORT = HERE / "verification_report.txt"
TEX = HERE / "submission_oqe" / "main.tex"
FREEZE = HERE / "freeze.py"
DONE = "ALL_FREEZE_RUNS_COMPLETE"
BAR = "#" * 64
PM = r"\s*(?:±|\ufffd|\?|\+/-)\s*"

FROZEN = [
    "compare_models.py", "validate_optimizers.py", "sweep_panel.py",
    "sweep_power.py", "day4_orientation.py", "day5_mobility.py",
    "day6_blockage.py", "day7_noise.py", "day8_proposed.py", "day9_ber.py",
]
DIAGNOSTIC = [
    "physical_c.py", "aperture_stop.py", "noise_regime.py",
    "bisector_bound.py", "crossover.py",
]

C = []


def claim(where, what, value, script, pattern, kind="exact"):
    C.append((where, what, value, script, pattern, kind))


claim("5.1", "LoS anchor, conventional map", "43.68-51.03 dB",
      "compare_models.py", r"conventional \(no IRS\)\s+43\.68\s+51\.03")
claim("5.1", "cosine-model maximum SNR", "76.46 dB",
      "compare_models.py", r"IRS, cosine model\s+[\d.]+\s+76\.46")
claim("Abstract, 5.2, Table 2", "model divergence, max and mean", "8.69 / 6.28 dB",
      "compare_models.py", r"8\.69 dB\s+\(mean 6\.28 dB\)")
claim("5.1", "best SNR over the panel sweep", "77.71 dB",
      "sweep_panel.py", r"best 77\.71 dB")
claim("5.10", "overstatement across panel sweep", "5.08-6.40 dB",
      "sweep_panel.py", r"5\.08 to 6\.40 dB")
claim("5.10", "overstatement across power sweep", "5.95-7.13 dB",
      "sweep_power.py", r"5\.95 to 7\.13 dB")

claim("5.3", "TLBO gap, assignment space", "2.62 / 2.58 dB (two platforms)",
      "validate_optimizers.py", r"TLBO\s+[\d.]+\s+(?:2\.62|2\.58)\s+45,030", "range")
claim("5.3", "GA gap, assignment space", "0.14 / 0.00 dB (two platforms)",
      "validate_optimizers.py", r"GA\s+[\d.]+\s+(?:0\.14|0\.00)\s+45,030", "range")
claim("5.3", "PSO gap, assignment space", "1.65 dB",
      "validate_optimizers.py", r"PSO\s+[\d.]+\s+1\.65\s+45,030")

claim("5.4", "orientation knowledge, specular sitting", "0.00 dB",
      "day4_orientation.py", r"specular\s+sitting\s+-?[\d.]+\s+0\.00")
claim("5.4", "orientation knowledge, specular walking", "0.00 dB",
      "day4_orientation.py", r"specular\s+walking\s+-?[\d.]+\s+0\.00")
claim("5.4", "tilt effect, cosine sitting", "5.75 dB (4.3-5.8)",
      "day4_orientation.py", r"cosine\s+sitting\s+-5\.75")
claim("5.4", "tilt effect, specular walking", "4.34 dB (4.3-5.8)",
      "day4_orientation.py", r"specular\s+walking\s+-4\.34")
claim("5.4", "outage with tilt, low end", "1.7 %",
      "day4_orientation.py", r"cosine\s+walking\s+-4\.92\s+0\.00\s+1\.7%")
claim("5.4", "outage with tilt, high end", "3.0 %",
      "day4_orientation.py", r"specular\s+sitting\s+-5\.21\s+0\.00\s+3\.0%")

claim("5.5, Table 2, Contrib. 2", "latency at 50 ms, no occupant", "13.30 +/- 1.72 dB",
      "day5_mobility.py", r"specular\s+1\.0 m/s ->\s+13\.30" + PM + r"1\.72 dB")
claim("5.5", "acceptance sweep, alpha_min = 0.10 deg", "15.01 dB",
      "day5_mobility.py", r"0\.10\s+1\.1cm\s+15\.01")
claim("5.5", "acceptance sweep, alpha_min = 5.00 deg", "0.28 dB",
      "day5_mobility.py", r"5\.00\s+52\.5cm\s+0\.28")

claim("Abstract, 5.6, Sec. 6", "latency with occupant / blockage", "12.99 / 1.08 dB",
      "day6_blockage.py", r"specular\s+blockage\s+1\.08 dB \+ latency 12\.99 dB = 14\.07")
claim("5.6", "cosine blockage", "1.22 dB",
      "day6_blockage.py", r"cosine\s+blockage\s+1\.22 dB \+ latency\s+0\.00 dB =\s+1\.23")
claim("5.6", "blockage, static furniture only", "0.65 dB",
      "day6_blockage.py", r"specular\s+static\s+0\.65")
claim("5.6", "outage, latency alone (0 -> 50 ms)", "0.5 % -> 53.3 %",
      "day6_blockage.py", r"specular\s+none\s+0\.5%.*?53\.3%")
claim("5.6", "outage at 50 ms, with occupant", "58.9 %",
      "day6_blockage.py", r"specular\s+mobile\s+.*?58\.9%")

claim("5.7", "ambient, moderate indoor", "3.12 dB",
      "day7_noise.py", r"moderate indoor.*?\s3\.12\b")
claim("5.7", "ambient, bright daylight", "9.16 dB",
      "day7_noise.py", r"bright.*?\s9\.16\b")
claim("5.7", "diffuse share of LoS power", "12.9 %",
      "day7_noise.py", r"12\.9% of LoS")
claim("5.7", "diffuse SNR gain", "1.09 dB",
      "day7_noise.py", r"\+1\.09 dB")
claim("5.7", "saturation, reference channel", "17.36 dB (17.4)",
      "day7_noise.py", r"specular\s+paper's model\s+[\d.]+\s+[\d.]+\s+14\.70\s+[\d.]+\s+17\.36")
claim("5.7", "saturation with diffuse", "15.67 dB (15.7)",
      "day7_noise.py", r"specular\s+\+ diffuse\s+[\d.]+\s+[\d.]+\s+13\.30\s+[\d.]+\s+15\.67")

claim("5.9", "reactive closed form at 10 ms", "11.22 dB",
      "day8_proposed.py", r"10ms\s+0\.00m\s+11\.22")
claim("5.9", "zero-latency bound at 10 ms", "15.46 dB",
      "day8_proposed.py", r"10ms\s+0\.00m\s+11\.22\s+[\d.]+\s+15\.46")
claim("Abstract, 5.9", "recovered fraction, reactive", "73 %",
      "day8_proposed.py", r"10ms\s+0\.00m\s+73%\s+100%")
claim("5.9", "predictive recovered at 1 cm", "66 %",
      "day8_proposed.py", r"10ms\s+0\.01m\s+73%\s+66%")
claim("5.9", "predictive recovered at 2 cm", "37 %",
      "day8_proposed.py", r"10ms\s+0\.02m\s+73%\s+37%")
claim("5.9, Contrib. 3", "predictive recovered at 5 cm", "12 %",
      "day8_proposed.py", r"10ms\s+0\.05m\s+73%\s+12%")
claim("5.9", "predictive recovered at 10 cm", "4 %",
      "day8_proposed.py", r"10ms\s+0\.10m\s+73%\s+4%")
claim("5.9", "crossover at 20 ms, sigma = 2 cm", "55.37 vs 55.36 dB",
      "day8_proposed.py", r"20ms\s+0\.02m.*?55\.37.*?55\.36")

claim("5.8", "divergence at K = 36", "4.04 dB",
      "day9_ber.py", r"\b36\s+[\d.]+\s+[\d.]+\s+4\.04")
claim("5.8", "divergence peak at K = 256", "6.39 dB",
      "day9_ber.py", r"\b256\s+[\d.]+\s+[\d.]+\s+6\.39")
claim("5.8", "divergence at K = 400", "6.38 dB",
      "day9_ber.py", r"\b400\s+[\d.]+\s+[\d.]+\s+6\.38")
claim("5.8", "divergence at K = 784", "6.11 dB",
      "day9_ber.py", r"\b784\s+[\d.]+\s+[\d.]+\s+6\.11")
claim("5.8, Table 2", "divergence at K = 1600", "5.69 dB",
      "day9_ber.py", r"\b1600\s+[\d.]+\s+[\d.]+\s+5\.69")
claim("5.8, Table 2", "FEC power, reactive closed form", "3.86 mW",
      "day9_ber.py", r"reactive closed form\s+[\d.e+-]+\s+3\.86 mW")
claim("5.8, Table 2", "FEC power, metaheuristic", "20.00 mW",
      "day9_ber.py", r"metaheuristic\s+[\d.e+-]+\s+20\.00 mW")
claim("5.8", "FEC power, zero-latency bound", "2.00 mW",
      "day9_ber.py", r"genie\s+[\d.e+-]+\s+2\.00 mW")
claim("5.8", "saving of reactive over metaheuristic", "7.14 dB",
      "day9_ber.py", r"reactive closed form\s+7\.14 dB")

claim("3.3", "worst-case margin, 1 m panel", "76x",
      "aperture_stop.py", r"1\.0\s+100\.00\s+76x")
claim("3.3, 5.10", "worst-case margin, 0.5 m panel", "19x",
      "aperture_stop.py", r"0\.5\s+25\.00\s+19x")
claim("3.3, 5.10", "worst-case margin, 4 m panel", "1255x",
      "aperture_stop.py", r"4\.0\s+1600\.00\s+1255x")
claim("5.8", "Delta_H table, K = 36", "3.12 / 3.28 / 0.95",
      "noise_regime.py", r"\b36\s+3\.12 dB\s+3\.28 dB\s+0\.95")
claim("5.8", "Delta_H table, K = 1600", "6.25 / 8.75 / 0.71",
      "noise_regime.py", r"\b1600\s+6\.25 dB\s+8\.75 dB\s+0\.71")
claim("3.4", "bisector shortfall at 0.5 deg", "0.00001 dB",
      "bisector_bound.py", r"0\.50 deg.*?0\.00001")
claim("3.4", "bisector shortfall at 5 deg", "0.0012 dB",
      "bisector_bound.py", r"5\.00 deg.*?0\.0012")

for c, g, b in (("0.0200", "0.55", "25.80"), ("0.1021", "2.35", "13.58"),
                ("0.1758", "3.61", "10.19"), ("0.3027", "5.32", "7.28"),
                ("0.5213", "7.50", "4.93"), ("1.5454", "13.05", "1.76")):
    claim("5.11, Table 3", f"row c = {c}", f"{g} / {b} dB",
          "crossover.py", rf"{re.escape(c)}\s+{re.escape(g)} dB\s+{re.escape(g)} dB\s+{re.escape(b)} dB")
claim("5.11, Eq. 20", "latency/blockage crossover", "6.3 dB surface gain",
      "crossover.py", r"IRS gain over the unaided link is 6\.3 dB")
claim("5.11, Table 3", "row c = 1", "10.67 / 2.80 dB (10.7 / 2.80)",
      "crossover.py", r"IRS gain 10\.67 dB, latency 10\.67 dB, blockage 2\.80 dB")


def child_env():
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    return e


def sections(text):
    return set(re.findall(r"### (?:DIAGNOSTIC )?(\S+\.py)\s+started", text))


def run_one(script, fh, tag):
    fh.write(f"\n{BAR}\n### {tag}{script}   started "
             f"{time.strftime('%H:%M:%S')}\n{BAR}\n".encode())
    fh.flush()
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE,
                       stdout=fh, stderr=subprocess.STDOUT, env=child_env())
    dt = time.time() - t0
    fh.write(f"### {script} finished in {dt:.0f} s (exit {r.returncode})\n".encode())
    fh.flush()
    print(f"  {'FAILED' if r.returncode else 'done  '}  {tag.lower()}{script:<24} {dt:6.0f} s")
    return r.returncode == 0


def run_all():
    missing = [s for s in FROZEN + DIAGNOSTIC if not (HERE / s).exists()]
    if missing:
        print("Missing from this folder:", ", ".join(missing))
        return False
    ok = True
    with open(LOG, "wb") as fh:
        print(f"Running {len(FROZEN)} frozen scripts and {len(DIAGNOSTIC)} "
              f"diagnostics -> {LOG.name}\n")
        for tag, group in (("", FROZEN), ("DIAGNOSTIC ", DIAGNOSTIC)):
            for s in group:
                ok &= run_one(s, fh, tag)
        fh.write((f"\n{DONE}\n" if ok else "\nFAILED\n").encode())
    return ok


def fill_diagnostics():
    raw = LOG.read_bytes()
    have = sections(raw.decode("utf-8", errors="replace"))
    todo = [s for s in DIAGNOSTIC if s not in have]
    if not todo:
        return True
    print(f"Log lacks {len(todo)} diagnostic(s); running them now:\n")
    footer = f"\n{DONE}\n".encode()
    if raw.endswith(footer):
        LOG.write_bytes(raw[:-len(footer)] + b"\n")
    ok = True
    with open(LOG, "ab") as fh:
        for s in todo:
            ok &= run_one(s, fh, "DIAGNOSTIC ")
        fh.write((f"\n{DONE}\n" if ok else "\nFAILED\n").encode())
    print()
    return ok


def body(text, script):
    m = re.search(r"### (?:DIAGNOSTIC )?" + re.escape(script) +
                  r"\s+started.*?\n(.*?)### " + re.escape(script) + " finished",
                  text, re.S)
    return m.group(1) if m else ""


def line_of(sec, m):
    a = sec.rfind("\n", 0, m.start()) + 1
    b = sec.find("\n", m.end())
    return sec[a:b if b != -1 else len(sec)].strip()


def pinned():
    if not FREEZE.exists():
        return []
    f = FREEZE.read_text(encoding="utf-8", errors="replace")
    out = []
    for name, paper, where in (("T_EVAL_MS", "0.169", "5.9, ms per evaluation"),
                               ("TAU_META_S", "7.69", "5.9, Fig. 11, s")):
        m = re.search(rf"^{name}\s*=\s*([\d.]+)", f, re.M)
        got = m.group(1) if m else "not found"
        out.append((name, paper, got, where))
    return out


def check():
    if not LOG.exists():
        print("No freeze_run.log here. Run without --check first.")
        return False
    text = LOG.read_bytes().decode("utf-8", errors="replace")
    absent = [s for s in FROZEN if s not in sections(text)]

    rows, ok = [], 0
    for where, what, value, script, pat, kind in C:
        sec = body(text, script)
        if not sec:
            rows.append(("NO RUN", where, what, value, script, "not in the log"))
            continue
        m = re.search(pat, sec)
        if m:
            st = "MATCH" if kind == "exact" else "IN RANGE"
            rows.append((st, where, what, value, script, line_of(sec, m)))
            ok += 1
        else:
            rows.append(("DIFFERS", where, what, value, script,
                         "expected value not found; inspect this script's output"))

    vo, d8 = body(text, "validate_optimizers.py"), body(text, "day8_proposed.py")
    mb = re.search(r"closed-form bisector\s+([\d.]+) us", d8)
    mg = re.search(r"GA\s+[\d.]+\s+[\d.]+\s+45,030\s+([\d.]+)\s+([\d.]+)s", vo)
    timing = []
    if mb and mg:
        us, tev, ga = float(mb.group(1)), float(mg.group(1)), float(mg.group(2))
        timing = [("closed-form update, K = 400", "64.9 us", f"{us} us"),
                  ("GA per evaluation", "0.263 ms", f"{tev} ms"),
                  ("converged GA run", "11.84 s", f"{ga} s"),
                  ("ratio GA / closed form", "1.8e+05", f"{ga / (us * 1e-6):.1e}")]

    w = max(len(r[2]) for r in rows)
    out = ["VERIFICATION REPORT",
           "Closed-Form Mirror Steering and the Cost of Control Latency",
           "in Optical IRS-Assisted Visible Light Communication",
           f"generated {time.strftime('%Y-%m-%d %H:%M')}",
           "=" * 78,
           f"{ok} of {len(rows)} cited numbers reproduced.",
           "  MATCH    = the exact value the paper states",
           "  IN RANGE = one of the two platform values the paper states (Section 5.3)",
           ""]
    if absent:
        out += [f"WARNING: frozen scripts missing from the log: {', '.join(absent)}", ""]
    for st, where, what, value, script, line in rows:
        out += [f"[{st:8}] {what:<{w}}  paper: {value}",
                f"           Section {where}  <-  {script}",
                f"           log: {line}", ""]

    out += ["=" * 78, "PINNED CONSTANTS (freeze.py)",
            "Used by the experiments in place of a fresh timing measurement."]
    for name, paper, got, where in pinned():
        flag = "MATCH" if got == paper else "DIFFERS"
        out.append(f"  [{flag:7}] {name:<11} paper {paper:<7} freeze.py {got:<10} ({where})")

    out += ["", "=" * 78, "MACHINE-DEPENDENT TIMINGS",
            "Measured on whichever computer ran this, so not expected to match.",
            "The paper's claim is only that the ratio is about five orders of",
            "magnitude (1e5). Check the last line, not the seconds.", ""]
    for what, paper, here in timing:
        out.append(f"  {what:<28} paper {paper:<10} this run {here}")
    if not timing:
        out.append("  timing lines not found in the log")

    out += ["", "=" * 78, "PROVENANCE OF THE TWO LATENCY FIGURES",
            "  13.30 +/- 1.72 dB  day5_mobility   50 ms, specular, no occupant",
            "  12.99 dB (13.0)    day6_blockage   50 ms, specular, mobile occupant,",
            "                                     measured with 1.08 dB blockage",
            "  Places in main.tex that cite either:"]
    if TEX.exists():
        for i, L in enumerate(TEX.read_text(encoding="utf-8").split("\n"), 1):
            if re.search(r"\\SI\{13\.0\}|13\.3 \\pm|13\.30 \\pm|12\.99", L):
                out.append(f"    line {i}: {L.strip()[:64]}")

    report = "\n".join(out)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {REPORT.name}.")
    return ok == len(rows) and not absent


if __name__ == "__main__":
    if "--check" in sys.argv:
        if LOG.exists() and not fill_diagnostics():
            sys.exit(1)
    elif not run_all():
        sys.exit(1)
    else:
        print()
    sys.exit(0 if check() else 1)
