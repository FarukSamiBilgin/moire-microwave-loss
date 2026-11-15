"""Automated check: the numbers quoted in the preprint (and in README.md)
against the independent re-derivation p4_yeniden_uretim.py.

Run from the repository root:
    python -I scripts/p4_yeniden_uretim.py --compare results/m30_loss.json
    python -I scripts/check_paper_numbers.py
The first command writes scripts/p4_yeniden_uretim_sonuc.json (a committed
copy is in results/). This script reads that file and asserts that each value
rounds to the figure printed in the paper. Exit code 0 means all pass.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for cand in (os.path.join(HERE, "p4_yeniden_uretim_sonuc.json"),
             os.path.join(ROOT, "results", "p4_yeniden_uretim_sonuc.json")):
    if os.path.exists(cand):
        SRC = cand
        break
else:
    sys.exit("run p4_yeniden_uretim.py first")

R = json.load(open(SRC, encoding="utf-8"))

# (key in re-derivation, value printed in the paper, relative tolerance)
CHECKS = [
    ("T1.x_at_p=0.5", 0.46, 0.01),
    ("T1.x_at_p=0.99", 5.7, 0.01),
    ("T1.x_at_p=0.997", 10.5, 0.01),
    ("T1.x=0.3.p_exact", 0.354, 0.002),
    ("T1.x=1.0.p_exact", 0.778, 0.001),
    ("T1.x=10.0.p_exact", 0.9967, 0.0001),
    ("T2.Q_MATBG_lower_direct", 20.9, 0.003),
    ("T2.Q_MATBG_lower_excess", 21.2, 0.003),
    ("T2.tan_delta_upper_direct", 4.79e-2, 0.002),
    ("Z.D2.p", 0.3885, 0.0005),
    ("Z.D2.tand_upper_tableQi", 4.3e-5, 0.01),
    ("D.N=5.p", 0.99998, 1e-5),
    ("D.N=5.C_fF", 4.092, 0.001),
    ("D.N=5.Z_kohm", 5.98, 0.001),
    ("D.N=5.hf_kB_mK", 312, 0.002),
    ("D.N=5.area_um2", 1.416, 0.001),
    ("S.stub_len_um_eps6.35", 11.57, 0.001),
    ("S.quarterwave_mm_5.5GHz_eps6.35", 5.4, 0.01),
    ("S.quarterwave_mm_4.2GHz_eps6.35", 7.1, 0.01),
    ("C.rho=100.Q_min", 138, 0.005),
    ("C.rho=100.Q_max", 3498, 0.001),
    ("C.rho=16000.Q_max", 21.9, 0.005),
    ("C.pad_gain_vs_400nm", 25, 0.001),
    ("C.pad_gain_vs_532nm", 18.8, 0.003),
    ("R.rho=100.tand=0.0.3sig_intercept", 1.09e-5, 0.005),
    ("H.electric_participation_gate_N5", 0.73, 0.01),
    ("H.Q_hbn_limited_N5_at_bound", 2.7e5, 0.02),
    ("F.T1=0.0001.Q_ind", 3.218e6, 0.001),
    ("F.T1=1e-06.Q_ind", 3.218e4, 0.001),
    ("B.balgley_over_viable", 55, 0.01),
    ("E.f=6.0.T1_us_p0.01", 85.57, 0.001),
    ("E.f=6.0.T1_us_p1", 0.86, 0.01),
]

fail = 0
for key, paper, tol in CHECKS:
    got = R[key]
    ok = abs(got - paper) <= tol * abs(paper)
    fail += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {key:40s} paper={paper:<10g} recomputed={got:.6g}")
print(f"\n{len(CHECKS) - fail}/{len(CHECKS)} passed")
sys.exit(1 if fail else 0)
