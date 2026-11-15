#!/usr/bin/env python3
"""M31 — arithmetic behind paper/M31_ROUTE_RANKING.md section 9.

Four independent checks, all elementary. Nothing here is taken from a paper;
every input that comes from a paper or from another module is named at the point
of use. Run:  python3 scripts/run_m31_route_ranking.py
Writes out/m31_route_ranking.json.
"""

import json
import math
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


# ---------------------------------------------------------------- 9.1 kappa(tau)
def kappa_exact(tau, n=200001):
    """|E_J2/E_J1| for a short junction, E(phi) = -Delta sqrt(1 - tau sin^2(phi/2)).

    Delta drops out of the ratio.  Sign convention: E = c0 - E_J1 cos(phi) - E_J2 cos(2 phi).
    """
    p = np.linspace(0.0, 2 * np.pi, n)
    E = -np.sqrt(1.0 - tau * np.sin(p / 2) ** 2)
    c1 = np.trapezoid(E * np.cos(p), p) / np.pi
    c2 = np.trapezoid(E * np.cos(2 * p), p) / np.pi
    return abs(-c2 / -c1)


def sec_9_1():
    taus = [1e-4, 3e-3, 0.069, 0.1, 0.5, 0.9, 0.999, 0.99999]
    rows = [
        {"tau": t, "kappa_exact": round(kappa_exact(t), 6), "tau_over_16": round(t / 16, 8)}
        for t in taus
    ]
    return {
        "model": "short junction, E(phi) = -Delta sqrt(1 - tau sin^2(phi/2))",
        "analytic_tunnel_limit": "E_J1 = Delta tau/4, E_J2 = Delta tau^2/64, hence kappa = tau/16",
        "saturation_as_tau_to_1": round(kappa_exact(0.99999), 4),
        "rows": rows,
        "reading": (
            "CHOICE C1 (kappa) and CHOICE C2 (tau) in M27 are the same parameter. "
            "At M27's tau = 3e-3 the delivered kappa is 1.9e-4, 533x below the assumed 0.1."
        ),
    }


# ------------------------------------------------- 9.2 self-consistent design point
def sec_9_2():
    # inputs, all from M27 Appendix A.4 / A.5
    N = 1221.0            # channels over 1 um^2 at delta-nu = 0.4
    Delta_GHz = 62.48     # BCS gap at T_c = 1.7 K (out/m23_majorana.json)
    EJ2_target = 1.908    # M27 reference design, GHz
    # E_J1(0) = N tau Delta / 4 ; kappa = tau/16 ; a2(45 deg, l=2) = [2+cos(180)]/3 = 1/3
    # => E_J2 = N Delta tau^2 / 192
    tau = math.sqrt(192 * EJ2_target / (N * Delta_GHz))
    return {
        "N_channels": N,
        "Delta_GHz": Delta_GHz,
        "a2_at_45deg_l2": round((2 + math.cos(math.radians(4 * 45))) / 3, 6),
        "EJ2_target_GHz": EJ2_target,
        "tau_self_consistent": round(tau, 4),
        "kappa_self_consistent": round(tau / 16, 6),
        "ratio_to_assumed_kappa_0p1": round(0.1 / (tau / 16), 1),
        "EJ1_at_zero_twist_GHz": round(N * tau * Delta_GHz / 4, 1),
        "EJ2_check_GHz": round(N * tau**2 * Delta_GHz / 192, 4),
        "EJ2_at_M27_tau_3e-3_GHz": round(N * (3e-3) ** 2 * Delta_GHz / 192, 6),
    }


# ------------------------------------------------------------ 9.3 umklapp kinematics
def sec_9_3():
    G_m = 0.5275   # nm^-1, M27 App. A.2 at L_m = 13.75 nm
    k_F = 0.1239   # nm^-1, M27 App. A.2 at delta-nu = 0.4, A_cell = 163.73 nm^2
    rows = []
    for Th in [27.2, 45.0, 60.0, 90.0]:
        th_eff = min(abs(Th - 60 * n) for n in range(3))
        G_int = 2 * G_m * math.sin(math.radians(th_eff) / 2)
        rows.append(
            {
                "Theta_deg": Th,
                "theta_eff_deg": round(th_eff, 2),
                "G_int_nm^-1": round(G_int, 4),
                "two_kF_nm^-1": round(2 * k_F, 4),
                "umklapp_open": bool(G_int < 2 * k_F),
            }
        )
    return {
        "note": (
            "a triangular moire superlattice has a six-fold reciprocal star, so the interface "
            "mismatch is periodic in 60 deg: theta_eff = min_n |Theta - 60n|, NOT Theta"
        ),
        "rows": rows,
    }


# ------------------------------------------------------- 9.4 phase held per micron
def sec_9_4():
    # M28's law: dphi/dl = 2/(3 sqrt3 xi).  xi values are Jha et al. arXiv:2403.02320 / Fig. 3.
    mats = [
        ("moire, Jha SQUID upper bound", 200.0),
        ("moire, Jha intrinsic-Ic estimate (hole)", 50.0),
        ("granular Al (Jha Fig. 3)", 24.0),
        ("NbN (Jha Fig. 3)", 15.0),
    ]
    rows = []
    for name, xi in mats:
        rows.append(
            {"material": name, "xi_nm": xi, "rad_per_um": round(2 / (3 * math.sqrt(3) * xi * 1e-3), 2)}
        )
    m200 = 2 / (3 * math.sqrt(3) * 200e-3)
    m50 = 2 / (3 * math.sqrt(3) * 50e-3)
    gral = 2 / (3 * math.sqrt(3) * 24e-3)
    nbn = 2 / (3 * math.sqrt(3) * 15e-3)
    return {
        "law": "dphi/dl = 2/(3 sqrt3 xi), M28 sec 2.1, from Jha et al. Eq. 15",
        "rows": rows,
        "grAl_over_moire": {"xi_200nm": round(gral / m200, 1), "xi_50nm": round(gral / m50, 1)},
        "NbN_over_moire": {"xi_200nm": round(nbn / m200, 1), "xi_50nm": round(nbn / m50, 1)},
        "reading": (
            "the moire superconductor is the WORST of the three at holding phase per unit length; "
            "M28 never applies its own law to the two comparison materials in its source's Fig. 3"
        ),
    }


# ------------------------------------------- 9.5 route D capacitance conflict (read-only)
def sec_9_5():
    path = os.path.join(OUT, "m30_loss.json")
    rows = []
    C_gate_published = 3.0  # fF, Tanaka et al. arXiv:2406.13740 Methods
    try:
        with open(path) as f:
            d = json.load(f)
        for r in d["design"]["ladder"]:
            rows.append(
                {
                    "N_squares": r["N_squares"],
                    "C_required_fF": r["C_required_fF"],
                    "below_published_C_gate": bool(r["C_required_fF"] < C_gate_published),
                }
            )
    except (OSError, KeyError) as exc:  # pragma: no cover
        rows = [{"error": f"could not read {path}: {exc}"}]
    return {
        "C_gate_published_fF": C_gate_published,
        "source": "Tanaka et al., arXiv:2406.13740 / Nature 638, 99 (2025), Methods",
        "rows": rows,
        "reading": (
            "the three rungs that carry the 1/N lever all specify a TOTAL resonator capacitance "
            "smaller than the published gate capacitance of a real MATBG cQED element"
        ),
    }


def main():
    result = {
        "_meta": {
            "script": "run_m31_route_ranking.py",
            "paper": "paper/M31_ROUTE_RANKING.md",
            "purpose": "reproducible arithmetic for section 9 of the route ranking",
            "inputs": [
                "paper/M27_FLOWERMON.md (N, Delta, G_m, k_F, E_J2 target)",
                "paper/M28_SUPERINDUCTANCE.md (phase law, xi values)",
                "out/m30_loss.json (ladder capacitances)",
            ],
        },
        "sec_9_1_kappa_tau_lock": sec_9_1(),
        "sec_9_2_self_consistent_design_point": sec_9_2(),
        "sec_9_3_umklapp_kinematics": sec_9_3(),
        "sec_9_4_phase_per_micron": sec_9_4(),
        "sec_9_5_route_D_capacitance_conflict": sec_9_5(),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "m31_route_ranking.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
