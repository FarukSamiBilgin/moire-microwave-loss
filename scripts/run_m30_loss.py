"""M30 -- ROUTE D: THE GATING MEASUREMENT.

M22 section E.7 (the survey's "6.3") names the cheapest decisive experiment in
the whole programme: the single-photon microwave loss of a magic-angle twisted
bilayer graphene element has never been published, so every coherence claim
about a moire qubit is unbounded from below.  M28 sketched a test resonator in
five sentences.  This module designs the experiment completely.

The question the module has to answer FIRST, because everything else is
downstream of it:

    Can a resonator containing a MATBG element even SEE that element's loss?

That is a participation-ratio question, and the literature already contains a
published failure.  Kreidel et al. (arXiv:2407.09916) embedded exfoliated
NbSe2 as a terminating inductor on a Nb quarter-wave line, found the sample
participation was "merely a few p.p.t.", and state plainly that Q_0 was then
too low to discern the sample's own Q factor.  Tanaka et al.
(arXiv:2406.13740, Nature 638, 99 (2025)) measured MATBG in exactly that
geometry and the resonance moves "only about 1%" between the insulating and
superconducting states -- a ~2% participation, which is why the extractable
bound on MATBG is Q >= 10-50 and not a measurement.

THE CENTRAL NEW RESULT (section 2).  For a transmission line of impedance Z0
terminated by an inductance L, the exact energy participation of that inductor
in the fundamental mode is

        p(phi) = sin(2 phi) / ( sin(2 phi)/2 + (pi/2 - phi) ),
        phi    = arctan(omega L / Z0)

which reduces to Kreidel's perturbative p = 2 |delta_omega / omega| when
omega L << Z0, and goes to 1 when omega L >> Z0.  The published moire cQED
geometry sits at the wrong end of this curve BY CONSTRUCTION: the perturbation
theory that makes the kinetic-inductance extraction clean is the same
approximation that makes the loss measurement blind.  The fix is not a better
fit, a colder fridge or a quieter amplifier.  It is to SHORTEN THE LINE until
the sample is the resonator.  For the M28 strip (430 nH) at 5 GHz that means a
14 um stub, i.e. a lumped 2.4 fF shunt -- which is precisely M28's number,
arrived at here from the distributed side.

THE SECOND NEW RESULT (section 4).  With p -> 1 the contacts move into series
with the entire resonant current, and the one published rf number for a moire
contact (R_c = 3.2 kOhm, Banerjee et al., arXiv:2406.13742) would cap Q_i near
4.  A single device therefore measures the contacts, not the material.  The
separation is a LENGTH LADDER: at fixed width, both L and the intrinsic series
resistance scale with the square count N, while the contact resistance does
not, so

        1/Q_i(N) = tan_delta_intrinsic + R_c / (omega L_sq N)

and a fit versus 1/N returns the intrinsic loss as the INTERCEPT and the
contact resistance as the SLOPE.  The ladder is not a convenience; without it
the experiment cannot answer its own question.

Everything below is either (a) a literature number tagged with its source,
(b) arithmetic on those numbers, marked as mine, or (c) a CHOICE registered
with a two-directional sensitivity.  Speculation is tagged SPECULATION.
"""
import json
import os

import numpy as np

# ---------------------------------------------------------------- constants
H = 6.62607015e-34
HBAR = H / (2.0 * np.pi)
KB = 1.380649e-23
C0 = 2.99792458e8
PHI0 = 2.067833848e-15

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

# ------------------------------------------------------------- literature
# Every entry here is read off a named paper.  Nothing is inferred.
LIT = {
    "tanaka_matbg_cqed": {
        "ref": "arXiv:2406.13740 / Nature 638, 99 (2025)",
        "authors": "Tanaka, Wang, Dinh, Rodan-Legrain, ... Jarillo-Herrero, Oliver",
        "geometry": "Al lambda/4 CPW terminated to ground through hBN/MATBG/hBN",
        "f_r_GHz": 4.2008154,
        "Qi": 1044.7,
        "Qi_err": 28.5,
        "Qc": 4161.0,
        "Qc_err": 54.4,
        "control_f_GHz": 6.0653899,
        "control_Qi": 74833.2,
        "control_Qi_err": 1830.2,
        "control_Qc": 24926.9,
        "f0_insulating_GHz": 3.8,
        "L_prx_nH": 1.32,
        "frac_shift_ins_to_sc": 0.01,
        "T_base_mK": 20.0,
        "T_eff_mK": 40.0,
        "Tc_half_K": 1.0,
        "contacts": "Ti/Al edge contacts, RIE + thermal evaporation; 5 galvanic contacts",
        "photon_number": "NOT STATED for Q_i.  Power was swept on f_r (Fig. 5d) and "
                         "bias current was swept on Q_i (Supp. Fig. 17), but Q_i was "
                         "never reported versus microwave power.",
        "their_own_words": "'The resonator with MATBG termination features a quality "
                           "factor exceeding 1000 at 20 mK ... capable of discerning "
                           "changes in the superfluid stiffness at the 1% level.'",
    },
    "banerjee_ttg_rf": {
        "ref": "arXiv:2406.13742 / Nature 638, 93 (2025)",
        "f_r_MHz": 350.0,
        "theta_deg": 1.55,
        "LK_nH": [120.0, 350.0],
        "Rc_kohm": 3.2,
        "rf_power_dBm": [-112.0, -102.0],
        "limit": "R_rf = R_2t = R_c: the measured rf resistance IS the contact "
                 "resistance, so the moire element's own dissipation sits below the "
                 "contact floor and no intrinsic loss can be extracted.",
    },
    "kreidel_vdw_resonator": {
        "ref": "arXiv:2407.09916 (Kreidel, Chu, Balgley, ... Hone, Fong)",
        "probe": "Nb lambda/4 CPW, 240 nm Nb, 2 kOhm/sq sheet R, float-zone Si",
        "Q0_typ": 1e5,
        "samples": "two 80 nm Al strips (10 x 420 um) and three exfoliated NbSe2 strips",
        "participation_statement": "'the participation ratio of the sample is merely a "
                                   "few p.p.t., the probe resonator quality factor Q_0 "
                                   "is too low for us to discern the true Q factor of "
                                   "the sample component.'",
        "p_formula": "1/Q_eff = (4/pi)(r/Z0) = 2 (delta_omega0/omega0) tan_delta",
        "TLS_peak_K": 0.2,
        "TLS_peak_meaning": "Q_i peaks at T ~ hbar omega_0 / k_B; the peak is suppressed "
                            "at higher drive power.  A published TLS fingerprint in this "
                            "exact device class.",
        "attenuation_dB": 65.0,
        "flux_control": "holes periodically etched into the Nb ground plane to trap "
                        "stray vortices",
    },
    "zaman_nbse2_KI": {
        "ref": "arXiv:2511.08466 (Zaman, Wang, ... Jarillo-Herrero, Oliver), v3 Jan 2026",
        "probe": "Al lambda/4 CPW, 250 nm MBE Al on high-resistivity Si",
        "Lsq_monolayer_nH": 1.2,
        "Kerr_Hz_per_photon": [-0.008, -14.7],
        "T_base_mK": 10.0,
        "spectator_Qi": 5.985e6,
        "spectator_Qc": 1.0e5,
        "spectator_f_GHz": 6.0583,
        "single_photon": True,
        "note": "THE KEY PRECEDENT: this is the M30 measurement, done at the single-"
                "photon level, on a van der Waals superconductor -- but NbSe2, which is "
                "not a moire material.  It supplies the recipe, the contact process and "
                "the control-resonator discipline.",
        # d_nm, W_um, L_um, f_sample_GHz, f_Al_GHz, Lk_nH, Lsq_nH, Qi
        "devices": [
            ("D1", 1.99, 3.1, 3.1, 3.62, 5.30, 1.195, 1.195, 0.8e4),
            ("D2", 4.08, 2.5, 5.7, 4.14, 5.21, 0.643, 0.282, 6.0e4),
            ("D3", 5.86, 2.6, 3.0, 4.88, 5.33, 0.220, 0.191, 0.7e4),
            ("D4", 7.13, 4.8, 6.7, 4.93, 5.35, 0.200, 0.143, 2.4e4),
            ("D5", 7.98, 4.5, 12.0, 4.63, 5.33, 0.360, 0.135, 1.2e4),
            ("D6", 8.81, 3.9, 8.6, 4.79, 5.34, 0.274, 0.124, 2.8e4),
            ("D7", 8.90, 4.5, 4.2, 5.26, 5.49, 0.100, 0.107, 2.1e4),
            ("D8", 10.07, 2.5, 6.6, 5.16, 5.49, 0.149, 0.056, 0.7e4),
            ("D9", 11.89, 6.0, 6.9, 5.46, 5.60, 0.057, 0.049, 0.6e4),
            ("D10", 24.73, 4.1, 7.5, 5.34, 5.37, 0.014, 0.008, 18e4),
            ("D11", 30.20, 3.4, 6.1, 5.42, 5.44, 0.009, 0.005, 2.5e4),
            ("D12", 52.12, 2.3, 4.6, 5.60, 5.62, 0.008, 0.004, 5.8e4),
        ],
        "authors_attribution": "'likely due to additional dissipation arising from "
                               "imperfect electrical contacts between the Al and the "
                               "NbSe2, as well as dielectric loss introduced by residual "
                               "polymer contamination from the dry transfer process.'",
    },
    "balgley_vdw_transmon": {
        "ref": "arXiv:2512.08059 (Balgley, Park, Chu, ... Hone, Fong), Dec 2025",
        "device": "all-van-der-Waals merged-element transmon, NbSe2/WSe2/NbSe2",
        "T1_mean_us": [1.11, 1.67, 0.27, 0.20],
        "T1_best_us": [1.39, 2.46, 0.38, None],
        "T2E_best_us": [1.47, 4.82, None, None],
        "p_qubit_pct": [97, 96, 91, 98],
        "tan_delta_qubit": [3.09e-5, 1.69e-5, 12.20e-5, 15.76e-5],
        "tan_delta_capacitor": [3.31e-5, 5.60e-5, 9.08e-5, 14.02e-5],
        "p_capacitor_pct": [16, 18, 18, 12],
        "single_photon_dBm": -145.0,
        "p_formula": "p = |sin(varphi)| / (varphi + |sin(varphi)|); "
                     "1/Q_term = 2 p tan_delta + (1-p)/Q_open",
        "T_mK": 15.0,
        "attenuation_dB": [70.0, 84.0],
        "MXC_attenuator_dB": 40.0,
        "note": "the state of the art for a van der Waals QUBIT: T1 ~ 1-2 us, loss "
                "tangent 1.7e-5 to 1.6e-4, dielectric-loss limited, NOT quasiparticle "
                "limited.  This is the prior that M30's thresholds must be read against.",
    },
    "hBN": {
        "ref": "arXiv:2109.00015 / Nature Materials 21, 398 (2022)",
        "tan_delta_upper": 5.0e-6,
        "eps_perp": 3.76,
        "eps_ref": "Laturia, de Put, Vandenberghe, npj 2D Mater. Appl. 2, 6 (2018)",
        "photons": 1.0,
        "T_mK": 10.0,
    },
    "grAl": {
        "ref": "arXiv:2411.12611",
        "Qi_best_allgrAl": 3.66e6,
        "Qi_best_hybrid": 4.72e6,
        "Qi_worst": 0.46e6,
        "Lsq_pH_max": 320.0,
        "photons": 1.0,
        "trend": "Q_i falls monotonically with sheet inductance (also seen in NbN, InO)",
    },
    "jha_matt_Lsq": {
        "ref": "arXiv:2403.02320 / Phys. Rev. Lett. 134, 216001 (2025)",
        "material": "magic-angle twisted TRILAYER graphene, 1.71-1.73 deg",
        "Lsq_nH": [10.0, 150.0],
        "xi_nm_upper": 200.0,
        "loss": "NOT MEASURED; 'it will be intriguing to investigate the kinetic "
                "impedance at high frequencies'",
    },
    "graphene_qubit": {
        "ref": "arXiv:2506.04959 (Aparicio, Messelot et al.)",
        "T1_ns": 716.0,
        "T1_err_ns": 17.0,
        "note": "best graphene superconducting qubit; Wang et al., Nat. Nanotech. 14, "
                "120 (2019) graphene gatemon ~70 ns",
    },
    "fit_methods": {
        "circle_fit": "Probst, Song, Bushev, Ustinov, Weides, Rev. Sci. Instrum. 86, "
                      "024706 (2015), arXiv:1410.3365",
        "circle_fit_optimisation": "Baity et al., Phys. Rev. Research 6, 013329 (2024)",
        "loss_review": "McRae, Wang, Bejanin, Pagel, Dunsworth, Mariantoni, "
                       "Rev. Sci. Instrum. 91, 091101 (2020), arXiv:2006.04718",
        "photon_number": "n_r = 2 kappa P_rf / (hbar omega_r (kappa + gamma)^2); "
                         "form as used in arXiv:2511.08466 Eq. 5, after Norris et al., "
                         "EPJ Quantum Technology 11, 5 (2024)",
    },
    "amplifier": {
        "ref": "Ranadive et al., Nature Electronics (2025); catalogued in "
               "out/m13_readout_chain.json",
        "name": "JTWPA 3-11 GHz",
        "n_add": 1.0,
        "P1dB_dBm": -102.0,
    },
}

# --------------------------------------------------------------- CHOICES
CHOICES = [
    dict(key="material", value="MATBG, theta = 1.10 deg",
         basis="the programme is a TBG programme (M22, M28); MATBG is also the only "
               "moire material with an existing cQED measurement to build on "
               "(arXiv:2406.13740)",
         up="magic-angle twisted TRILAYER at 1.71 deg -- the ONLY moire material with a "
            "published CPR-derived L_sq (arXiv:2403.02320), higher T_c, and a stiffer "
            "superconductor; but it is not the programme's material and M28's design is "
            "already anchored to it",
         down="a NbSe2 control strip -- fully de-risked (arXiv:2511.08466) but answers "
              "nothing about moire"),
    dict(key="L_sq", value="29.3 nH/sq",
         basis="M28's chosen operating point, reached by gating the measured 10-150 "
               "nH/sq window of arXiv:2403.02320 downward; it is the value the M28 "
               "fluxonium design actually uses",
         up="150 nH/sq -> same L in 5x fewer squares, so the ladder's N lever shrinks "
            "and contact separation gets harder; Z rises, which helps Q_contact",
         down="10 nH/sq -> a longer strip for the same L, a better N lever, but 15x more "
              "area exposed to twist disorder"),
    dict(key="strip_width_w", value="400 nm",
         basis="M28's w/xi = 2 point; narrow enough to stay 2D against vortex entry, "
               "wide enough to stay above the quasi-1D phase-slip regime",
         up="2 um -> 5x lower R_sheet contribution per square is unchanged, but contact "
            "resistance per contact falls 5x because R_c scales as rho_c/W",
         down="200 nm -> w ~ xi, quasi-1D; phase slips replace vortices and NO "
              "measurement exists for a moire superconductor in that regime"),
    dict(key="ladder_N_squares", value="[2, 5, 14.7, 40, 100]",
         basis="a factor 50 in N is a factor 50 lever on the contact term, which is what "
               "separates R_c from tan_delta; 14.7 is M28's design point and is kept as "
               "the ladder's centre so M30 measures the element M28 designed",
         up="a decade more (N = 300) -> C would have to fall below 0.5 fF, under stray; "
            "the top of the ladder is capacitance-limited, not material-limited",
         down="a factor 5 lever only -> the intercept uncertainty roughly triples"),
    dict(key="f_r_band", value="3.8 - 7.5 GHz",
         basis="inside the JTWPA 3-11 GHz band already catalogued in "
               "out/m13_readout_chain.json, and hf/k_B = 180-360 mK sits above the "
               "20 mK base so the TLS bath is in its ground state",
         up="10 GHz -> hf/k_B = 480 mK, a cleaner TLS ground state, but C falls toward "
            "stray and the amplifier band runs out",
         down="1 GHz -> much higher Z and a much better Q_contact, but hf/k_B = 48 mK is "
              "comparable to the base temperature and the TLS physics is thermally "
              "smeared"),
    dict(key="Qc_target", value="matched to the expected Q_i, decade by decade",
         basis="circle-fit precision on Q_i degrades as (1 + Q_i/Q_c); Q_c ~ Q_i is the "
               "flat part of that curve.  Because Q_i is the unknown, the ladder carries "
               "TWO coupling capacitors per N, spanning Q_c = 1e4 and 1e6",
         up="Q_c = 1e7 -> the resonance is too shallow to fit at n=1",
         down="Q_c = 1e3 -> Q_l is coupling-dominated and sigma(Q_i)/Q_i blows up by "
              "Q_i/Q_c"),
    dict(key="rho_c_design_target", value="100 Ohm.um on 10 um pads = 20 Ohm total",
         basis="the pre-registered gate on cooldown 2.  Wang et al., Science 342, 614 "
               "(2013) edge contacts to a 2D crystal reach ~100 Ohm.um; flaring the "
               "moire into a 10 um pad at each end gives 10 Ohm per contact.  The gate is "
               "set at 10x margin, rho_c <= 1000 Ohm.um",
         up="16000 Ohm.um -- the published moire figure of 3.2 kOhm under an ASSUMED "
            "2 um contact width -> the whole ladder is pinned below Q ~ 22 and the "
            "intercept resolves only to Q ~ 575; the experiment fails",
         down="10 Ohm.um, or a 25 um pad -> the intercept resolves into the 1e6 "
              "decade and the VIABLE band comes within reach even galvanically"),
    dict(key="target_T1_for_thresholds", value="100 us",
         basis="M28's headline requirement, Q_ind >= 3.2e6; it is also the level at "
               "which a moire fluxonium would be worth building against a 2D transmon "
               "at ~1 ms (Bland et al., Nature 647, 343 (2025))",
         up="1 ms -> Q_ind = 3.2e7, tan_delta = 3.1e-8, beyond any measured "
            "superinductor including grAl at 4.7e6",
         down="10 us -> Q_ind = 3.2e5; comparable to the realised 0-pi's T2E = 25.8 us, "
              "so still a meaningful device"),
    dict(key="sigma_Qi_target", value="1 % (1 sigma, per point)",
         basis="the level at which the 1/N intercept is limited by device-to-device "
               "fabrication scatter rather than by counting statistics; Tanaka's own "
               "circle fit reached 2.7% on Q_i = 1045",
         up="0.3 % -> 10x the averaging time, still hours not days",
         down="3 % -> the intercept uncertainty triples and the bottom kill band widens "
              "by half a decade"),
    dict(key="n_sys", value="3.0 quanta",
         basis="JTWPA n_add = 1.0 (out/m13_readout_chain.json) plus half a quantum of "
               "vacuum plus ~1.5 quanta for insertion loss between sample and amplifier; "
               "a deliberately pessimistic bookkeeping",
         up="12 quanta -- no TWPA at all, HEMT only -> 4x the averaging time, which is "
            "still hours.  THE TWPA IS OPTIONAL FOR THIS EXPERIMENT",
         down="1.5 quanta with a KI-TWPA (n_add = 0.4, arXiv:2402.11751) -> 2x faster"),
    dict(key="attenuation_ladder", value="[0, 20, 10, 10, 30] dB at 300K/50K/4K/still/MXC",
         basis="70 dB cold plus ~10 dB cable loss puts n = 1 at a VNA output of about "
               "-57 dBm and n = 1e6 at about +3 dBm, the whole sweep inside one "
               "instrument's range with a room-temperature step attenuator for the top "
               "decade.  Comparable to the 65 dB of arXiv:2407.09916 and the 70-84 dB "
               "of arXiv:2512.08059",
         up="90 dB -> quieter, but the VNA must run near +23 dBm for the saturation "
            "sweep and the MXC attenuator heat load rises 100x",
         down="50 dB -> the 4 K blackbody leak is only 40 dB down and the residual "
              "photon occupancy at the sample rises above 1e-2"),
    dict(key="participation_convention", value="exact distributed p(phi), section 2",
         basis="derived here and checked against BOTH published perturbative forms: it "
               "reduces to Kreidel's p = 2|delta_omega/omega| for omega L << Z0 and to "
               "L_k/L_total for a lumped mode",
         up="using 2|delta f/f| everywhere overstates p by up to 2x in the "
            "non-perturbative regime, which OVERSTATES the bound on tan_delta",
         down="using L_k/L_total everywhere understates p by 2x in the perturbative "
              "regime, which is the conservative direction for a bound"),
]


# ============================================================ section 2
def participation_exact(L, f, Z0=50.0):
    """Exact energy participation of a terminating inductance L on a shorted
    transmission-line stub of impedance Z0, at the fundamental mode.

    Derivation (mine).  Write I(z) = A cos(beta z + phi), V(z) = j Z0 A sin(beta z + phi)
    with z measured from the terminated end.  The boundary condition V(0)/I(0) = j omega L
    gives tan(phi) = omega L / Z0; the open end at z = l gives beta l + phi = pi/2, which
    is exactly the resonance condition tan(beta l) = Z0 / (omega L) quoted as Eq. 12 of
    arXiv:2511.08466.  Integrating the magnetic energy,

        p = sin(2 phi) / ( sin(2 phi)/2 + (pi/2 - phi) ).

    Limits:  omega L << Z0  ->  p -> 4 phi / pi = 2 |delta_omega/omega|  (Kreidel).
             omega L >> Z0  ->  p -> 1  (the sample IS the resonator).
    """
    w = 2.0 * np.pi * f
    phi = np.arctan2(w * L, Z0)
    return np.sin(2.0 * phi) / (np.sin(2.0 * phi) / 2.0 + (np.pi / 2.0 - phi)), phi


def stub_length(phi, f, eps_eff=6.34):
    """Physical length of the shorted stub, from beta l = pi/2 - phi."""
    beta = 2.0 * np.pi * f * np.sqrt(eps_eff) / C0
    return (np.pi / 2.0 - phi) / beta


def stub_capacitance(l, Z0=50.0, eps_eff=6.34):
    """Distributed capacitance of the stub: C' = sqrt(eps_eff)/(c Z0)."""
    return l * np.sqrt(eps_eff) / (C0 * Z0)


def section_participation():
    res = {}

    # -- 2a: the curve itself
    curve = []
    for wL_over_Z0 in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        phi = np.arctan(wL_over_Z0)
        p = np.sin(2 * phi) / (np.sin(2 * phi) / 2 + (np.pi / 2 - phi))
        curve.append(dict(wL_over_Z0=wL_over_Z0, phi_rad=round(float(phi), 5),
                          p=round(float(p), 5),
                          p_perturbative_2dfof=round(float(4 * phi / np.pi), 5),
                          stub_frac_of_quarterwave=round(float((np.pi / 2 - phi) / (np.pi / 2)), 5)))
    res["participation_curve"] = curve
    res["participation_curve_reading"] = (
        "p climbs through 0.5 at omega L / Z0 = 1.4 and reaches 0.99 only above "
        "omega L / Z0 ~ 80.  The perturbative form 2|df/f| tracks the exact curve to "
        "better than 5% only below omega L / Z0 ~ 0.35, i.e. exactly where p < 0.4.  "
        "There is no regime in which the measurement is both perturbative and sensitive."
    )

    # -- 2b: the published moire geometry, scored
    tk = LIT["tanaka_matbg_cqed"]
    p_tanaka = 2.0 * tk["frac_shift_ins_to_sc"]
    Q_matbg_lower = p_tanaka * tk["Qi"]
    # excess-loss variant: attribute the whole gap to the control resonator to the MATBG
    excess = 1.0 / tk["Qi"] - 1.0 / tk["control_Qi"]
    Q_matbg_excess = p_tanaka / excess
    res["published_moire_geometry"] = dict(
        source=tk["ref"],
        Qi=tk["Qi"], Qi_err=tk["Qi_err"], Qc=tk["Qc"], f_GHz=tk["f_r_GHz"],
        control_Qi=tk["control_Qi"], control_f_GHz=tk["control_f_GHz"],
        fractional_shift=tk["frac_shift_ins_to_sc"],
        p_from_2dfof=round(float(p_tanaka), 4),
        Q_MATBG_lower_bound=round(float(Q_matbg_lower), 1),
        tan_delta_upper_bound=round(float(1.0 / Q_matbg_lower), 5),
        Q_MATBG_from_excess_loss=round(float(Q_matbg_excess), 1),
        photon_number=tk["photon_number"],
        verdict="a BOUND, not a measurement, and it is weak because p = 2%.",
    )

    # -- 2c: the published vdW ladder, scored device by device.  Two conventions.
    zm = LIT["zaman_nbse2_KI"]
    rows = []
    Lg_ref = None
    for (dev, d_nm, W, Lu, f_s, f_al, Lk, Lsq, Qi) in zm["devices"]:
        # exact participation from the measured L_k at the measured loaded frequency
        p_exact, phi = participation_exact(Lk * 1e-9, f_s * 1e9)
        # lumped cross-check: p = 1 - (f_loaded/f_bare)^2  at fixed C
        p_lumped = 1.0 - (f_s / f_al) ** 2
        # bound on the sample's own loss, subtracting the spectator resonator
        floor = (1.0 - p_exact) / zm["spectator_Qi"]
        tand = max(1.0 / Qi - floor, 0.0) / p_exact
        rows.append(dict(device=dev, d_nm=d_nm, Lk_nH=Lk, Lsq_nH_per_sq=Lsq,
                         f_sample_GHz=f_s, f_Al_GHz=f_al, Qi=Qi,
                         p_exact=round(float(p_exact), 4),
                         p_lumped_1_minus_fsq=round(float(p_lumped), 4),
                         tan_delta_upper=float(f"{tand:.3e}"),
                         Q_sample_lower=round(float(1.0 / tand), 1),
                         figure_of_merit_pQi=round(float(p_exact * Qi), 1)))
    best = min(rows, key=lambda r: r["tan_delta_upper"])
    res["published_vdw_ladder"] = dict(
        source=zm["ref"], material="NbSe2 (NOT moire)", rows=rows,
        best_bound_device=best["device"],
        best_tan_delta_upper=best["tan_delta_upper"],
        best_Q_sample_lower=best["Q_sample_lower"],
        spectator_Qi=zm["spectator_Qi"],
        reading=(
            "twelve devices, single-photon, one chip family, and the best bound on the "
            "vdW element itself is tan_delta <= %.1e (Q >= %.0f).  Q_i does NOT fall "
            "monotonically with L_k: D2 has 2.3x the kinetic inductance of D6 and 2x the "
            "Q_i.  The 10x device-to-device scatter at fixed L_k is the signature of a "
            "per-device extrinsic channel -- which is what the authors themselves say: "
            "%s"
            % (best["tan_delta_upper"], best["Q_sample_lower"], zm["authors_attribution"])
        ),
        why_it_matters_for_M30=(
            "this is M30's experiment, already performed at single-photon level by the "
            "same group that made the MATBG measurement, on a vdW superconductor that is "
            "not moire.  It de-risks the fabrication and the analysis completely.  It "
            "also shows the ceiling of the terminating-inductor geometry as published: "
            "the highest participation anywhere in the twelve devices is p = %.2f, and "
            "the device giving the best loss bound sits at p = %.2f."
            % (max(r["p_exact"] for r in rows), best["p_exact"])
        ),
    )
    return res


# ============================================================ section 3
def section_design():
    """The ladder.  Five square counts, five frequencies, one flake, one feedline.

    The strip WIDTH is not a free aesthetic choice here.  At fixed square count N the
    inductance is fixed, but the physical length is N*w, and M28's phase law
    dphi/dl|max = 2/(3 sqrt(3) xi) makes the available phase headroom proportional to
    that length.  Widening a low-N rung therefore buys phase headroom AND lowers the
    contact resistance (R_c = 2 rho_c / w) at zero cost in inductance.  This is the
    opposite of M28's optimum, where narrower was better on every axis -- because a
    qubit wants a small footprint and a measurement wants a large signal.
    """
    res = {}
    Lsq = 29.3e-9           # CHOICE
    w_floor = 400e-9        # CHOICE: M28's w/xi = 2 floor, keeps the strip 2D
    xi = 200e-9             # arXiv:2403.02320 upper bound
    Kc_nA_per_um = 200.0    # arXiv:2403.02320, max sheet critical current
    headroom = 3.0          # CHOICE: phase safety factor over the zero-point phase
    dphi_dl = 2.0 / (3.0 * np.sqrt(3.0) * xi)       # rad/m, M28's law
    Ns = [2.0, 5.0, 14.7, 40.0, 100.0]     # CHOICE
    f_targets = [7.5e9, 6.5e9, 5.5e9, 4.5e9, 3.8e9]   # CHOICE, JTWPA band

    rows = []
    for N, f in zip(Ns, f_targets):
        L = Lsq * N
        wr = 2.0 * np.pi * f
        C = 1.0 / (wr ** 2 * L)
        Z = np.sqrt(L / C)
        p, phi = participation_exact(L, f)
        l_stub = stub_length(phi, f)
        C_stub = stub_capacitance(l_stub)
        I_zpf = np.sqrt(HBAR * wr / (2.0 * L))
        phi_zpf = 2.0 * np.pi * I_zpf * L / PHI0
        len_req = headroom * phi_zpf / dphi_dl
        w = max(w_floor, len_req / N)
        strip_len = N * w
        rows.append(dict(
            N_squares=N, strip_width_nm=round(float(w * 1e9), 1),
            strip_length_um=round(float(strip_len * 1e6), 3),
            L_nH=round(float(L * 1e9), 2),
            f_r_GHz=round(f / 1e9, 3), C_required_fF=round(float(C * 1e15), 3),
            Z_kohm=round(float(Z / 1e3), 2),
            wL_over_Z0=round(float(wr * L / 50.0), 1),
            participation=round(float(p), 5),
            equivalent_stub_length_um=round(float(l_stub * 1e6), 2),
            stub_distributed_C_fF=round(float(C_stub * 1e15), 3),
            hf_over_kB_mK=round(float(H * f / KB * 1e3), 1),
            I_zpf_nA=round(float(I_zpf * 1e9), 3),
            I_depairing_nA=round(float(Kc_nA_per_um * w * 1e6), 1),
            I_zpf_over_I_dep=round(float(I_zpf * 1e9 / (Kc_nA_per_um * w * 1e6)), 4),
            phase_zpf_rad=round(float(phi_zpf), 4),
            phase_available_rad=round(float(dphi_dl * strip_len), 3),
            phase_headroom_factor=round(float(dphi_dl * strip_len / phi_zpf), 2),
            width_over_xi=round(float(w / xi), 2),
            area_um2=round(float(strip_len * w * 1e12), 3),
        ))
    res["ladder"] = rows
    res["ladder_note"] = (
        "the 'equivalent stub length' and 'stub distributed C' columns are the same "
        "device read as a shorted CPW stub instead of a lumped LC.  They agree with the "
        "required C to within the conformal-mapping accuracy, which is the point: the "
        "M30 device is the Tanaka/Zaman geometry with the line shortened from a quarter "
        "wave (~5.8 mm) to a 4-46 um stub.  Nothing else about the fabrication changes."
    )
    res["ladder_headline"] = (
        "participation runs %.4f to %.5f across the ladder -- 50x better than the "
        "published moire geometry's 0.02 and about 2x better than the highest "
        "participation anywhere in the published vdW ladder."
        % (rows[0]["participation"], rows[-1]["participation"])
    )
    res["width_rule"] = dict(
        law="dphi/dl|max = 2/(3 sqrt(3) xi) = %.3f rad/um (M28, from arXiv:2403.02320)"
            % (dphi_dl / 1e6),
        rule="w = max(2 xi, headroom * phi_zpf / (N * dphi/dl)) with headroom = %.0f"
             % headroom,
        consequence="the two low-N rungs are drawn WIDE (%.0f nm and %.0f nm) and the "
                    "three high-N rungs sit at the 400 nm floor.  Widening also lowers "
                    "the contact resistance as 1/w, which is folded into the fit in "
                    "section 4."
                    % (rows[0]["strip_width_nm"], rows[1]["strip_width_nm"]),
        divergence_from_M28="M28 registered w/xi = 2 as 'the key geometric lever, "
                            "narrower is better on every axis'.  That is true for the "
                            "QUBIT and false for the MEASUREMENT.  The two optima "
                            "disagree and the M30 mask must not simply copy M28's "
                            "geometry.",
    )

    res["linearity_check"] = dict(
        I_zpf_nA_range=[rows[-1]["I_zpf_nA"], rows[0]["I_zpf_nA"]],
        I_depairing_basis="K_c^max = 200 nA/um (arXiv:2403.02320), so I_dep = K_c * w",
        worst_ratio_I_zpf_over_I_dep=max(r["I_zpf_over_I_dep"] for r in rows),
        conservative_alternative=(
            "if one instead compares against the JUNCTION critical currents quoted in "
            "the same paper (I_c1 = 6 nA on the hole branch, 45 nA on the electron "
            "branch, for 1 um x 1 um and 0.5 um x 2.5 um junctions), the top rung's "
            "zero-point current is a comparable fraction of the 6 nA figure.  Those are "
            "junction critical currents, not the sheet depairing current, and the sheet "
            "figure is the right comparison for a continuous strip -- but the ambiguity "
            "is real and it is the reason the phase headroom factor is set to 3 rather "
            "than to 1.2 as in M28."),
        Kerr_reference="NbSe2 kinetic inductors show K/2pi = -0.008 to -14.7 Hz/photon "
                       "(arXiv:2511.08466); a moire strip with 20-200x the sheet "
                       "inductance and a far smaller depairing current will be much more "
                       "nonlinear, so the n = 1 point is mandatory and the Kerr shift is "
                       "itself a measurable.  SPECULATION on the magnitude.",
        verdict="with the width rule applied, the single-photon current is at most "
                "%.1f per cent of the sheet depairing current and the zero-point phase "
                "uses at most one third of the available phase, so every rung is a "
                "linear resonator at n = 1.  The high-power end of the sweep WILL "
                "bifurcate, exactly as arXiv:2511.08466 saw at -82 dBm, and that "
                "bifurcation is the in-situ calibration of the depairing current."
                % (100 * max(r["I_zpf_over_I_dep"] for r in rows)),
    )
    return res, rows


# ============================================================ section 4
def section_contacts(ladder):
    """Separating contact loss from intrinsic loss: the 1/N fit."""
    res = {}
    Lsq = 29.3e-9

    # Contacts are specified as a contact resistivity rho_c in Ohm.um, because the
    # ladder's rungs have different strip widths (section 3) and R_c = 2 rho_c / w_pad.
    # The contact PAD may be wider than the strip; the pad width is a CHOICE.
    W_PAD_UM = 10.0     # CHOICE: flared contact pads, 10 um wide
    RHO_SCENARIOS = [
        (100.0, "edge contact to a 2D crystal at the demonstrated 100 Ohm.um "
                "(Wang et al., Science 342, 614 (2013)); this is the design target"),
        (1000.0, "10x worse than the demonstrated figure, allowing for the flat band's "
                 "low carrier density under the contact"),
        (16000.0, "the one published moire rf number, 3.2 kOhm total "
                  "(arXiv:2406.13742), converted under an ASSUMED 2 um contact width -- "
                  "the width is not available to me and this conversion is flagged"),
    ]

    # 4a: what a single device would report, per rung
    scenarios = []
    for rho, label in RHO_SCENARIOS:
        Rc = 2.0 * rho / W_PAD_UM
        rows = []
        for r in ladder:
            wr = 2.0 * np.pi * r["f_r_GHz"] * 1e9
            L = r["L_nH"] * 1e-9
            rows.append(dict(N_squares=r["N_squares"], f_r_GHz=r["f_r_GHz"],
                             Z_kohm=r["Z_kohm"],
                             Q_contact_limit=round(float(wr * L / Rc), 1)))
        scenarios.append(dict(rho_c_ohm_um=rho, contact_pad_width_um=W_PAD_UM,
                              R_c_total_ohm=round(float(Rc), 1), label=label, rows=rows,
                              Q_contact_range=[rows[0]["Q_contact_limit"],
                                               rows[-1]["Q_contact_limit"]]))
    res["contact_ceiling"] = scenarios
    res["contact_specification"] = dict(
        rho_c_definition="R_c(total) = 2 rho_c / W_pad, two contacts in series",
        contact_pad_width_um=W_PAD_UM,
        why_a_pad="the strip is 400 nm to 1 um wide but the CONTACT need not be: flaring "
                  "the moire into a 10 um pad at each end before the edge contact cuts "
                  "R_c by up to 25x at no cost in inductance, because the flared region "
                  "contributes a fraction of a square.  This is the cheapest single "
                  "improvement available to the design and it is a layout change.",
        catastrophic_case="if the 3.2 kOhm of arXiv:2406.13742 is a fixed lumped "
                          "interface resistance that does NOT scale with contact width, "
                          "no pad helps and the galvanic design is dead on arrival.  "
                          "That is exactly what kill K0 tests, by DC, before cooling.",
    )
    res["contact_ceiling_reading"] = (
        "Q_contact = omega L / R_c = Z / R_c.  At the published moire contact resistance "
        "the whole ladder is pinned below Q ~ 10-100 and the experiment measures the "
        "contacts.  This is the single largest risk in the design and it is why the "
        "contact resistance is a PRE-REGISTERED GATE measured by DC 4-probe on a "
        "companion Hall bar before the resonator chip is cooled."
    )

    # 4b: the 1/N fit.  Forward-model, then fit, then propagate.
    def fit_intercept(rho_c, tand_true, sigma_frac, Lsq=Lsq, n_devices_per_N=1):
        """Weighted least squares of y = 1/Q_i against the contact regressor.

        Model:  1/Q_i(rung) = tan_delta + (2 rho_c / W_pad) / (omega L_sq N).
        The intrinsic loss is the INTERCEPT and the contact resistivity is the SLOPE.
        Returns (intercept, sigma_intercept, rho_c_recovered, sigma_rho_c).
        """
        Ns_ = np.array([r["N_squares"] for r in ladder], float)
        fs_ = np.array([r["f_r_GHz"] * 1e9 for r in ladder], float)
        w_ = 2.0 * np.pi * fs_
        Rc = 2.0 * rho_c / W_PAD_UM
        regressor = 1.0 / (w_ * Lsq * Ns_)              # multiplies R_c
        y = tand_true + Rc * regressor
        sig = sigma_frac * y                            # frac error on Q_i -> on 1/Q_i
        A = np.column_stack([np.ones_like(y), regressor * 2.0 / W_PAD_UM])
        Wm = np.diag(1.0 / sig ** 2) * n_devices_per_N
        cov = np.linalg.inv(A.T @ Wm @ A)
        beta = cov @ (A.T @ Wm @ y)
        return (float(beta[0]), float(np.sqrt(cov[0, 0])),
                float(beta[1]), float(np.sqrt(cov[1, 1])))

    fit_rows = []
    for rho, _lbl in RHO_SCENARIOS:
        for tand_true in [3.1e-7, 3.1e-6, 3.1e-5, 3.1e-4]:
            for sfrac in [0.01, 0.03]:
                b0, s0, b1, s1 = fit_intercept(rho, tand_true, sfrac)
                fit_rows.append(dict(
                    rho_c_ohm_um=rho, tan_delta_true=tand_true, sigma_Qi_frac=sfrac,
                    intercept=float(f"{b0:.3e}"),
                    sigma_intercept=float(f"{s0:.3e}"),
                    detected_at_3sigma=bool(b0 > 3.0 * s0),
                    resolvable_floor_tan_delta=float(f"{3.0*s0:.3e}"),
                    resolvable_floor_Q=round(float(1.0 / (3.0 * s0)), 1),
                    rho_c_recovered_ohm_um=round(float(b1), 1),
                    sigma_rho_c_ohm_um=round(float(s1), 2)))
    res["intercept_fit"] = fit_rows

    # 4c: the headline -- the loss floor the experiment can reach, per rho_c
    floors = {}
    for rho, _lbl in RHO_SCENARIOS:
        b0, s0, _, _ = fit_intercept(rho, 0.0, 0.01)
        floors[f"rho_c_{int(rho)}_ohm_um"] = dict(
            R_c_total_ohm=round(float(2 * rho / W_PAD_UM), 1),
            three_sigma_tan_delta_floor=float(f"{3*s0:.3e}"),
            equivalent_Q_ceiling=round(float(1.0 / (3 * s0)), 1),
            equivalent_T1_us_at_M28_fluxonium=round(
                float(1.0 / (3 * s0) / 3.218e6 * 100.0), 3))
    res["measurement_floor_vs_contact"] = floors
    res["measurement_floor_reading"] = (
        "with sigma(Q_i)/Q_i = 1 per cent per device and the 5-rung ladder, the 3-sigma "
        "floor on the intrinsic loss tangent is set almost entirely by the contacts.  At "
        "the design target of 100 Ohm.um the experiment resolves Q up to ~{:.3g}; at the "
        "moire-derived 16000 Ohm.um it resolves only up to ~{:.3g}, which is not enough "
        "to distinguish 'weak' from 'dead'.".format(
            floors['rho_c_100_ohm_um']['equivalent_Q_ceiling'],
            floors['rho_c_16000_ohm_um']['equivalent_Q_ceiling'])
    )

    # 4c-bis: the same floor if the ladder is run at the TOP of the gate-tunable
    # L_sq window instead of at M28's operating point.  The contact term is
    # R_c/(omega L_sq N), so it is inversely proportional to L_sq: the measurement
    # gets 5x more sensitive at 150 nH/sq, while the QUBIT design does not care.
    floors_150 = {}
    for rho, _lbl in RHO_SCENARIOS:
        b0, s0, _, _ = fit_intercept(rho, 0.0, 0.01, Lsq=150e-9)
        floors_150[f"rho_c_{int(rho)}_ohm_um"] = dict(
            three_sigma_tan_delta_floor=float(f"{3*s0:.3e}"),
            equivalent_Q_ceiling=round(float(1.0 / (3 * s0)), 1))
    res["measurement_floor_at_150nH_per_sq"] = dict(
        rows=floors_150,
        why="L_sq is gate-tunable over 10-150 nH/sq in situ (arXiv:2403.02320).  M28 "
            "chose 29.3 nH/sq because that is the right point for the FLUXONIUM.  It is "
            "the wrong point for the MEASUREMENT: the contact term scales as 1/L_sq, so "
            "running the ladder at the top of the window buys a factor 5 in sensitivity "
            "for free, on the same device, by turning a gate voltage.",
        caveat="the gate voltage that maximises L_sq is not the gate voltage that "
               "maximises the superfluid density, and the intrinsic loss is itself a "
               "function of filling.  The gate is therefore a PHYSICAL axis (report "
               "tan_delta(nu)) and NOT a second fit axis; the 1/N ladder must be refit "
               "at each filling.  Treating the gate as a lever on R_c would assume the "
               "intrinsic loss is filling-independent, which is exactly the thing being "
               "measured.")

    # 4c-ter: the geometric ceiling of the galvanic design, independent of the fit
    ceiling_rows = []
    for C_min_fF, f_min_GHz in [(0.5, 2.0), (1.0, 3.0), (2.0, 4.0)]:
        Zmax = 1.0 / (2 * np.pi * f_min_GHz * 1e9 * C_min_fF * 1e-15)
        for rho, _lbl in RHO_SCENARIOS:
            Rc = 2.0 * rho / W_PAD_UM
            ceiling_rows.append(dict(C_stray_floor_fF=C_min_fF, f_min_GHz=f_min_GHz,
                                     Z_max_kohm=round(float(Zmax / 1e3), 1),
                                     rho_c_ohm_um=rho, R_c_total_ohm=round(float(Rc), 1),
                                     Q_single_device_ceiling=round(float(Zmax / Rc), 1)))
    res["geometric_ceiling"] = dict(
        rows=ceiling_rows,
        statement="Q_contact = Z/R_c and Z = 1/(omega C) is bounded above by the stray "
                  "capacitance floor and by the requirement hf >> k_B T.  No ladder "
                  "cleverness beats Z_max/R_c by more than the fit leverage, which this "
                  "5-rung design supplies as roughly one decade.",
        consequence="THE HONEST LIMIT OF THE GALVANIC DESIGN.  It decides DEAD vs WEAK "
                    "cleanly at any plausible contact resistance, decides WEAK vs "
                    "MARGINAL only if R_c <= ~20 Ohm, and CANNOT certify the VIABLE "
                    "band (Q >= 3.2e6) at all.  Certifying VIABLE requires taking the "
                    "contacts out of the resonant loop.")

    # 4d: the contactless alternative, costed
    eps_hbn = LIT["hBN"]["eps_perp"]
    eps0 = 8.8541878128e-12
    series_rows = []
    for t_nm, area_um2 in [(3.0, 10.0), (10.0, 25.0), (30.0, 100.0)]:
        Cs = eps0 * eps_hbn * (area_um2 * 1e-12) / (t_nm * 1e-9)
        # two in series with the resonator's own C
        Cs_eff = Cs / 2.0
        f = 5.5e9
        C_res = 1.0 / ((2 * np.pi * f) ** 2 * (29.3e-9 * 14.7))
        p_cap = C_res / Cs_eff
        series_rows.append(dict(
            hBN_thickness_nm=t_nm, plate_area_um2=area_um2,
            C_series_each_fF=round(float(Cs * 1e15), 1),
            resonator_C_fF=round(float(C_res * 1e15), 3),
            capacitive_participation_of_hBN=round(float(p_cap), 5),
            loss_added=float(f"{p_cap*LIT['hBN']['tan_delta_upper']:.2e}"),
            equivalent_Q=round(float(1.0 / (p_cap * LIT["hBN"]["tan_delta_upper"])), 0)))
    res["contactless_option"] = dict(
        idea="feed the resonant current into and out of the moire strip through two "
             "large-area hBN parallel-plate capacitors instead of galvanic edge "
             "contacts, removing R_c from the loop entirely",
        rows=series_rows,
        hBN_tan_delta_used=LIT["hBN"]["tan_delta_upper"],
        hBN_ref=LIT["hBN"]["ref"],
        the_blocker=(
            "a fully floating moire island can be given a displacement field but NOT a "
            "filling factor: a top and bottom gate induce equal and opposite surface "
            "charge on a floating conductor, leaving the net density unchanged.  MATBG "
            "without control of nu is useless.  The DC path cannot be removed."),
        the_fix=(
            "make the mode DIFFERENTIAL and put ONE galvanic DC bias contact at the "
            "strip's electrical centre, which is the rf voltage node.  The rf current "
            "leaked into that contact scales as the square of the fractional placement "
            "error, so a 1% placement error suppresses the contact loss by ~1e4.  This "
            "is a layout change, not a new process.  SPECULATION: no such device has "
            "been built for a moire superconductor, and the symmetry of the mode has to "
            "be verified by a full-wave solve before it is trusted."),
        recommendation=(
            "carry BOTH on the same mask.  The galvanic ladder is the primary "
            "measurement because it is the published, de-risked process "
            "(arXiv:2511.08466); the differential/node-contact variant is the escape "
            "route if the DC 4-probe pre-gate shows R_c(total) > 200 Ohm."),
    )

    # 4e: the node-contact variant, quantified
    # A differential (balanced) LC mode puts a virtual rf ground at the strip's
    # electrical centre.  One galvanic DC bias contact placed there, behind an rf
    # choke of impedance Z_bias, diverts a current eps*V_max/(R_c + Z_bias), where
    # eps is the FRACTIONAL PLACEMENT ERROR of the contact along the strip.
    #   1/Q_node = eps^2 R_c Z / |R_c + Z_bias|^2 ,  Z = 1/(omega C) = omega L.
    # (MY ARITHMETIC.  P_diss = |I_div|^2 R_c / 2, U = C V_max^2 / 2, 1/Q = P/(omega U).)
    node_rows = []
    Z_res = 14.88e3     # the ladder centre, from section 3
    for eps in [0.10, 0.03, 0.01]:
        for Rc in [3200.0, 200.0, 20.0]:
            for Zbias in [50.0, 1.0e4, 1.0e5, 1.0e6]:
                inv_Q = eps ** 2 * Rc * Z_res / abs(Rc + Zbias) ** 2
                node_rows.append(dict(
                    placement_error_frac=eps, R_c_ohm=Rc, Z_bias_ohm=Zbias,
                    inv_Q_from_node_contact=float(f"{inv_Q:.3e}"),
                    Q_node_ceiling=float(f"{1.0/inv_Q:.3e}")))
    res["node_contact_variant"] = dict(
        idea="a differential lumped mode: the shunt capacitance is split into two halves, "
             "one at each end of the moire strip, both referenced to ground, so the "
             "strip's electrical centre is a virtual rf ground.  A single galvanic DC "
             "bias contact is placed there, behind an on-chip rf choke.  Filling factor "
             "is controllable; the contact carries no rf current to the extent that the "
             "mode is balanced.",
        formula="1/Q_node = eps^2 R_c Z / |R_c + Z_bias|^2 (MY ARITHMETIC)",
        Z_resonator_ohm=Z_res,
        rows=node_rows,
        reading="with a bare 50 Ohm bias line the node contact is WORSE than a galvanic "
                "pair -- the diverted current sees no impedance and the loss is enormous. "
                "The choke is not optional.  With Z_bias = 1e5 Ohm (a high-impedance "
                "meander or an on-chip spiral) and a 1 per cent placement error, the "
                "node-contact ceiling clears 1e7 even at the published 3.2 kOhm contact "
                "resistance, which is the only route in this design to certifying the "
                "VIABLE band.",
        risks=[
            "SPECULATION: no differential node-contacted moire resonator has been built. "
            "The balance of the mode has to be verified by a full-wave solve before the "
            "eps^2 suppression is trusted; the programme has an accepted solver "
            "(out/m12_fullwave_validate.json, TE101 rel err 4.7e-7).",
            "the choke is a new element with its own loss and its own self-resonance, "
            "which must sit above the ladder's 7.5 GHz top.",
            "a balanced mode couples to the feedline differently and Q_c has to be "
            "re-engineered; a symmetric two-port tap is the standard fix.",
        ],
    )
    return res


# ============================================================ section 5
def section_thresholds():
    """What Q_i values mean, tied to a target T1."""
    res = {}
    # M28's fluxonium channel: Gamma_1 = (2 E_L/hbar)(1/Q_ind)|<0|phi|1>|^2
    E_L = 0.38e9 * H
    phi01 = 2.5959
    prefactor = 2.0 * E_L / HBAR * phi01 ** 2      # 1/s, per unit 1/Q_ind
    bands = []
    for T1 in [1e-6, 1e-5, 1e-4, 1e-3]:
        Q = prefactor * T1
        bands.append(dict(target_T1_us=T1 * 1e6,
                          Q_ind_required=float(f"{Q:.3e}"),
                          tan_delta_tolerated=float(f"{1.0/Q:.3e}")))
    res["m28_fluxonium_requirement"] = dict(
        formula="Gamma_1 = (2 E_L/hbar)(1/Q_ind)|<0|phi|1>|^2, T->0",
        E_L_GHz=0.38, phi_01=phi01, participation="p ~ 1 by construction",
        rows=bands,
        source="out/m28_superinductance.json:loss; reproduced here")

    # The verdict bands
    verdict = [
        dict(band="VIABLE", Q_i=">= 3.2e6", tan_delta="<= 3.1e-7",
             meaning="a moire-superinductor fluxonium reaches T1 >= 100 us.  This would "
                     "also make MATBG the best superinductor material ever measured, "
                     "ahead of granular Al at 4.72e6 (arXiv:2411.12611).",
             prior="SPECULATION that this is reachable.  No van der Waals element of any "
                   "kind has been measured above tan_delta = 1.7e-5 "
                   "(arXiv:2512.08059), which is 55x worse than this band's floor."),
        dict(band="MARGINAL", Q_i="3.2e5 - 3.2e6", tan_delta="3.1e-6 - 3.1e-7",
             meaning="T1 = 10-100 us.  Comparable to the realised 0-pi's T2E = 25.8 us "
                     "(arXiv:1910.07542) and to the rhombi chain's T1 = 70 us "
                     "(arXiv:1311.6521).  Worth building; not competitive with a 2D "
                     "transmon at ~1 ms (Bland et al., Nature 647, 343 (2025)).",
             prior="would require the moire element to beat every measured vdW element "
                   "by 5-50x."),
        dict(band="WEAK", Q_i="3.2e4 - 3.2e5", tan_delta="3.1e-5 - 3.1e-6",
             meaning="T1 = 1-10 us.  This is where the vdW literature actually sits: "
                     "all-vdW transmons at T1 = 1.11-1.67 us mean and tan_delta = "
                     "1.7e-5 to 1.6e-4 (arXiv:2512.08059); best graphene qubit 716 ns "
                     "(arXiv:2506.04959).  Publishable physics, not a competitive qubit.",
             prior="THE MOST LIKELY OUTCOME on the current evidence."),
        dict(band="DEAD", Q_i="< 3.2e4", tan_delta="> 3.1e-5",
             meaning="route B (M28) is dead and so is every moire CIRCUIT qubit that "
                     "needs the moire element to carry participation of order one.",
             prior="entirely compatible with every number published to date, including "
                   "the Q >= 10-50 bound extractable from arXiv:2406.13740."),
    ]
    res["verdict_bands"] = verdict

    # The dilution escape, and why it does not save a superinductor
    dil = []
    for p in [1.0, 0.5, 0.1, 0.01]:
        for tand in [3.1e-5, 3.1e-6]:
            f = 6e9
            T1 = 1.0 / (2 * np.pi * f * p * tand)
            dil.append(dict(participation=p, tan_delta=tand, f_GHz=f / 1e9,
                            T1_us=round(float(T1 * 1e6), 2)))
    res["dilution_escape"] = dict(
        formula="T1 = 1/(2 pi f p tan_delta)  (Balgley et al., arXiv:2512.08059)",
        rows=dil,
        valid_for="a JUNCTION or a CAPACITOR, which can be shrunk until p ~ 1%: "
                  "arXiv:2512.08059 explicitly projects T1 ~ 160 us for a conventional "
                  "transmon with a p = 1% NbSe2/WSe2 junction.",
        NOT_valid_for="a SUPERINDUCTOR.  A superinductor's entire function is to carry "
                      "essentially all of the circuit's inductive energy, so p ~ 1 is "
                      "not a design choice that can be relaxed.  Diluting a moire "
                      "superinductor to p = 1% means it is no longer the inductor.  "
                      "This asymmetry is why M30's kill criteria bind route B and do "
                      "NOT bind a moire gatemon.",
    )

    # what the topological route actually needs from this number
    res["what_it_gates"] = dict(
        gates_hard=[
            "M28 route B (moire superinductance / moire fluxonium): p ~ 1, so Q_i IS the "
            "answer.",
            "any moire cos(2phi) or protected circuit element.",
            "any resonator whose inductor is moire, including a moire-based parametric "
            "amplifier or KID.",
        ],
        gates_softly=[
            "readout of a topological qubit: a dispersive parity measurement needs the "
            "READOUT resonator's Q_i >> Q_c ~ 1e3-1e4 (out/m13_readout_chain.json), but "
            "that resonator can be Nb.  A low moire Q_i forces the readout circuit off "
            "the flake, costing area and a coupling capacitor, not the physics.",
            "M15's transmon: the junction can in principle be diluted, so a bad Q_i "
            "degrades but does not forbid it.",
        ],
        does_not_gate=[
            "the QAH/SC homojunction physics of M22 section C -- a chiral Majorana edge "
            "mode does not care about microwave loss.",
            "out/m23_majorana.json's argument that a Majorana qubit's lifetime follows "
            "from the gap and the parity poisoning rate rather than from a material "
            "relaxation time.  M30 does NOT gate the topological programme.  It gates "
            "the CIRCUIT programme and the cQED readout of the topological programme.",
        ],
        honest_caveat=(
            "M22 section E.7 is right that loss matters less for a topological qubit "
            "than for a circuit qubit.  M30 is still the cheapest decisive experiment "
            "because it is the only one that returns a NUMBER rather than a design, and "
            "because its fabrication is a strict subset of every other device the "
            "programme wants to build."),
    )
    return res


# ============================================================ section 6
def section_chain(ladder):
    """Wiring, attenuation, photon number, amplifier, measurement time."""
    res = {}
    n_add = LIT["amplifier"]["n_add"]
    n_sys = 3.0                                  # CHOICE
    atten = [0.0, 20.0, 10.0, 10.0, 30.0]        # CHOICE: 300K/50K/4K/still/MXC
    T_stage = [300.0, 50.0, 4.0, 0.7, 0.02]
    cable_dB = 10.0
    total_dB = sum(atten) + cable_dB

    def nbar(P_W, f, Qi, Qc):
        """n_r = 2 kappa P / (hbar omega (kappa+gamma)^2), arXiv:2511.08466 Eq. 5."""
        w = 2 * np.pi * f
        kappa = w / Qc
        gamma = w / Qi
        return 2.0 * kappa * P_W / (HBAR * w * (kappa + gamma) ** 2)

    def P_for_nbar(n, f, Qi, Qc):
        w = 2 * np.pi * f
        kappa = w / Qc
        gamma = w / Qi
        return n * HBAR * w * (kappa + gamma) ** 2 / (2.0 * kappa)

    # 6a: the power the VNA must produce, across the plausible Q_i range
    prows = []
    for Qi in [1e3, 1e4, 1e5, 1e6]:
        Qc = Qi                                   # CHOICE: matched coupling
        for n in [0.1, 1.0, 1e3, 1e6]:
            f = 5.5e9
            P = P_for_nbar(n, f, Qi, Qc)
            P_dBm = 10 * np.log10(P * 1e3)
            prows.append(dict(Qi=Qi, Qc=Qc, nbar=n, f_GHz=f / 1e9,
                              P_sample_dBm=round(float(P_dBm), 1),
                              VNA_output_dBm=round(float(P_dBm + total_dB), 1)))
    res["drive_power"] = dict(rows=prows, total_line_attenuation_dB=total_dB,
                              ladder_dB=atten, cable_loss_dB=cable_dB,
                              formula=LIT["fit_methods"]["photon_number"])

    # 6b: residual thermal photons at the sample
    def n_th(T, f):
        x = H * f / (KB * T)
        return 1.0 / np.expm1(x)

    f = 5.5e9
    n_res = 0.0
    for i, (a_dB, T) in enumerate(zip(atten, T_stage)):
        A = 10 ** (a_dB / 10.0)
        downstream = 10 ** (sum(atten[i + 1:]) / 10.0)
        contrib = (1.0 - 1.0 / A) * n_th(T, f) / downstream if A > 1 else 0.0
        n_res += contrib
    n_res += n_th(300.0, f) / 10 ** (total_dB / 10.0)
    res["thermal_photons"] = dict(
        f_GHz=f / 1e9, residual_nbar_at_sample=float(f"{n_res:.2e}"),
        requirement="<< 0.1 so the n = 1 point is drive-defined, not thermally defined",
        passes=bool(n_res < 0.01),
        note="the standard attenuator-chain bookkeeping; each stage contributes "
             "(1 - 1/A_j) n_th(T_j) divided by the attenuation that follows it.")

    # 6c: heat load at the mixing chamber at the top of the power sweep
    P_top_VNA_dBm = max(r["VNA_output_dBm"] for r in prows)
    P_at_MXC_in_dBm = P_top_VNA_dBm - (sum(atten[:-1]) + cable_dB)
    P_at_MXC_in_W = 10 ** (P_at_MXC_in_dBm / 10.0) * 1e-3
    diss = P_at_MXC_in_W * (1 - 10 ** (-atten[-1] / 10.0))
    res["heat_load"] = dict(
        worst_case_VNA_output_dBm=round(float(P_top_VNA_dBm), 1),
        power_into_MXC_attenuator_nW=round(float(P_at_MXC_in_W * 1e9), 3),
        dissipated_at_MXC_nW=round(float(diss * 1e9), 3),
        typical_MXC_cooling_power_uW_at_20mK=15.0,
        margin=round(float(15e-6 / max(diss, 1e-30)), 1),
        verdict="the worst point of the whole campaign dissipates a few nW at the mixing "
                "chamber, three to four orders below a standard XLD's cooling power.  "
                "The power sweep is not thermally limited.")

    # 6d: measurement time
    def trace_time(Qi, Qc, f, sigma_frac, n=1.0, n_pts=201, n_sys=n_sys):
        """Time to reach a fractional uncertainty sigma_frac on Q_i at photon number n.

        Working rule (MINE, not from a paper): the circle fit's fractional error on Q_i
        scales as (1 + Q_i/Q_c) / (SNR_amplitude * sqrt(N_points)).  The method itself is
        Probst et al. (arXiv:1410.3365), optimised point placement per Baity et al.,
        Phys. Rev. Research 6, 013329 (2024); the SCALING is a design rule, flagged.
        """
        P = P_for_nbar(n, f, Qi, Qc)
        need_snr_amp = (1.0 + Qi / Qc) / (sigma_frac * np.sqrt(n_pts))
        need_snr_pow = need_snr_amp ** 2
        # noise power referred to the sample in bandwidth B: n_sys * hbar * omega * B
        B = P / (need_snr_pow * n_sys * HBAR * 2 * np.pi * f)
        return n_pts / B, B, P

    trows = []
    for Qi in [1e3, 1e4, 1e5, 1e6]:
        for sfrac in [0.01, 0.003]:
            t, B, P = trace_time(Qi, Qi, 5.5e9, sfrac)
            trows.append(dict(Qi=Qi, Qc=Qi, sigma_Qi_frac=sfrac,
                              IF_bandwidth_Hz=float(f"{B:.3g}"),
                              trace_time_s=float(f"{t:.3g}"),
                              P_sample_dBm=round(float(10 * np.log10(P * 1e3)), 1)))
    res["trace_time"] = dict(rows=trows, n_sys=n_sys, n_points=201,
                             method=LIT["fit_methods"]["circle_fit"],
                             scaling_rule="MINE, flagged: sigma(Q_i)/Q_i = "
                                          "(1 + Q_i/Q_c)/(SNR_amp sqrt(N_pts))")

    # 6e: the campaign
    n_res_dev = 10          # 5 square counts x 2 coupling capacitors
    n_powers = 15           # n = 0.1 to 1e6, one point per half decade
    n_temps = 12            # 20 mK to 1.2 K
    n_gates = 9             # filling factors across the dome plus 2 controls
    t_trace = max(r["trace_time_s"] for r in trows if r["sigma_Qi_frac"] == 0.01)
    t_settle_T = 20 * 60.0  # arXiv:2512.08059: "stabilized for at least 20 minutes"
    t_settle_V = 5.0
    acq_power = n_res_dev * n_powers * t_trace
    acq_temp = n_temps * (t_settle_T + n_res_dev * t_trace)
    acq_gate = n_gates * (t_settle_V + n_res_dev * t_trace)
    total_s = acq_power + acq_temp + acq_gate
    res["campaign"] = dict(
        n_resonators=n_res_dev, n_power_points=n_powers, n_temperatures=n_temps,
        n_gate_points=n_gates,
        worst_trace_time_s=round(float(t_trace), 3),
        power_sweep_hours=round(float(acq_power / 3600), 2),
        temperature_sweep_hours=round(float(acq_temp / 3600), 2),
        gate_sweep_hours=round(float(acq_gate / 3600), 2),
        total_acquisition_hours=round(float(total_s / 3600), 2),
        dominated_by="temperature settling, not signal to noise",
        cooldown_days=2.0,
        verdict="THE MEASUREMENT ITSELF TAKES A DAY.  Everything else in the schedule is "
                "fabrication and cooldowns.  That is the whole argument for doing it "
                "first.")

    # 6f: bill of materials
    res["bill_of_materials"] = dict(
        rf_lines_in=1, rf_lines_out=1,
        dc_lines=4,
        fast_lines=0,
        instruments=["2-port VNA", "DC voltage source for the gates",
                     "lock-in + current source for the companion Hall bar",
                     "dilution refrigerator with a 20 mK base"],
        amplifier=dict(
            baseline="HEMT only (n_sys ~ 12): the campaign lengthens by 4x, to about "
                     "four days of acquisition.  STILL FEASIBLE.",
            preferred="JTWPA 3-11 GHz (n_add = 1.0, out/m13_readout_chain.json) with two "
                      "isolators on the signal port",
            conclusion="THE PARAMETRIC AMPLIFIER IS OPTIONAL FOR M30.  It is mandatory "
                       "for M15, which needs single-shot readout."),
        filtering="pi-filter at 3 K and a QFilter-class low-pass at the mixing chamber on "
                  "every DC line (as arXiv:2406.13740); Cryoperm magnetic shield around "
                  "the sample stage (as arXiv:2512.08059); flux-trapping holes etched in "
                  "the ground plane (as arXiv:2407.09916)",
        attenuation_dB=atten, cable_loss_dB=cable_dB, total_dB=total_dB,
    )
    return res


# ============================================================ section 7
def section_fab_and_kills():
    res = {}
    res["fabrication"] = [
        dict(step=1, what="base layer",
             how="250 nm Nb (or MBE Al) on high-resistivity / float-zone Si; "
                 "photolithography, CF4 RIE, O2 descum, Ar surface etch; flux-trapping "
                 "holes in the ground plane; buffered-oxide etch of residual fluoride",
             source="arXiv:2407.09916 and arXiv:2511.08466 verbatim",
             risk="low -- a published recipe that yields Q_0 = 1e5-6e6"),
        dict(step=2, what="graphite or Ti/Al back gate in an etched window",
             how="e-beam lithography and evaporation inside the resonator cutout",
             source="arXiv:2406.13740",
             risk="low"),
        dict(step=3, what="the magic-angle stack",
             how="tear-and-stack hBN/tBLG/hBN at 1.10 +/- 0.05 deg in an argon glovebox; "
                 "contact-mode AFM clean of the bottom hBN; chloroform removal of "
                 "polycarbonate",
             source="arXiv:2406.13740 Methods",
             risk="HIGH AND DOMINANT -- this is the schedule.  Twist-angle yield is the "
                  "binding constraint and it is the SAME constraint M15 faces."),
        dict(step=4, what="pattern the strips",
             how="e-beam lithography + RIE into the five-rung ladder plus a companion "
                 "Hall bar on the same flake",
             source="standard",
             risk="medium -- the 400 nm strip width is at the edge of comfortable RIE "
                  "definition through an encapsulated stack"),
        dict(step=5, what="edge contacts",
             how="RIE to expose the edge, in-situ Ar ion milling, angled Ti/Al (or "
                 "NbN) evaporation with substrate rotation",
             source="arXiv:2511.08466 Supp. section 2 ('Optimization of the Contact "
                    "Resistance between hBN encapsulated NbSe2 and Al')",
             risk="HIGH -- this is the pre-registered gate, see kill K2"),
        dict(step=6, what="the shunt capacitor and coupling capacitor",
             how="a 1-8 fF interdigitated or parallel-plate Nb structure and a "
                 "feedline coupling capacitor sized for the two Q_c decades",
             source="standard",
             risk="medium -- 0.6 fF at the ladder top is at the stray-capacitance floor "
                  "and must be verified by a full-wave solve, which the programme "
                  "already has an accepted solver for (out/m12_fullwave_validate.json)"),
        dict(step=7, what="DC characterisation before cooldown 2",
             how="4-probe R(T), R(nu), contact resistance and a Landau fan on the "
                 "companion Hall bar",
             source="arXiv:2406.13740 does exactly this on the same chip",
             risk="low, and it is what makes the resonator data interpretable"),
    ]

    res["controls"] = [
        "a bare Nb/Al reference resonator on the same chip, expected Q_i ~ 1e6 "
        "(arXiv:2511.08466 measured 5.985e6) -- proves the base layer and the chain.",
        "a control resonator with the same contacts and capacitor but the strip "
        "replaced by a shorting Nb bar -- isolates the contact and capacitor loss "
        "independently of the 1/N fit.",
        "THE SAME STRIP GATED OUT OF THE SUPERCONDUCTING DOME (to the correlated "
        "insulator at nu = +2 and to charge neutrality) -- the strip is its own control, "
        "and this is the one control the moire platform can do that no thin film can.",
        "the companion Hall bar, giving T_c, nu, R_sheet, R_contact and the twist angle "
        "from the Landau fan, on the same flake, in the same cooldown.",
    ]

    res["observables"] = [
        dict(observable="Q_i versus photon number, n = 0.1 to 1e6",
             diagnostic="a TLS-saturation curve, Q_i rising as roughly sqrt(1 + (n/nc)) "
                        "toward a high-power plateau, means the loss is TWO-LEVEL "
                        "SYSTEMS -- an interface/contamination problem with a known "
                        "fabrication cure.  A FLAT Q_i(n) means the loss is resistive: "
                        "contacts, normal regions from twist disorder, or "
                        "quasiparticles.",
             ref=LIT["fit_methods"]["loss_review"]),
        dict(observable="Q_i versus temperature, 20 mK to 1.2 K",
             diagnostic="a peak near hf/k_B = 180-360 mK is the TLS fingerprint and was "
                        "observed at 0.2 K in exactly this device class "
                        "(arXiv:2407.09916).  A monotonic fall tracking exp(-Delta/kT) "
                        "is quasiparticles.  A rise through T_c ~ 1 K identifies which "
                        "loss is superconducting-state loss at all.",
             ref="arXiv:2407.09916; arXiv:2512.08059 Fig. 4 inset"),
        dict(observable="f_r and Q_i versus filling factor nu",
             diagnostic="the superfluid stiffness D_s(nu) comes free with the frequency, "
                        "and the loss can be mapped across the dome.  If Q_i is high at "
                        "the dome centre and collapses at the edges, the loss is "
                        "percolation between superconducting islands -- which is the "
                        "signature that log-correlated twist disorder "
                        "(arXiv:2606.17191) predicts.",
             ref="arXiv:2406.13740 for D_s(nu); arXiv:2606.17191 for the percolation "
                 "prediction"),
        dict(observable="D_s(T) power law at n = 1",
             diagnostic="BONUS, and it addresses M22 section E.2.  Tanaka et al. extract "
                        "power-law exponents indicating an anisotropic gap.  Doing it at "
                        "p ~ 1 and at the single-photon level removes the microwave-"
                        "current suppression they had to correct for.  CAVEAT: "
                        "arXiv:2606.17191 shows log-correlated twist disorder mimics "
                        "nodal behaviour, so this diagnostic is compromised and must be "
                        "reported with the Landau-fan twist-angle map beside it.",
             ref="arXiv:2406.13740; arXiv:2606.17191"),
        dict(observable="the Kerr shift and the bifurcation power",
             diagnostic="calibrates the depairing current in situ, and the Kerr "
                        "coefficient is the moire analogue of the -0.008 to -14.7 "
                        "Hz/photon measured for NbSe2.  A moire strip should be far more "
                        "nonlinear.  SPECULATION on the magnitude.",
             ref="arXiv:2511.08466"),
    ]

    res["kill_criteria"] = [
        dict(id="K0", when="before cooldown 2",
             test="DC 4-probe contact resistance on the companion Hall bar",
             kill="rho_c > 1000 Ohm.um, i.e. R_c(total) > 200 Ohm on 10 um pads -> DO NOT COOL.  Refabricate with wider "
                  "contact pads, or switch to the differential node-contact variant.  "
                  "At the published 3.2 kOhm the ladder cannot resolve above Q ~ 1e2 "
                  "and the cooldown is wasted."),
        dict(id="K1", when="first analysis",
             test="the gated-out control: Q_i with the strip driven insulating",
             kill="if Q_i is UNCHANGED, the resonator is not sensitive to the moire "
                  "element at all and the whole measurement is void.  This is the "
                  "Kreidel failure mode (arXiv:2407.09916), stated in their own words, "
                  "and it is the reason section 2 exists."),
        dict(id="K2", when="first analysis",
             test="the 1/N fit intercept versus its 3-sigma uncertainty",
             kill="if the intercept is consistent with zero because the contact term "
                  "dominates by more than ~100x, the experiment has only bounded the "
                  "contacts.  Report the bound, refabricate, do not proceed to design."),
        dict(id="K3", when="the answer", test="intercept tan_delta at n = 1",
             kill="tan_delta > 3.1e-5 (Q < 3.2e4), reproduced on at least two flakes, "
                  "with the gated-out control confirming the loss tracks the "
                  "superconducting state -> ROUTE B IS DEAD.  Stop the M28 "
                  "superinductance programme.  The moire element cannot carry "
                  "participation of order one."),
        dict(id="K4", when="the answer", test="intercept tan_delta at n = 1",
             kill="tan_delta <= 3.1e-6 (Q >= 3.2e5) -> ROUTE B IS UNBLOCKED.  Build the "
                  "M28 fluxonium.  This is the only outcome that justifies further "
                  "moire circuit design."),
        dict(id="K5", when="diagnosis", test="Q_i(n) and Q_i(T)",
             kill="flat in power AND flat in temperature below T_c -> the loss is "
                  "neither TLS nor quasiparticles.  It is contacts or normal regions "
                  "from twist disorder.  The cure is fabrication, not material, and the "
                  "programme should say so rather than reporting a material number."),
    ]
    return res


# ============================================================ section 8
def section_vs_m15():
    """Cost and schedule against the transmon of out/m15_first_device.json."""
    res = {}
    m30 = dict(
        device="M30 loss ladder: 10 lumped resonators + 1 companion Hall bar on 1 flake",
        process_modules=["base superconductor layer", "back gate", "magic-angle stack",
                         "strip patterning", "edge contacts", "shunt/coupling caps"],
        n_process_modules=6,
        junction_module=False,
        rf_lines=2, dc_lines=4, fast_lines=0,
        instruments=["VNA", "DC source", "lock-in"],
        parametric_amplifier="optional",
        single_shot_readout=False,
        purcell_filter=False,
        acquisition_days=1.0,
        cooldowns_planned=2,
        yields_a_number_even_if="the device is not a qubit, not in the transmon regime, "
                                "and has the wrong I_c R_n -- because a resonator has no "
                                "E_J/E_C window to hit",
    )
    m15 = dict(
        device="M15 first device: 6-cell gatemon ladder (out/m15_first_device.json)",
        process_modules=["base superconductor layer", "back gate", "magic-angle stack",
                         "junction patterning", "edge contacts", "Al/AlOx junction + "
                         "bandage", "44.6 fF island + 310 um routed line",
                         "readout resonator (M16)", "Purcell filter (M18)"],
        n_process_modules=9,
        junction_module=True,
        rf_lines=3, dc_lines=2, fast_lines=1,
        instruments=["VNA", "AWG", "IQ mixers + LO chain", "digitiser", "DC source"],
        parametric_amplifier="mandatory (JTWPA 3-11 GHz, single-shot readout)",
        single_shot_readout=True,
        purcell_filter=True,
        acquisition_days=14.0,
        cooldowns_planned=3,
        yields_a_number_only_if="I_c R_n lands in 7-60 uV AND the cell is 'live' AND "
                                "'readable' -- M15's own coverage table shows only 2-3 "
                                "of 6 cells are usable at any given anchor",
    )
    res["m30"] = m30
    res["m15"] = m15

    res["ratios"] = dict(
        process_modules=round(m15["n_process_modules"] / m30["n_process_modules"], 2),
        instrument_channels=round((m15["rf_lines"] + m15["fast_lines"]) /
                                  (m30["rf_lines"] + m30["fast_lines"]), 2),
        acquisition_days=round(m15["acquisition_days"] / m30["acquisition_days"], 2),
        note="ESTIMATE.  These are counts of modules, lines and days, not currency.  "
             "The programme has never costed a device in currency and this module will "
             "not start.",
    )

    res["the_subset_argument"] = (
        "M30's fabrication is a STRICT SUBSET of M15's.  Both need the same base layer, "
        "the same back gate, the same magic-angle stack, the same edge contacts.  M15 "
        "additionally needs an Al/AlOx junction module, a 44.6 fF island, a readout "
        "resonator, a Purcell filter, and a fast control chain.  Building M30 first "
        "therefore costs NOTHING that M15 would not have paid anyway -- the stack is the "
        "schedule, and M30 is the pilot run for it.")

    res["the_attribution_argument"] = (
        "M15's own list of what device one settles (out/m15_first_device.json:"
        "what_device_one_settles) includes T1 by inversion recovery, flagged 'NEVER "
        "MEASURED IN TBG BY ANYONE'.  But a transmon T1 is a SUM: junction dielectric "
        "loss, moire element loss, contact loss, quasiparticles, Purcell, and the "
        "package, and the design has no way to separate them.  Balgley et al. "
        "(arXiv:2512.08059) needed a SEPARATE resonator measurement of the same "
        "material to attribute their qubit's T1 to dielectric loss -- exactly M30 -- and "
        "they found qubit-derived and resonator-derived loss tangents agreeing to within "
        "a factor 2.  If M15 runs first and returns T1 = 700 ns, nobody can say whether "
        "the moire is at fault.  If M30 runs first, M15's T1 is predicted before it is "
        "measured.")

    res["the_counter_argument"] = (
        "Against building M30 first: out/m23_majorana.json argues explicitly that a "
        "topological qubit's lifetime follows from the gap and the parity-poisoning "
        "rate, both computable, so 'the topology can be designed now and the number "
        "measured afterwards'.  That is correct, and M30 does not gate the topological "
        "programme.  The rebuttal is scope, not physics: M30 gates route B entirely, "
        "gates every moire resonator and amplifier, and makes M15 interpretable -- three "
        "things for the price of a subset of one device's fabrication.")

    res["recommendation"] = (
        "BUILD M30 FIRST.  Four reasons, in order of strength: (1) M30's fabrication is "
        "a strict subset of M15's, so it is not a detour; (2) M30 returns a number "
        "unconditionally, while M15 returns one only if I_c R_n lands in a window it "
        "cannot control; (3) without M30, M15's T1 cannot be attributed to anything; "
        "(4) a null result at K3 cancels M28 and every downstream moire circuit module, "
        "which is the largest single piece of design work the programme could avoid.  "
        "The honest cost of being wrong about this ordering is one flake and two weeks.")
    return res


# ================================================================== main
def main():
    print("=" * 78)
    print("M30 -- ROUTE D: THE GATING MEASUREMENT")
    print("single-photon microwave loss of a magic-angle moire circuit element")
    print("=" * 78)

    res = {"_meta": {
        "module": "M30 -- design of the single-photon microwave loss measurement of a "
                  "MATBG circuit element",
        "script": "scripts/run_m30_loss.py",
        "deterministic": True,
        "rng": None,
        "inputs_read": {
            "paper/M22_TOPOLOGY_SURVEY.md": "section E.7 (the survey's 6.3): no "
                                            "single-photon microwave loss measurement "
                                            "exists for any MATBG or twisted-trilayer "
                                            "circuit element",
            "out/m23_majorana.json": "Delta_topo = 0.258 meV at Tc = 1.7 K, xi_M = 49.5 "
                                     "nm, and the argument that a Majorana qubit's "
                                     "lifetime is computable rather than measured",
            "out/m25_tetron.json": "tetron footprint 1.14 um2, E_C = 80.6 GHz, and the "
                                   "still-missing parity measurement circuit",
            "out/m28_superinductance.json": "Q_ind >= 3.22e6 for T1 = 100 us; the 5.87 "
                                            "um x 400 nm strip; the Q >= 10-50 bound",
            "out/m15_first_device.json": "the 6-cell transmon ladder this module is "
                                         "compared against",
            "out/m13_readout_chain.json": "the amplifier catalogue",
        },
        "claim_discipline": "ANALYTIC DESIGN AND ARITHMETIC.  NOT a measurement, NOT an "
                            "EM solve, NOT a fabrication trial.  Every free parameter is "
                            "registered in section 9 with a two-directional sensitivity. "
                            "Speculation is tagged SPECULATION.",
    }}
    res["literature"] = LIT

    print("\n[2] PARTICIPATION -- can the measurement see the element at all?")
    part = section_participation()
    res["participation"] = part
    print(f"    published moire geometry: p = "
          f"{part['published_moire_geometry']['p_from_2dfof']}, "
          f"Q_MATBG >= {part['published_moire_geometry']['Q_MATBG_lower_bound']}")
    print(f"    published vdW ladder best: p = "
          f"{max(r['p_exact'] for r in part['published_vdw_ladder']['rows'])}, "
          f"tan_delta <= {part['published_vdw_ladder']['best_tan_delta_upper']:.2e}")

    print("\n[3] THE DEVICE -- five rungs, one flake")
    des, ladder = section_design()
    res["design"] = des
    for r in des["ladder"]:
        print(f"    N={r['N_squares']:6.1f} sq  L={r['L_nH']:8.2f} nH  "
              f"f={r['f_r_GHz']:5.2f} GHz  C={r['C_required_fF']:6.3f} fF  "
              f"Z={r['Z_kohm']:7.2f} kOhm  p={r['participation']:.5f}  "
              f"stub={r['equivalent_stub_length_um']:5.2f} um")

    print("\n[4] CONTACTS -- the thing that decides whether this works")
    con = section_contacts(ladder)
    res["contacts"] = con
    for k, v in con["measurement_floor_vs_contact"].items():
        print(f"    {k:24s}  3-sigma floor tan_delta = "
              f"{v['three_sigma_tan_delta_floor']:.2e}  -> Q ceiling "
              f"{v['equivalent_Q_ceiling']:.3g}")

    print("\n[5] THRESHOLDS")
    th = section_thresholds()
    res["thresholds"] = th
    for b in th["verdict_bands"]:
        print(f"    {b['band']:9s} Q_i {b['Q_i']:16s} tan_delta {b['tan_delta']}")

    print("\n[6] THE CHAIN")
    ch = section_chain(ladder)
    res["chain"] = ch
    print(f"    total line attenuation {ch['drive_power']['total_line_attenuation_dB']} dB")
    print(f"    residual thermal photons at the sample "
          f"{ch['thermal_photons']['residual_nbar_at_sample']:.2e}")
    print(f"    worst-case MXC dissipation "
          f"{ch['heat_load']['dissipated_at_MXC_nW']} nW "
          f"(margin {ch['heat_load']['margin']}x)")
    print(f"    total acquisition {ch['campaign']['total_acquisition_hours']} hours")

    print("\n[7] FABRICATION, CONTROLS, KILLS")
    fk = section_fab_and_kills()
    res["experiment"] = fk
    for k in fk["kill_criteria"]:
        print(f"    {k['id']}: {k['test']}")

    print("\n[8] AGAINST M15")
    cmp_ = section_vs_m15()
    res["comparison_with_m15"] = cmp_
    print(f"    process modules {cmp_['m30']['n_process_modules']} vs "
          f"{cmp_['m15']['n_process_modules']}; acquisition "
          f"{cmp_['m30']['acquisition_days']} d vs {cmp_['m15']['acquisition_days']} d")
    print(f"    -> {cmp_['recommendation'][:60]}...")

    # ---------------------------------------------------------- synthesis
    g200 = con["measurement_floor_vs_contact"]["rho_c_1000_ohm_um"]["equivalent_Q_ceiling"]
    g20 = con["measurement_floor_vs_contact"]["rho_c_100_ohm_um"]["equivalent_Q_ceiling"]
    h20 = con["measurement_floor_at_150nH_per_sq"]["rows"]["rho_c_100_ohm_um"]["equivalent_Q_ceiling"]
    node = next(r["Q_node_ceiling"] for r in con["node_contact_variant"]["rows"]
                if r["placement_error_frac"] == 0.01 and r["R_c_ohm"] == 20.0
                and r["Z_bias_ohm"] == 1.0e4)
    res["sensitivity_summary"] = dict(
        rows=[
            dict(variant="published moire geometry (arXiv:2406.13740)",
                 participation=0.02, resolves_up_to_Q=round(0.02 * 1044.7, 1),
                 decides="nothing -- it is a bound of Q >= 21"),
            dict(variant="galvanic ladder, L_sq = 29.3 nH/sq, rho_c = 1000 Ohm.um",
                 participation=1.0, resolves_up_to_Q=g200,
                 decides="DEAD vs not-DEAD"),
            dict(variant="galvanic ladder, L_sq = 29.3 nH/sq, rho_c = 100 Ohm.um",
                 participation=1.0, resolves_up_to_Q=g20,
                 decides="DEAD vs WEAK"),
            dict(variant="galvanic ladder, L_sq = 150 nH/sq, rho_c = 100 Ohm.um",
                 participation=1.0, resolves_up_to_Q=h20,
                 decides="DEAD vs WEAK vs MARGINAL"),
            dict(variant="differential node-contact, rho_c = 100 Ohm.um, Z_bias = 1e4 Ohm (a 320 nH on-chip choke), eps = 1 per cent",
                 participation=1.0, resolves_up_to_Q=node,
                 decides="everything up to the VIABLE boundary -- SPECULATIVE, never "
                         "built"),
        ],
        headline=(
            "participation is necessary but not sufficient.  Driving p from 0.02 to 1.0 "
            "buys 50x; the remaining ceiling is set by the CONTACTS, and beating it "
            "needs either 20 Ohm contacts and the top of the gate-tunable L_sq window "
            "(reaching Q ~ 5e5, enough to decide three of the four bands) or the "
            "differential node-contact variant (reaching Q ~ 3e6, enough to decide all "
            "four).  Both fit on one mask."),
    )

    res["headline"] = (
        "The measurement M22 section E.7 asks for cannot be done in the geometry the "
        "moire cQED literature uses.  For a sample terminating a Z0 = 50 Ohm line the "
        "exact participation is p = sin(2 phi)/(sin(2 phi)/2 + pi/2 - phi) with "
        "phi = arctan(omega L/Z0), so the perturbative regime that makes the kinetic-"
        "inductance extraction clean is the same regime in which p << 1 and the loss is "
        "invisible.  Shortening the line from a 5.8 mm quarter wave to a 4-46 um stub "
        "drives p to 0.9999, and reproduces M28's 2.4 fF lumped design from the "
        "distributed side.  With p ~ 1 the contacts become the ceiling: at the one "
        "published moire rf contact resistance (3.2 kOhm, arXiv:2406.13742) the whole "
        "ladder is pinned below Q ~ 22, so a length ladder is mandatory and the contact "
        "resistance is a pre-registered gate.  The experiment is one flake, six process "
        "modules all of which M15 needs anyway, two RF lines, no fast lines, an optional "
        "parametric amplifier, and about four hours of acquisition.")

    res["choices"] = CHOICES
    res["no_literature_found"] = [
        "No measurement of Q_i versus photon number for ANY moire circuit element.  "
        "Tanaka et al. (arXiv:2406.13740) report Q_i = 1044.7 +/- 28.5 and sweep "
        "microwave power on the RESONANCE FREQUENCY (Fig. 5d) and DC bias current on "
        "Q_i (Supp. Fig. 17), but never report Q_i versus microwave power.  M22 section "
        "E.7 stands, NARROWED: the statement is now specific to moire, because "
        "arXiv:2511.08466 (Nov 2025) did publish single-photon Q_i for twelve NbSe2 "
        "elements.",
        "No measurement of Q_i versus temperature for any moire circuit element.",
        "No measurement of the microwave loss of a moire galvanic contact, which "
        "arXiv:2406.13742 shows dominates the rf resistance at R_c = 3.2 kOhm.",
        "No published sheet kinetic inductance for magic-angle twisted BILAYER graphene "
        "at CPR precision.  The 10-150 nH/sq window this design uses is TRILAYER at "
        "1.71 deg (arXiv:2403.02320).",
        "No moire circuit element has ever been operated at a participation ratio above "
        "a few per cent.  The highest in the literature, computed here from "
        "arXiv:2511.08466's own numbers, is p = 0.53 and it is NbSe2, not moire.",
        "No measurement of the vortex-entry or phase-slip RATE of a moire kinetic "
        "inductor.",
        "No measurement of quasiparticle-parity lifetime in a moire circuit.",
        "No moire superconducting qubit of any kind.",
    ]

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "m30_loss.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote out/m30_loss.json")


if __name__ == "__main__":
    main()
