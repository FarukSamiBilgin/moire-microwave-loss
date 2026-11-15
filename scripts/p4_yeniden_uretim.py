"""
p4_yeniden_uretim.py -- independent re-derivation of every computed number in
P4 (moire microwave loss). Written WITHOUT reading out/m30_loss.json; the
comparison against the artifact is done afterwards by compare_json() (run with
--compare <path-to-m30_loss.json>).

Only CODATA constants and the paper's stated inputs are used. Every printed
line is  key = value  so the output can be diffed.
"""
import sys, json, math
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

h = 6.62607015e-34
hbar = h / (2 * math.pi)
kB = 1.380649e-23
e = 1.602176634e-19
Phi0 = h / (2 * e)
Z0 = 50.0
eps0 = 8.8541878128e-12

R = {}  # results


def put(k, v):
    R[k] = v
    print(f"{k} = {v!r}" if not isinstance(v, float) else f"{k} = {v:.6g}")


# ---------------------------------------------------------------------------
# (1) participation of a terminating inductance -- derivation check
# ---------------------------------------------------------------------------
def p_exact(x):
    """Closed form, Eq. (2); x = omega L / Z0."""
    ph = math.atan(x)
    s = math.sin(2 * ph)
    return s / (s / 2 + (math.pi / 2 - ph))


def p_numeric(x):
    """Brute force: integrate magnetic energy along the line numerically.
    I(z)=cos(beta z + phi), line occupies beta z in [0, pi/2 - phi].
    Energies in units of Z0*A^2/(4 omega):
      sample: (omega L / Z0) cos^2 phi
      line  : int_0^{pi/2-phi} cos^2(u + phi) du   (L' dz = Z0/omega d(beta z))"""
    ph = math.atan(x)
    Es = x * math.cos(ph) ** 2
    El, _ = quad(lambda u: math.cos(u + ph) ** 2, 0.0, math.pi / 2 - ph)
    return Es / (Es + El)


def p_pert(x):
    """2|d omega/omega| for a lambda/4 line whose end is loaded: omega/omega0 = 1-2phi/pi."""
    return 4 * math.atan(x) / math.pi


def stub_frac(x):
    return (math.pi / 2 - math.atan(x)) / (math.pi / 2)


def p_lumped(x):
    """L/(L + L'l) with L'l = Z0*(pi/2)/omega  -> x/(x+pi/2)."""
    return x / (x + math.pi / 2)


print("# (1) participation")
for x in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0]:
    put(f"T1.x={x}.p_exact", p_exact(x))
    put(f"T1.x={x}.p_numeric", p_numeric(x))
    put(f"T1.x={x}.p_pert", p_pert(x))
    put(f"T1.x={x}.stub_frac", stub_frac(x))
    put(f"T1.x={x}.p_lumped", p_lumped(x))
    put(f"T1.x={x}.pert_over_exact", p_pert(x) / p_exact(x))
# small-x limit 4x/pi
put("T1.smallx_ratio_p_over_4x_pi_at_1e-4", p_exact(1e-4) / (4e-4 / math.pi))
# crossings
put("T1.x_at_p=0.5", brentq(lambda x: p_exact(x) - 0.5, 1e-3, 1e3))
put("T1.x_at_p=0.99", brentq(lambda x: p_exact(x) - 0.99, 1e-3, 1e4))
put("T1.x_at_p=0.997", brentq(lambda x: p_exact(x) - 0.997, 1e-3, 1e4))
put("T1.x_where_pert_5pct_high", brentq(lambda x: p_pert(x) / p_exact(x) - 1.05, 1e-3, 10))

# ---------------------------------------------------------------------------
# (2) Tanaka bounds (inputs: published f-shift 1 %, Qi, Al control Qi)
# ---------------------------------------------------------------------------
print("# (2) Tanaka")
frac = 0.01
Qi_T, Qi_T_err = 1044.7, 28.5
Qi_ctrl = 74833.0
p_T = 2 * frac
put("T2.p_2dfof", p_T)
# exact p for the same shift: 4phi/pi = 0.02 -> phi
phi_T = p_T * math.pi / 4
put("T2.p_exact_same_shift", p_exact(math.tan(phi_T)))
Q_dir = p_T * Qi_T
put("T2.Q_MATBG_lower_direct", Q_dir)
Q_exc = p_T / (1 / Qi_T - 1 / Qi_ctrl)
put("T2.Q_MATBG_lower_excess", Q_exc)
put("T2.tan_delta_upper_direct", 1 / Q_dir)
put("T2.tan_delta_upper_excess", 1 / Q_exc)
put("T2.Qi_rel_err", Qi_T_err / Qi_T)

# ---------------------------------------------------------------------------
# (3)+(7) design ladder, stub
# ---------------------------------------------------------------------------
print("# (3)/(7) ladder")
Lsq = 29.3e-9
xi = 0.2e-6
dphidl = 2 / (3 * math.sqrt(3) * xi)  # rad/m
put("D.dphi_dl_rad_per_um", dphidl * 1e-6)
Kc = 200e-9 / 1e-6  # A/m
ladder = [(2, 7.5e9), (5, 6.5e9), (14.7, 5.5e9), (40, 4.5e9), (100, 3.8e9)]
eps_eff = (11.7 + 1) / 2  # assumption for stub length; reported separately
v = 299792458 / math.sqrt(eps_eff)
LAD = []
for N, f in ladder:
    w_ = 2 * math.pi * f
    L = Lsq * N
    C = 1 / (w_ ** 2 * L)
    Z = w_ * L
    x = Z / Z0
    phz = (2 * math.pi / Phi0) * math.sqrt(hbar * w_ * L / 2)
    wid = max(2 * xi, 3 * phz / (N * dphidl))
    length = N * wid
    area = N * wid ** 2
    Izpf = math.sqrt(hbar * w_ / (2 * L))
    Idep = Kc * wid
    avail = N * wid * dphidl
    d = dict(N=N, f=f, L_nH=L * 1e9, C_fF=C * 1e15, Z_kohm=Z / 1e3, wL_Z0=x,
             p=p_exact(x), one_minus_p=1 - p_exact(x), hf_kB_mK=h * f / kB * 1e3,
             phi_zpf=phz, width_nm=wid * 1e9, length_um=length * 1e6,
             area_um2=area * 1e12, I_zpf_nA=Izpf * 1e9, I_dep_nA=Idep * 1e9,
             ratio=Izpf / Idep, avail=avail, headroom=avail / phz,
             C_per_area_fF_um2=C * 1e15 / (area * 1e12),
             E_L_GHz=(Phi0 / (2 * math.pi)) ** 2 / L / h / 1e9)
    LAD.append(d)
    for k, val in d.items():
        put(f"D.N={N}.{k}", val)
# stub for N=14.7 element
d = LAD[2]
w_ = 2 * math.pi * d["f"]
x = d["wL_Z0"]
bl = math.pi / 2 - math.atan(x)
put("S.betal_rad", bl)
put("S.eps_eff_assumed", eps_eff)
put("S.stub_len_um_eps6.35", bl / (w_ / v) * 1e6)
put("S.quarterwave_mm_5.5GHz_eps6.35", (v / d["f"]) / 4 * 1e3)
put("S.quarterwave_mm_4.2GHz_eps6.35", (v / 4.2008154e9) / 4 * 1e3)
put("S.stub_distC_fF", (bl / (w_ / v)) / (Z0 * v) * 1e15)
# what eps_eff gives lambda/4 = 5.8 mm at 5.5 GHz?
put("S.eps_eff_for_5.8mm_at_5.5GHz", (299792458 / (4 * 5.8e-3 * 5.5e9)) ** 2)
put("S.eps_eff_for_5.8mm_at_4.2GHz", (299792458 / (4 * 5.8e-3 * 4.2008154e9)) ** 2)
put("S.stub_len_um_if_lambda4=5.8mm", 5.8e3 * bl / (math.pi / 2))

# ---------------------------------------------------------------------------
# (4) contact ceiling
# ---------------------------------------------------------------------------
print("# (4) contacts")
Wpad = 10.0  # um
for rho in [100, 1000, 16000]:
    Rc = 2 * rho / Wpad
    Qs = [LAD[i]["Z_kohm"] * 1e3 / Rc for i in range(5)]
    put(f"C.rho={rho}.Rc_total", Rc)
    put(f"C.rho={rho}.Q_min", min(Qs))
    put(f"C.rho={rho}.Q_max", max(Qs))
# conversion of Banerjee 3.2 kOhm to rho_c under different width readings
for W in [2.0, 10.0]:
    put(f"C.rho_from_3.2k_total_two_contacts_W={W}", 3200 * W / 2)
    put(f"C.rho_from_3.2k_per_contact_W={W}", 3200 * W)
put("C.pad_gain_vs_400nm", Wpad / 0.4)
put("C.pad_gain_vs_532nm", Wpad / 0.5321)
put("C.pad_gain_vs_904nm", Wpad / 0.9037)
Zmax = 1 / (2 * math.pi * 2e9 * 0.5e-15)
put("C.Zmax_kohm_0.5fF_2GHz", Zmax / 1e3)
put("C.Qceil_rho100", Zmax / 20)
put("C.Qceil_rho16000", Zmax / 3200)

# ---------------------------------------------------------------------------
# (5) 1/N regression, weighted, 1 % per point
# ---------------------------------------------------------------------------
print("# (5) regression")
Zs = np.array([dd["Z_kohm"] * 1e3 for dd in LAD])
xs = 1 / Zs  # regressor 1/(omega Lsq N) = 1/Z
put("R.regressor_span_xmax_over_xmin", float(xs.max() / xs.min()))
put("R.N_span", 100 / 2)


def wls(xv, yv, sig):
    W = np.diag(1 / sig ** 2)
    X = np.vstack([np.ones_like(xv), xv]).T
    cov = np.linalg.inv(X.T @ W @ X)
    return np.sqrt(np.diag(cov))


for rho in [100, 1000, 16000]:
    Rc = 2 * rho / Wpad
    for tand in [0.0, 1e-6, 1e-5, 1e-4]:
        y = tand + Rc * xs
        sa, sb = wls(xs, y, 0.01 * y)
        put(f"R.rho={rho}.tand={tand}.3sig_intercept", 3 * sa)
        put(f"R.rho={rho}.tand={tand}.sigma_rho", sb * Wpad / 2)
        put(f"R.rho={rho}.tand={tand}.Q_equiv_1_over_3sig", 1 / (3 * sa))
# at 150 nH/sq
scale = 150 / 29.3
put("R.Lsq_gain_150", scale)
y = 20 * xs / scale
sa, sb = wls(xs / scale, y, 0.01 * y)
put("R.150nH.rho100.Q_equiv", 1 / (3 * sa))
# degeneracy: C_gate proportional to area
for cg in [1.0, 2.0, 3.0]:  # fF/um^2 illustrative
    for dd in LAD:
        put(f"G.cg={cg}.N={dd['N']}.Cgate_fF", cg * dd["area_um2"])
# hBN thickness that gives c_g (eps_perp = 3.76)
for t in [10e-9, 20e-9, 30e-9, 50e-9]:
    put(f"G.cg_fF_per_um2_hBN_t={t*1e9:.0f}nm", eps0 * 3.76 / t * 1e15 / 1e12)
# Z independent of N when C = cg*N*w^2, L = Lsq*N (fixed w=400 nm)
for N in [14.7, 40, 100]:
    Cg = 2.0e-15 / 1e-12 * N * (0.4e-6) ** 2  # 2 fF/um^2
    put(f"G.Z_kohm_fixedw_cg2.N={N}", math.sqrt(Lsq * N / Cg) / 1e3)

# ---------------------------------------------------------------------------
# (6) hBN
# ---------------------------------------------------------------------------
print("# (6) hBN")
tan_hbn = 5e-6
put("H.Q_if_tand_equals_bound", 1 / tan_hbn)
p_e_hbn = 3.0 / 4.092
put("H.electric_participation_gate_N5", p_e_hbn)
put("H.Q_hbn_limited_N5_at_bound", 1 / (p_e_hbn * tan_hbn))

# ---------------------------------------------------------------------------
# (8) fluxonium requirement
# ---------------------------------------------------------------------------
print("# (8) fluxonium")
EL_h = 0.38e9
phi01 = 2.5959
pref = 2 * (2 * math.pi * EL_h) * phi01 ** 2  # Gamma*Q
put("F.Gamma_times_Q", pref)
put("F.f_eff_GHz", pref / (2 * math.pi) / 1e9)
for T1 in [1e-6, 10e-6, 100e-6, 1e-3]:
    Q = pref * T1
    put(f"F.T1={T1}.Q_ind", Q)
    put(f"F.T1={T1}.tand", 1 / Q)
put("F.L_for_EL0.38_nH", (Phi0 / (2 * math.pi)) ** 2 / (h * EL_h) * 1e9)
for fq in [0.2e9, 0.5e9, 1.0e9]:
    for T in [0.02, 0.04]:
        put(f"F.coth_fq={fq/1e9}GHz_T={T*1e3:.0f}mK",
            1 / math.tanh(h * fq / (2 * kB * T)))

# ---------------------------------------------------------------------------
# (9) bands
# ---------------------------------------------------------------------------
print("# (9) bands")
for t in [3.1e-5, 3.108e-5, 3.1e-6, 3.108e-6, 3.1e-7]:
    put(f"B.Q_for_tand={t}", 1 / t)
put("B.K3_K4_gap_decades", math.log10(3.1e-5 / 3.1e-6))
put("B.balgley_over_viable", 1.7e-5 / (1 / 3.218e6))
put("B.balgley1.69_over_viable", 1.69e-5 / (1 / 3.218e6))

# ---------------------------------------------------------------------------
# (10) dilution escape
# ---------------------------------------------------------------------------
print("# (10) dilution")
for f in [5.0e9, 5.5e9, 6.0e9, 6.5e9]:
    put(f"E.f={f/1e9}.T1_us_p0.01", 1 / (2 * math.pi * f * 0.01 * 3.1e-5) * 1e6)
    put(f"E.f={f/1e9}.T1_us_p1", 1 / (2 * math.pi * f * 1.0 * 3.1e-5) * 1e6)
put("E.f_implied_by_85.57us_GHz", 1 / (2 * math.pi * 85.57e-6 * 0.01 * 3.1e-5) / 1e9)

# ---------------------------------------------------------------------------
# extras: Zaman rows, drive power
# ---------------------------------------------------------------------------
print("# extras")
for name, Lk, fs, Qi in [("D1", 1.195e-9, 3.62e9, 8.0e3), ("D2", 0.643e-9, 4.14e9, 6.0e4),
                         ("D12", 0.008e-9, 5.60e9, 5.8e4)]:
    x = 2 * math.pi * fs * Lk / Z0
    p = p_exact(x)
    put(f"Z.{name}.p", p)
    put(f"Z.{name}.tand_upper_tableQi", 1 / (p * Qi))
    put(f"Z.{name}.Q_lower_tableQi", p * Qi)
# drive power n=1, Qi=Qc=1e4, 5.5 GHz
w_ = 2 * math.pi * 5.5e9
for Qi in [1e3, 1e4, 1e5, 1e6]:
    kc = w_ / Qi
    ki = w_ / Qi
    P = hbar * w_ * (kc + ki) ** 2 / (2 * kc)
    put(f"P.Qi={Qi:.0e}.P_dBm_n1", 10 * math.log10(P / 1e-3))


# ---------------------------------------------------------------------------
# second pass (added after reading the artifact AND Tanaka et al.; the inputs
# below are from Tanaka arXiv:2406.13740 text: L_prx = 1.32 nH in parallel,
# f0(insulating) = 3.8 GHz, fitted f_r = 4.2008154 GHz, D_s max 1.7e8 1/H)
# ---------------------------------------------------------------------------
print("# second pass")
# (a) sensitivity of the Tanaka bound to which 'shift' is used
for lab, sh in [("1pct_text", 0.01), ("fit_vs_nominal", (4.2008154 - 3.8) / 4.2008154)]:
    pp = 2 * sh
    put(f"X.tanaka.{lab}.p_pert", pp)
    put(f"X.tanaka.{lab}.Q_lower_direct", pp * Qi_T)
    put(f"X.tanaka.{lab}.tand_upper", 1 / (pp * Qi_T))
# (b) parallel-termination model: L_prx || L_TBG terminating a 50 Ohm line
Lprx = 1.32e-9
for f in [3.8e9, 4.2e9]:
    x = 2 * math.pi * f * Lprx / Z0
    put(f"X.parallel.f={f/1e9}.x_Lprx", x)
    put(f"X.parallel.f={f/1e9}.p_termination", p_exact(x))
    # L_TBG giving p_TBG = 0.02 (energy split of a parallel pair ~ 1/L)
    ptot = p_exact(x)
    frac = 0.02 / ptot
    LTBG = Lprx * (1 - frac) / frac
    put(f"X.parallel.f={f/1e9}.L_TBG_for_pTBG0.02_nH", LTBG * 1e9)
    put(f"X.parallel.f={f/1e9}.squares_l_over_w_at_Ds1.7e8", LTBG * 1.7e8)
# naive: if L_TBG = 1/D_s (one square) were the SOLE termination
for Ds in [1.7e8, 1.25e8]:
    x = 2 * math.pi * 4.2e9 * (1 / Ds) / Z0
    put(f"X.naive_sole_termination.Ds={Ds:.3g}.x", x)
    put(f"X.naive_sole_termination.Ds={Ds:.3g}.p", p_exact(x))
# (c) Zaman excess-loss method: tand <= [1/Qi - (1-p)/Q_spec]/p
Qspec = 5.985e6
for name, Lk, fs, Qi in [("D1", 1.195e-9, 3.62e9, 8.0e3), ("D2", 0.643e-9, 4.14e9, 6.0e4),
                         ("D12", 0.008e-9, 5.60e9, 5.8e4)]:
    p = p_exact(2 * math.pi * fs * Lk / Z0)
    t = (1 / Qi - (1 - p) / Qspec) / p
    put(f"X.zaman.{name}.tand_excess_1mp", t)
    put(f"X.zaman.{name}.Q_excess_1mp", 1 / t)
# Tanaka with the same (1-p) convention
put("X.tanaka.Q_excess_1mp", 1 / ((1 / Qi_T - (1 - 0.02) / Qi_ctrl) / 0.02))
# (d) sigma_rho_c at tan_delta_true = 3.1e-5 (artifact's intercept_fit index 4/12/20)
for rho in [100, 1000, 16000]:
    Rc = 2 * rho / Wpad
    y = 3.1e-5 + Rc * xs
    sa, sb = wls(xs, y, 0.01 * y)
    put(f"X.reg.rho={rho}.tand3.1e-5.sigma_rho", sb * Wpad / 2)
    put(f"X.reg.rho={rho}.tand3.1e-5.3sig", 3 * sa)
# (e) contact row 3 alternatives at 10 um pads
for lab, rho in [("2um_total", 3200.0), ("2um_per_contact", 6400.0), ("as_printed", 16000.0)]:
    Rc = 2 * rho / Wpad
    put(f"X.contact.{lab}.Rc_10um", Rc)
    put(f"X.contact.{lab}.Q_range", [round(LAD[0]['Z_kohm'] * 1e3 / Rc, 2),
                                     round(LAD[4]['Z_kohm'] * 1e3 / Rc, 1)])
# (f) fluxonium with thermal factor; f01 unknown -> illustrative
for fq in [0.3e9, 0.6e9, 1.0e9]:
    for T in [0.02, 0.04]:
        c = 1 / math.tanh(h * fq / (2 * kB * T))
        put(f"X.flux.Q100us_with_coth.fq={fq/1e9}.T={T*1e3:.0f}mK", 3.21788e6 * c)
# (g) hBN: participation of the 3 fF gate in N=5 total 4.092 fF
put("X.hbn.Q_ceiling_if_tand_at_bound_pe0.733", 1 / (0.733 * 5e-6))


# ---------------------------------------------------------------------------
def compare_json(path):
    with open(path, encoding="utf-8") as fh:
        J = json.load(fh)
    print("# --- artifact comparison ---")

    def g(*keys):
        o = J
        for k in keys:
            o = o[k]
        return o

    out = {}
    try:
        for i, row in enumerate(g("participation", "participation_curve")):
            out[f"J.curve[{i}]"] = row
        out["J.pub_moire"] = g("participation", "published_moire_geometry")
        out["J.ladder"] = g("design", "ladder")
        out["J.contact_ceiling"] = g("contacts", "contact_ceiling")
        out["J.mfvc"] = g("contacts", "measurement_floor_vs_contact")
        out["J.fluxonium"] = g("thresholds", "m28_fluxonium_requirement")
        out["J.dilution"] = g("thresholds", "dilution_escape")
        out["J.bands"] = g("thresholds", "verdict_bands")
    except Exception as ex:  # noqa
        print("missing key", ex)
    for k, v in out.items():
        print(k, "=", json.dumps(v, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    out_json = __file__.replace(".py", "_sonuc.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        compare_json(sys.argv[2])
