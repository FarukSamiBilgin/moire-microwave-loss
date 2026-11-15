"""Figures for the P4 author-voice preprint.

Fig. 1: participation of a terminating inductance, Eq. (2), exact vs
perturbative, with the resonant line length as a fraction of lambda/4.
Fig. 2: the quality-factor scale: four verdict bands, the published bound
from Tanaka et al., the contact ceilings of Table II and the hBN cap.
Every plotted number is either a closed form evaluated here or a number
quoted in main.tex (published, or recomputed by p4_yeniden_uretim.py).
Run:  python -I fig_uret.py   (writes ../figures/*.pdf and *.png)

Repository note: identical to the Zenodo-record version except that the
output folder is ../figures and a PNG copy of each figure is also written
(for display on GitHub).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
    "mathtext.fontset": "cm", "axes.linewidth": 0.6,
})


def p_exact(x):
    phi = np.arctan(x)
    return np.sin(2 * phi) / (np.sin(2 * phi) / 2 + (np.pi / 2 - phi))


def stub_frac(x):
    return (np.pi / 2 - np.arctan(x)) / (np.pi / 2)


# ---------------- Fig. 1 ----------------
x = np.logspace(-3, 3, 600)
fig, ax = plt.subplots(figsize=(3.37, 2.35))
ax.plot(x, p_exact(x), color="#1f4e79", lw=1.5, label=r"$p$, exact, Eq. (2)")
# perturbative 2|dw/w| = 2*(1 - stub_frac) for the terminated open line
ax.plot(x, 2 * (1 - stub_frac(x)), color="#c55a11", lw=1.1, ls="--",
        label=r"$2|\delta\omega/\omega|$, perturbative")
ax.plot(x, stub_frac(x), color="0.45", lw=1.0, ls=":",
        label=r"line length / $(\lambda/4)$")
ax.axhline(1.0, color="0.75", lw=0.5)
ax.axhline(0.02, color="#7f7f7f", lw=0.6, ls="-.")
ax.text(1.3e-3, 0.07, r"$p = 0.02$: published MATBG device",
        fontsize=6.2, color="0.25")
for xv, lab in [(119.7, r"$N=5$"), (297.7, r"$N=14.7$")]:
    ax.plot([xv], [p_exact(xv)], "o", ms=3.2, color="#1f4e79")
ax.annotate("lumped designs\n(Table II)", xy=(119.7, 0.995), xytext=(6, 0.55),
            fontsize=6.2, arrowprops=dict(arrowstyle="->", lw=0.5))
ax.set_xscale("log")
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(0, 2.05)
ax.set_xlabel(r"$\omega L / Z_0$")
ax.set_ylabel("participation / length fraction")
ax.legend(loc="upper left", frameon=False, handlelength=2.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig1_participation.pdf"))
fig.savefig(os.path.join(OUT, "fig1_participation.png"), dpi=220)
plt.close(fig)

# ---------------- Fig. 2 ----------------
fig, ax = plt.subplots(figsize=(3.37, 2.55))
bands = [(0.5, 3.2e4, "#f2dcdb", "dead"), (3.2e4, 3.2e5, "#fbe5d6", "weak"),
         (3.2e5, 3.2e6, "#fff2cc", "marginal"), (3.2e6, 3e7, "#e2efda", "viable")]
for lo, hi, c, lab in bands:
    ax.axvspan(lo, hi, color=c, lw=0)
    ax.text(np.sqrt(lo * hi) if hi < 3e7 else 1e7, 6.55, lab, ha="center",
            va="center", fontsize=6.5, color="0.2")
rows = [
    # (label, lo, hi, kind)
    (r"published MATBG bound, $Q\geq21$", 21, 21, "bound"),
    (r"NbSe$_2$ bound (D2), $Q\geq2.3\times10^4$", 2.33e4, 2.33e4, "bound"),
    (r"contacts, $\rho_c=100\,\Omega\,\mu$m", 138.1, 3497.8, "range"),
    (r"contacts, $\rho_c=1000\,\Omega\,\mu$m", 13.8, 349.8, "range"),
    (r"contacts, published 3.2 k$\Omega$", 0.9, 21.9, "range"),
    (r"hBN gate at its bound ($N=5$)", 2.73e5, 2.73e5, "cap"),
]
for i, (lab, lo, hi, kind) in enumerate(rows):
    y = 5.6 - i
    if kind == "range":
        ax.plot([lo, hi], [y, y], color="#1f4e79", lw=3.0, solid_capstyle="butt")
    elif kind == "bound":
        ax.annotate("", xy=(lo * 6, y), xytext=(lo, y),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#c55a11"))
        ax.plot([lo], [y], "|", ms=7, color="#c55a11")
    else:
        ax.annotate("", xy=(lo / 6, y), xytext=(lo, y),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3"))
        ax.plot([lo], [y], "|", ms=7, color="0.3")
    ax.text(0.65, y + 0.33, lab, fontsize=6.0, color="0.15")
ax.set_xscale("log")
ax.set_xlim(0.5, 3e7)
ax.set_ylim(-0.1, 7.0)
ax.set_yticks([])
ax.set_xlabel("quality factor of the moiré element, $Q$")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig2_q_scale.pdf"))
fig.savefig(os.path.join(OUT, "fig2_q_scale.png"), dpi=220)
plt.close(fig)
print("wrote", os.listdir(OUT))
