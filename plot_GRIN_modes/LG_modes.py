#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import scipy as sp
from scipy.special import factorial


def parabolic_profile(r : np.ndarray, b : float,
                      n0 : float) -> np.ndarray:
    return n0**2*(1 - r**2/b**2)

def beta_LG(l : int, p : int, alpha : float, k : float, n0 : float):
    return np.sqrt(k**2 * n0**2 - 2 * alpha*(2*p + abs(l) + 1))

def C_lp(l : int, p : int, alpha : float):
    return np.sqrt(alpha * factorial(p) / (np.pi * factorial( p + abs(l))))

def LG_mode(l : int,
            p : int,
            k : float,
            n0 : float,
            b : float,
            r: np.ndarray,
            phi: np.ndarray,
            z: float,
            R: float) -> np.ndarray:
    alpha = k * n0/b
    N = 2*p + abs(l)
    if N>alpha*R**2/2-1:
        raise ValueError(f"Mode (l={l}, p={p}) is not guided for the given parameters.",
                         f"{alpha=:.3f}, {R=:.3f}, {N=}")
    
    C = C_lp(l, p, alpha)
    beta = beta_LG(l, p, alpha, k, n0)

    psi_LG = C*(alpha * r**2)**(abs(l)/2)*\
    np.exp(-alpha * r**2 / 2)*\
    sp.special.eval_genlaguerre(p, abs(l), alpha * r**2)*\
    np.exp(1j * l * phi)*\
    np.exp(1j * beta * z)
    return psi_LG

# %%
x = np.linspace(-10, 10, 1000) #um
X,Y = np.meshgrid(x, x)
r = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

NA = 0.25
b = 80 #um
R = 6 #um; core-boundary guide for the plots, independent of b

n0 = NA*b/R

profile = parabolic_profile(r[len(r)//2, :], b, n0)

plt.plot(x, np.sqrt(profile))
plt.title("Parabolic refractive index profile")
plt.xlabel("x (um)")
plt.ylabel("n(r)")
plt.grid()
plt.show()

# %%
p, l = 1,2
lam = 0.532 #um
k = 2*np.pi/lam
z=0 #um

psi = LG_mode(l, p, k, n0, b, r, phi, z, R)
amplitude = np.abs(psi)
peak_amplitude = amplitude.max()
relative_amplitude = amplitude / peak_amplitude if peak_amplitude > 0 else amplitude
phase = np.angle(psi)

# HSV: hue encodes phase, saturation is fixed, brightness encodes amplitude.
# Wrapping identifies -pi and pi as the same phase; zero amplitude is black.
hue = ((phase+np.pi) / (2 * np.pi)) % 1.0
phase_amplitude_rgb = hsv_to_rgb(
    np.stack((hue, np.ones_like(hue), relative_amplitude), axis=-1)
)
extent = [x.min(), x.max(), x.min(), x.max()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.suptitle(f"LG mode (l={l}, p={p}), z={z:g} um")

amplitude_image = axes[0].imshow(
    relative_amplitude, extent=extent, origin="lower", aspect="equal",
    cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
)
axes[0].set_title("Relative wave amplitude")
fig.colorbar(amplitude_image, ax=axes[0], label=r"$|\psi| / \max|\psi|$")

axes[1].imshow(
    phase_amplitude_rgb, extent=extent, origin="lower", aspect="equal",
    interpolation="nearest",
)
axes[1].set_title("Hue: phase | Brightness: relative amplitude")
# Show the phase-to-hue key at full brightness.
phase_key = plt.cm.ScalarMappable(
    norm=plt.Normalize(vmin=-np.pi, vmax=np.pi), cmap="hsv",
)
phase_colorbar = fig.colorbar(phase_key, ax=axes[1], label="Phase (rad; full brightness)")
phase_colorbar.set_ticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])

for ax in axes:
    ax.add_patch(plt.Circle(
        (0, 0), R, fill=False, edgecolor="white", linestyle="--", linewidth=1.5,
        label=f"Core boundary (R = {R:g} um)",
    ))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(x.min(), x.max())
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")

axes[1].legend(loc="upper right", fontsize=8, facecolor="0.15", labelcolor="white")

plt.show()


# %%
