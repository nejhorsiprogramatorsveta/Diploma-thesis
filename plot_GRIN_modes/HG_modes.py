#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import scipy as sp
from scipy.special import factorial

def parabolic_profile(x : np.ndarray, y : np.ndarray, b : float, n0 : float) -> np.ndarray:
    r = np.sqrt(x**2 + y**2)
    return n0**2*(1 - r**2/b**2)

def beta_HG(nx : int, ny : int, alpha : float, k : float, n0 : float):
    return np.sqrt(k**2*n0**2 - 2*alpha*(nx + ny + 1))

def C_nx_ny(nx : int, ny : int, alpha : float):
    return np.sqrt(alpha/(np.pi*2**(nx+ny)*factorial(nx)*factorial(ny)))

def HG_mode(nx : int,
            ny: int,
            k : float,
            n0 : float,
            b : float,
            x: np.ndarray,
            y: np.ndarray,
            z: float,
            R: float) -> np.ndarray:
    alpha = k * n0/b
    N = nx+ny
    if N>alpha*R**2/2-1:
        raise ValueError(f"Mode (nx={nx}, ny={ny}) is not guided for the given parameters.",
                         f"{alpha=:.3f}, {R=:.3f}, {N=}")
    
    C = C_nx_ny(nx, ny, alpha)
    beta = beta_HG(nx, ny, alpha, k, n0)

    psi_HG = C*np.exp(-alpha*(x**2 + y**2)/2)*\
    sp.special.eval_hermite(nx, np.sqrt(alpha)*x)*\
    sp.special.eval_hermite(ny, np.sqrt(alpha)*y)*\
    np.exp(1j*beta*z)
    return psi_HG

#%%
x = np.linspace(-10, 10, 1000) #um
X,Y = np.meshgrid(x, x)

NA = 0.25
b = 80 #um
R = 6 #um; core-boundary guide for the plots, independent of b

n0 = NA*b/R

profile = parabolic_profile(x, np.zeros_like(x), b, n0)

plt.plot(x, np.sqrt(profile))
plt.title("Parabolic refractive index profile")
plt.xlabel("x (um)")
plt.ylabel("n(r)")
plt.grid()
plt.show()
# %%
nx, ny = 0,2
lam = 0.532 #um
k = 2*np.pi/lam
z=0 #um

psi_HG = HG_mode(nx, ny, k, n0, b, X, Y, z,R)

amplitude = np.abs(psi_HG)
peak_amplitude = amplitude.max()
relative_amplitude = amplitude / peak_amplitude if peak_amplitude > 0 else amplitude
phase = np.angle(psi_HG)

# HSV: hue encodes phase, saturation is fixed, brightness encodes amplitude.
# Wrapping identifies -pi and pi as the same phase; zero amplitude is black.
hue = ((phase+np.pi) / (2 * np.pi)) % 1.0
phase_amplitude_rgb = hsv_to_rgb(
    np.stack((hue, np.ones_like(hue), relative_amplitude), axis=-1)
)
extent = [x.min(), x.max(), x.min(), x.max()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.suptitle(f"HG mode (nx={nx}, ny={ny}), z={z:g} um")

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
