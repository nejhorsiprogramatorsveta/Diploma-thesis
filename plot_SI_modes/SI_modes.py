#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import scipy as sp


# %%
def step_index_profile(x : np.ndarray, a : float,
                       n0 : float, n1 : float) -> np.ndarray:
    return np.where(np.abs(x) < a, n0, n1)

def boundary_equation(u , V : float, l : int):
    w = np.sqrt((V - u) * (V + u))
    eq = u * sp.special.jv(l+1, u) * sp.special.kv(l, w)-\
        w * sp.special.kv(l+1, w) * sp.special.jv(l, u)
    return eq

def roots_of_ul(l : int, n : int, a : float, k0 : float, NA : float):
    """Return guided roots with radial indices 0 through n, in increasing order."""
    V = a * k0 * NA
    if n < 0:
        raise ValueError("The radial mode index n must be at least 0.")
    if not np.isfinite(V) or V <= 0:
        raise ValueError("V = a * k0 * NA must be finite and positive.")

    l = abs(l)
    count = n + 1
    upper_bounds = sp.special.jn_zeros(l, count)
    if l == 0:
        lower_bounds = np.zeros(count)
        if count > 1:
            lower_bounds[1:] = sp.special.jn_zeros(1, count-1)
    else:
        lower_bounds = sp.special.jn_zeros(l-1, count)

    roots = []
    # Each pair of Bessel zeros brackets one radial branch above its cutoff.
    for lower, upper in zip(lower_bounds, upper_bounds):
        if lower >= V:
            break
        # K_l(w) is singular at w=0, so keep the endpoint strictly below V.
        upper = min(upper, np.nextafter(V, 0.0))
        root = sp.optimize.brentq(boundary_equation, lower, upper, args=(V, l))
        roots.append(root)

    if len(roots) < count:
        raise ValueError(
            f"Only {len(roots)} guided modes exist for |l|={l} at V={V:.6g}; "
            f"requested radial mode n={n}."
        )
    return np.array(roots)

def radial_part(r : np.ndarray, l : int, beta : float, k0 : float,
                n0 : float, a : float, NA : float) -> np.ndarray:
    q = np.sqrt(k0**2 * n0**2 - beta**2)
    kappa = np.sqrt(beta**2 - k0**2 * (n0**2 - NA**2))
    radial = np.where(r < a,
                      sp.special.jv(l, q*r)/sp.special.jv(l, q*a),
                      sp.special.kv(l, kappa*r)/sp.special.kv(l, kappa*a))
    return radial

def SI_mode(l : int, n : int, k0 : float, n0 : float, a : float, NA : float,
            r : np.ndarray, phi : np.ndarray, z : np.ndarray) -> np.ndarray:
    roots = roots_of_ul(l, n, a, k0, NA)
    u = roots[n]
    beta = np.sqrt(k0**2 * n0**2 - (u/a)**2)
    radial = radial_part(r, l, beta, k0, n0, a, NA)
    angular = np.exp(1j * l * phi)
    z_part = np.exp(1j * beta * z) # z=0
    return radial * angular * z_part
    
# %%
x = np.linspace(-10, 10, 1000) #um
X,Y = np.meshgrid(x, x)
r = np.sqrt(X**2 + Y**2)
NA = 0.15
a = 6 #um
n0 = 1.5
n1 = np.sqrt(n0**2 - NA**2)
lam = 532e-3 #um
k0 = 2*np.pi/lam

profile = step_index_profile(x, a, n0, n1)
plt.plot(x, profile)
plt.xlabel("x (um)")
plt.ylabel("n(r)")
plt.show()
# %%
n, l = 2 , 1

psi = SI_mode(l , n , k0, n0, a, NA, r, np.arctan2(Y, X), 0)
amplitude = np.abs(psi)
peak_amplitude = amplitude.max()
relative_amplitude = amplitude / peak_amplitude if peak_amplitude > 0 else amplitude
phase = np.angle(psi)

hue = ((phase+np.pi) / (2 * np.pi)) % 1.0
phase_amplitude_rgb = hsv_to_rgb(
    np.stack((hue, np.ones_like(hue), relative_amplitude), axis=-1)
)
extent = [x.min(), x.max(), x.min(), x.max()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.suptitle(f"LG mode (l={l}, p={n}), z={0:g} um")

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
        (0, 0), a, fill=False, edgecolor="white", linestyle="--", linewidth=1.5,
        label=f"Core boundary (R = {a:g} um)",
    ))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(x.min(), x.max())
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")

axes[1].legend(loc="upper right", fontsize=8, facecolor="0.15", labelcolor="white")

plt.show()
# %%
