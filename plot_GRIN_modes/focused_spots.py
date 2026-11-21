#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import scipy as sp
from scipy.special import factorial
import tqdm

def beta_LG(l : int, p : int, alpha : float, k : float, n0 : float):
    return np.sqrt(k**2 * n0**2 - 2 * alpha*(2*p + abs(l) + 1))

def C_lp(l : int, p : int, alpha : float):
    return np.sqrt(alpha * factorial(p) / (np.pi * factorial( p + abs(l))))

def F_pl(p : int, l : int, alpha : float, r : np.ndarray) -> np.ndarray:
    u_lp = (alpha * r**2)**(abs(l)/2)*\
            np.exp(-alpha * r**2 / 2)*\
            sp.special.eval_genlaguerre(p, abs(l), alpha * r**2)
    
    return C_lp(l, p, alpha) * u_lp
        


def c_p0(p : int, r : np.ndarray, alpha : float) -> float:
    integrand = F_pl(p, 0, alpha, r) * r
    return  2 * np.pi * sp.integrate.trapezoid(integrand, x=r)

def psi_z(A : np.ndarray, l : int, k : float, n0 : float, b : float, x: np.ndarray, R : float, z: float) -> np.ndarray:
    X,Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)

    r_cp = np.linspace(0, R, 4001)
    alpha = k * n0/b
    p_max = int(np.floor(alpha*R**2/2-1)/2)
    print(f"Maximum p for guided modes: {p_max}")


    field = np.zeros(r.shape, dtype=complex)
    for p in range(p_max+1):
        field += c_p0(p, r_cp, alpha)  * F_pl(p, l, alpha, r) * np.exp(1j * beta_LG(l, p, alpha, k, n0) * z)

    return A * field
#%%
x = np.linspace(-10, 10, 1000) #um


NA = 0.4
b = 80 #um
R = 6 #um; core-boundary guide for the plots, independent of b

n0 = NA*b/R
lam = 0.532 #um
k = 2*np.pi/lam
zf = np.pi*b/2

X, Y = np.meshgrid(x, x)
A = 1

z = zf+10


field1= psi_z(A, 0, k, n0, b, x, R, z)
field2 = psi_z(A, 0, k, n0, b, x, R, z+10)
plt.imshow(np.abs(field1)**2, extent=(-10, 10, -10, 10), cmap="viridis")
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={zf}")
plt.show()

common_max = max(np.max(np.abs(field1)**2), np.max(np.abs(field2)**2))

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(np.abs(field1)**2, extent=(-10, 10, -10, 10),cmap = "viridis", vmin=0, vmax=common_max)
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={z}")
plt.subplot(1,2,2)
plt.imshow(np.abs(field2)**2, extent=(-10, 10, -10, 10),cmap = "viridis", vmin=0, vmax=common_max)
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={z+10}")
plt.show()

#%%
from matplotlib.animation import FuncAnimation

# Průřez: x a y jsou v mikrometrech.
# Menší počet bodů než původních 1000 urychlí výpočet.
x = np.linspace(-10, 10, 500)  # um
X, Y = np.meshgrid(x, x)

# Polohy podél vlákna, rovněž v mikrometrech.
z_values = np.linspace(0, 4*zf, 400)  # um

# 1. Předem vypočítáme intenzitu pro všechny polohy z.
intensities = []

for z in z_values:
    field = psi_z(A, 0, k, n0, b, x, R, z)
    intensity = np.abs(field)**2
    intensities.append(intensity)

intensities = np.asarray(intensities)

# 2. Jedna společná normalizace pro celou animaci.
global_max = np.max(intensities)

if global_max == 0:
    raise ValueError("Intenzita je ve všech snímcích nulová.")

relative_intensities = intensities / global_max

# 3. Připravíme obrázek a barevnou škálu.
fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    relative_intensities[0],
    extent=(x[0], x[-1], x[0], x[-1]),
    origin="lower",
    cmap="viridis",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

# Červená kružnice vyznačuje poloměr jádra.
core_boundary = plt.Circle(
    (0, 0), R,
    color="red",
    fill=False,
    linewidth=2,
    label=f"Poloměr jádra R = {R} um",
)
ax.add_patch(core_boundary)

ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_aspect("equal")
ax.legend(loc="upper right")

title = ax.set_title(f"Relativní intenzita, z = {z_values[0]:.1f} um")
colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Relativní intenzita")

# 4. Při každém snímku změníme pouze data obrázku a titulek.
def update(frame):
    image.set_data(relative_intensities[frame])
    title.set_text(f"Relativní intenzita, z = {z_values[frame]:.1f} um")
    return image, title

animation = FuncAnimation(
    fig,
    update,
    frames=len(z_values),
    interval=80,       # prodleva mezi snímky v ms
    blit=False,
    repeat=True,
)
# animation.save("intensity_animation.gif", writer="pillow", fps=12)
plt.show()

#%% ############## dopad rovinne vlny pod obecnym uhlem ##############

class Planewave:
    def __init__(self, 
                 A : float,
                 k : float,
                 n0 :float,
                 b :float, 
                 x :float,
                 R :float,
                 theta :float,
                 phi_k : float,
                 z :float, ):
        self.A = A
        self.k = k
        self.n0 = n0
        self.b = b
        self.x = x
        self.R = R
        self.theta = theta
        self.z = z
        self.alpha = k * n0/b
        self.phi_k = phi_k
        self.kx = self.k * np.sin(self.theta) * np.cos(self.phi_k)
        self.ky = self.k * np.sin(self.theta) * np.sin(self.phi_k)
        self.k_perp = np.hypot(self.kx, self.ky)
        self.modes = self.allowed_modes()
        self._precalculate_c_pl()
        self._precalculate_fields()

    def beta_LG(self, p : int, l : int):
        return np.sqrt(self.k**2 * self.n0**2 - 2 * self.alpha*(2*p + abs(l) + 1))

    def K_pl(self,p : int, l : int ):
        return np.sqrt(self.alpha * factorial(p) / (np.pi * factorial( p + abs(l))))

    def F_pl(self, p : int, l : int, r : np.ndarray) -> np.ndarray:
        u_lp = (self.alpha * r**2)**(abs(l)/2)*\
                np.exp(-self.alpha * r**2 / 2)*\
                sp.special.eval_genlaguerre(p, abs(l), self.alpha * r**2)
        
        return self.K_pl(p, l) * u_lp
    def c_pl(self, p : int, l : int, r : np.ndarray) -> float:
        integrand = sp.special.jv(l, self.k_perp * r)*self.F_pl(p, l, r) * r
        return  self.A * 2 * np.pi * np.exp(-1j * l * self.phi_k) *\
              (1j)**l * sp.integrate.simpson(integrand, x=r)

    def allowed_modes(self):
        N_limit = self.alpha * self.R**2 / 2 - 1
        N_max = int(np.ceil(N_limit)) - 1

        modes = []

        for l in range(-N_max, N_max + 1):
            p_max = (N_max - abs(l)) // 2

            for p in range(p_max + 1):
                modes.append((p, l))
        self.modes = modes
        return modes
    def _precalculate_c_pl(self):
        r_cpl = np.linspace(0, self.R, 4001)
        self.c_pl_values = {}
        for p, l in self.modes:
            self.c_pl_values[(p, l)] = self.c_pl(p, l, r_cpl)
    def _precalculate_fields(self):
        X, Y = np.meshgrid(self.x, self.x)
        r = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)

        self.fields = {}
        for p, l in self.modes:
            field = self.F_pl(p, l, r) * np.exp(1j * l * phi)
            self.fields[(p, l)] = field
    def calculate_field(self) -> np.ndarray:

        field = np.zeros((len(self.x), len(self.x)), dtype=complex)
        for p, l in self.modes:
            field += self.c_pl_values[(p, l)] *\
                self.fields[(p, l)] *\
                np.exp(1j * self.beta_LG(p, l) * self.z)

        return field


# %%
x = np.linspace(-10, 10, 1000) #um

# fiber
NA = 0.4
b = 80 #um
R = 6 #um; core-boundary guide for the plots, independent of b

n0 = NA*b/R
lam = 0.532 #um
k = 2*np.pi/lam

# plane wave:
A = 1
theta = np.deg2rad(5)
phi_k = np.deg2rad(180)
# distance along the fiber axis where we want to evaluate the field
zf = np.pi*b/2
L_rep = 2*np.pi*b-np.pi*R*NA/(2*k*n0)

wave = Planewave(A, k, n0, b, x, R, theta, phi_k, 1*L_rep+zf)
field = wave.calculate_field()

plt.imshow(np.abs(field)**2, extent=(-10, 10, -10, 10), cmap="viridis")
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={L_rep:.1f} um, theta={np.rad2deg(theta):.1f} deg, phi_k={np.rad2deg(phi_k):.1f} deg")
plt.show()

#%%
from matplotlib.animation import FuncAnimation
# Průřez: x a y jsou v mikrometrech.
# Menší počet bodů než původních 1000 urychlí výpočet.
x = np.linspace(-10, 10, 1000)  # um
X, Y = np.meshgrid(x, x)
theta = np.deg2rad(60)
phi_k = np.deg2rad(0)
# Polohy podél vlákna, rovněž v mikrometrech.
z_values = np.linspace(0, 4*zf, 200)  # um

# 1. Předem vypočítáme intenzitu pro všechny polohy z.
intensities = []
wave = Planewave(A, k, n0, b, x, R, theta, phi_k, 0)
print(f"Field precomputed for {len(z_values)} z-values.")

for z in tqdm.tqdm(z_values):
    wave.z = z
    field = wave.calculate_field()
    intensity = np.abs(field)**2
    intensities.append(intensity)

intensities = np.asarray(intensities)

# 2. Jedna společná normalizace pro celou animaci.
global_max = np.max(intensities)

if global_max == 0:
    raise ValueError("Intenzita je ve všech snímcích nulová.")

relative_intensities = intensities / global_max

# 3. Připravíme obrázek a barevnou škálu.
fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    relative_intensities[0],
    extent=(x[0], x[-1], x[0], x[-1]),
    origin="lower",
    cmap="viridis",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

# Červená kružnice vyznačuje poloměr jádra.
core_boundary = plt.Circle(
    (0, 0), R,
    color="red",
    fill=False,
    linewidth=2,
    label=f"Poloměr jádra R = {R} um",
)
ax.add_patch(core_boundary)

ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_aspect("equal")
ax.legend(loc="upper right")

title = ax.set_title(f"Relativní intenzita, z = {z_values[0]:.1f} um")
colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Relativní intenzita")

# 4. Při každém snímku změníme pouze data obrázku a titulek.
def update(frame):
    image.set_data(relative_intensities[frame])
    title.set_text(f"Relativní intenzita, z = {z_values[frame]:.1f} um")
    return image, title

animation = FuncAnimation(
    fig,
    update,
    frames=len(z_values),
    interval=80,       # prodleva mezi snímky v ms
    blit=False,
    repeat=True,
)
animation.save(f"intensity_{np.round(np.rad2deg(theta))}_{np.round(np.rad2deg(phi_k))}.gif", writer="pillow", fps=12)
plt.show()