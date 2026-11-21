#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import scipy as sp
from scipy.special import factorial
import tqdm
from matplotlib.animation import FuncAnimation
import hue


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
        self._precalculate_beta_LG()
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
    
    def _precalculate_beta_LG(self):
        self.beta_LG_values = np.zeros((len(self.modes)), dtype=float)
        for i, (p, l) in enumerate(self.modes):
            self.beta_LG_values[i] = self.beta_LG(p, l)
        print("Finished calculating Betas")

    def _precalculate_c_pl(self):
        r_cpl = np.linspace(0, self.R*1.1, 4001)
        self.c_pl_values = np.zeros((len(self.modes)), dtype=complex)
        for i, (p, l) in enumerate(self.modes):
            self.c_pl_values[i] = self.c_pl(p, l, r_cpl)
        print("Finished calculating coefficients")

    def _precalculate_fields(self):
        X, Y = np.meshgrid(self.x, self.x)
        r = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)

        self.fields = np.empty((len(self.x), len(self.x), len(self.modes)), dtype=complex)
        for mode_idx, (p, l) in enumerate(self.modes):
            self.fields[..., mode_idx] = self.F_pl(p, l, r) * np.exp(1j * l * phi)
        print("Finished calculating fields")
    def calculate_field(self) -> np.ndarray:
        weights = self.c_pl_values * np.exp(1j * self.beta_LG_values * self.z)
        output_field = np.einsum('ijk,k->ij', self.fields, weights)
        return output_field

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
theta = np.deg2rad(10)
phi_k = np.deg2rad(0)
# distance along the fiber axis where we want to evaluate the field
zf = np.pi*b/2
L_rep = 2*np.pi*b-np.pi*R*NA/(2*k*n0)
z=3*L_rep+zf
z=0

wave = Planewave(A, k, n0, b, x, R, theta, phi_k, z)
field = wave.calculate_field()

plot = hue.Hue(field, R)
plot.plot_hue_amplitude(extent=(-10, 10, -10, 10), cmap="hsv")

plt.figure(figsize=(7, 6))
plt.imshow(np.abs(field)**2, extent=(-10, 10, -10, 10), cmap="viridis")
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={wave.z:.1f} um, theta={np.rad2deg(theta):.1f} deg, phi_k={np.rad2deg(phi_k):.1f} deg")
plt.show()

#%%

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
# animation.save(f"intensity_{np.round(np.rad2deg(theta))}_{np.round(np.rad2deg(phi_k))}.gif", writer="pillow", fps=12)
plt.show()
#%%
class SelfImaging:
    def __init__(self, field, k, n0, b, x, R):
        self.input_field = np.asarray(field)
        self.k = k
        self.n0 = n0
        self.b = b
        x = np.asarray(x)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be a one-dimensional coordinate axis with at least two points.")
        self.X, self.Y = np.meshgrid(x, x)
        self.x = self.X
        if self.x.shape != self.input_field.shape:
            raise ValueError(
                f"Input field shape {self.input_field.shape} must match "
                f"the coordinate grid shape {self.x.shape}."
            )
        self.r = np.sqrt(self.X**2 + self.Y**2)
        self.phi = np.arctan2(self.Y, self.X)
        self.R = R
        self.alpha = k * n0/b
        self.modes = self.allowed_modes()
        self._precalculate_beta_LG()
        self._precalculate_LG_modes()
        self._precalculate_c_pl()

    def allowed_modes(self):
        N_limit = self.alpha * self.R**2 / 2 - 1
        N_max = int(np.ceil(N_limit)) - 1
        modes = []
        for l in range(-N_max, N_max + 1):
            p_max = (N_max - abs(l)) // 2

            for p in range(p_max + 1):
                modes.append((p, l))
        return modes
    
    def beta_LG(self, p : int, l : int):
        return np.sqrt(self.k**2 * self.n0**2 - 2 * self.alpha*(2*p + abs(l) + 1))
    
    def K_pl(self, l : int, p : int, alpha : float):
        return np.sqrt(alpha * factorial(p) / (np.pi * factorial( p + abs(l))))

    def LG_mode(self,p : int, l : int) -> np.ndarray:
        N = 2*p + abs(l)
        if N>=self.alpha*self.R**2/2-1:
            raise ValueError(f"Mode (l={l}, p={p}) is not guided for the given parameters.",
                            f"{self.alpha=:.3f}, {self.R=:.3f}, {N=}")
        
        C = self.K_pl(l, p, self.alpha)

        psi_LG = C*(self.alpha * self.r**2)**(abs(l)/2)*\
        np.exp(-self.alpha * self.r**2 / 2)*\
        sp.special.eval_genlaguerre(p, abs(l), self.alpha * self.r**2)*\
        np.exp(1j * l * self.phi)
        return psi_LG

    def _precalculate_beta_LG(self):
        self.beta_LG_values = np.zeros((len(self.modes)), dtype=float)
        for i, (p, l) in enumerate(self.modes):
            self.beta_LG_values[i] = self.beta_LG(p, l)
        print("Finished calculating Betas")


    def _precalculate_LG_modes(self):
        self.LG_modes = np.empty((*self.x.shape, len(self.modes)), dtype=complex)
        for mode_idx, (p, l) in enumerate(self.modes):
            self.LG_modes[..., mode_idx] = self.LG_mode(p, l)
        print("Finished calculating LG modes")

    def _precalculate_c_pl(self):
        self.c_pl_values = np.zeros((len(self.modes)), dtype=complex)
        dx = self.x[0, 1] - self.x[0, 0]
        for i in range(len(self.modes)):
            self.c_pl_values[i] = np.sum(
                np.conj(self.LG_modes[..., i]) * self.input_field
            ) * dx**2
        print("Finished calculating coefficients")
        

    def calculate_field(self, z : float) -> np.ndarray:
        self.z = z
        weights = self.c_pl_values * np.exp(1j * self.beta_LG_values * z)
        output_field = np.einsum('ijk,k->ij', self.LG_modes, weights)
        return output_field

# %%
x = np.linspace(-10, 10, 1000) #um
X,Y = np.meshgrid(x,x)
# fiber
NA = 0.4
b = 80 #um
R = 6 #um; core-boundary guide for the plots, independent of b
n0 = NA*b/R
lam = 0.532 #um
k = 2*np.pi/lam

# plane wave:
theta = np.deg2rad(10)
phi_k = np.deg2rad(0)
kx = k*np.sin(theta)*np.cos(phi_k)
ky = k*np.sin(theta)*np.sin(phi_k)

field = np.exp(1j* (kx*X+ky*Y))

# distance along the fiber axis where we want to evaluate the field
zf = np.pi*b/2
L_rep = 2*np.pi*b-np.pi*R*NA/(2*k*n0)
z=3*L_rep+zf
z=0
image = SelfImaging(field,k,n0,b,x,R)
wave = image.calculate_field(z)

plot = hue.Hue(wave,R)
plot.plot_hue_amplitude()

plt.figure(figsize=(7, 6))
plt.imshow(np.abs(wave)**2, extent=(-10, 10, -10, 10), cmap="viridis")
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={image.z:.1f} um, theta={np.rad2deg(theta):.1f} deg, phi_k={np.rad2deg(phi_k):.1f} deg")
plt.show()

# %%
# kruh:
r= np.sqrt((X-2/3*R)**2+(Y-0)**2)
field = np.where(r<2,1,0)
z=101*zf
plt.imshow(field,extent=(-10, 10, -10, 10),cmap="gray")
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.show()

image = SelfImaging(field,k,n0,b,x,R)
wave = image.calculate_field(z)

plot = hue.Hue(wave,R)
plot.plot_hue_amplitude()

plt.figure(figsize=(7, 6))
plt.imshow(np.abs(wave)**2, extent=(-10, 10, -10, 10), cmap="viridis")
plt.gca().add_patch(
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={image.z:.1f} um, theta={np.rad2deg(theta):.1f} deg, phi_k={np.rad2deg(phi_k):.1f} deg")
plt.show()

# %%
def parabolic_profile(r : np.ndarray, b : float,
                      n0 : float) -> np.ndarray:
    return n0**2*(1 - r**2/b**2)
#fiber
NA = 0.3
R = 25 #um
b = 100 #um
n0 = NA*b/R
lam = 0.532 #um
k = 2*np.pi/lam

x = np.linspace(-30,30,200)
profile = parabolic_profile(x[len(x)//2:],b,n0)
plt.plot(x[len(x)//2:],profile)
plt.show()

# random maska:
X, Y = np.meshgrid(x, x)

vertical = (
    (X >= -0.55 * R) & (X <= -0.25 * R)
    & (Y >= -0.50 * R) & (Y <= 0.55 * R)
)

horizontal = (
    (X >= -0.55 * R) & (X <= 0.50 * R)
    & (Y >= -0.50 * R) & (Y <= -0.20 * R)
)

field = (vertical | horizontal).astype(float)

fig, ax = plt.subplots()
ax.imshow(
    field,
    extent=(x[0], x[-1], x[0], x[-1]),
    cmap="gray",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
ax.add_patch(plt.Circle((0, 0), R, fill=False, color="red"))
ax.set(xlabel="x (µm)", ylabel="y (µm)", title="Vstupní binární maska")
plt.show()

L_rep = 2*np.pi*b-np.pi*R*NA/(2*k*n0)
print(L_rep)
zf = np.pi*b/2
z= 5*L_rep#20*zf

image = SelfImaging(field,k,n0,b,x,R)
wave = image.calculate_field(z)

plot = hue.Hue(wave,R)
plot.plot_hue_amplitude()

plt.figure(figsize=(7, 6))
plt.imshow(np.abs(wave)**2, extent=(-40, 40, -40, 40), cmap="viridis")
plt.gca().add_patch(    
    plt.Circle((0, 0), R, color="red", fill=False, linewidth=2)
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title(f"Plane wave in fiber at z={image.z:.1f} um")
plt.show()