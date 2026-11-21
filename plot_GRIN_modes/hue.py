import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

class Hue:
    def __init__(self, field : np.ndarray, R : float):
        self.field = field
        self.R = R
        self.max_amplitude = np.max(np.abs(field))
    def prepare_hue_amplitude(self):
        relative_amplitude = np.abs(self.field) / self.max_amplitude if self.max_amplitude > 0 else np.abs(self.field)
        phase = np.angle(self.field)

        # HSV: hue encodes phase, saturation is fixed, brightness encodes amplitude.
        # Wrapping identifies -pi and pi as the same phase; zero amplitude is black.
        hue = ((phase+np.pi) / (2 * np.pi)) % 1.0
        phase_amplitude_rgb = hsv_to_rgb(
            np.stack((hue, np.ones_like(hue), relative_amplitude), axis=-1)
        )
        return relative_amplitude, phase_amplitude_rgb
    def plot_hue_amplitude(self, extent=(-10, 10, -10, 10), cmap="hsv"):
        relative_amplitude, phase_amplitude_rgb = self.prepare_hue_amplitude()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        fig.suptitle(f"Hue and amplitude visualization")

        amplitude_image = axes[0].imshow(
            relative_amplitude, extent=extent, aspect="equal",
            cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
        )
        axes[0].set_title("Relative wave amplitude")
        fig.colorbar(amplitude_image, ax=axes[0], label=r"$|\psi| / \max|\psi|$")

        axes[1].imshow(
            phase_amplitude_rgb, extent=extent, aspect="equal",
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
                (0, 0), self.R, fill=False, edgecolor="white", linestyle="--", linewidth=1.5,
                label=f"Core boundary (R = {self.R:g} um)",
            ))
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")

        axes[1].legend(loc="upper right", fontsize=8, facecolor="0.15", labelcolor="white")

        plt.show()