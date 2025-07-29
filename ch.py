import threading
import time
import io
import base64
from typing import Dict

import numpy as np
import matplotlib

# Use a non-GUI backend because the simulation runs in a background thread
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fft2, ifft2
import flet as ft

# --------------------------------------------------
#   Utility functions
# --------------------------------------------------


def generate_initial_field(N: int, c0: float) -> np.ndarray:
    """Create random concentration field around `c0` (clipped to [0, 1])."""
    field = c0 + 0.01 * (np.random.rand(N, N) - 0.5)
    # Ensuring physical bounds 0≤c≤1 for extreme values of c0
    return np.clip(field, 0.0, 1.0)


def field_to_base64(c: np.ndarray) -> str:
    """Render field to PNG -> base64 string."""
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(c, cmap="viridis", origin="lower", vmax=1, vmin=0)
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------
#   Simulation worker
# --------------------------------------------------


def simulation_worker(
    page: ft.Page,
    img_ctl: ft.Image,
    params: Dict,
    stop_evt: threading.Event,
):
    """Run Cahn–Hilliard simulation in a background thread.

    Restarts automatically after finishing `steps`, unless `stop_evt` is set
    (indicating parameters have changed).
    """
    # while not stop_evt.is_set():
    # Copy params locally to avoid race conditions
    p = params.copy()
    N, dx, dt = p["N"], p["dx"], p["dt"]
    steps, M, kappa = p["steps"], p["M"], p["kappa"]
    disp_int, c0 = p["display_interval"], p["c0"]

    c = generate_initial_field(N, c0)
    k = 2 * np.pi * fftfreq(N, dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2, K4 = KX**2 + KY**2, (KX**2 + KY**2) ** 2

    for n in range(steps):
        if stop_evt.is_set():
            break

        # Semi-implicit Cahn–Hilliard update
        f_prime = 2 * c * (1 - 3 * c + 2 * c**2)
        c_k = fft2(c)
        f_k = fft2(f_prime)
        c_k = (c_k - dt * M * K2 * f_k) / (1 + 2 * dt * M * kappa * K4)
        c = np.real(ifft2(c_k))

        if n % disp_int == 0 or n == steps - 1:
            img_ctl.src_base64 = field_to_base64(c)
            img_ctl.update()  # Safe inside page.run_thread

        # Tiny pause to keep UI responsive
        time.sleep(0.001)
        # Loop again to start a fresh simulation unless parameters changed


# --------------------------------------------------
#   Flet UI
# --------------------------------------------------

DEFAULT_PARAMS: Dict = {
    "N": 256,
    "dx": 1.0,
    "dt": 0.3,
    "steps": 6000,
    "M": 1.0,
    "kappa": 0.2,
    "display_interval": 10,
    "c0": 0.5,  # <-- Initial concentration
}

# Parameters we want visible in the control bar (others remain but are hidden)
DISPLAY_PARAMS = ["M", "kappa", "c0"]


def main(page: ft.Page):
    page.title = "Cahn–Hilliard Phase-Field Simulator"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Image placeholder
    img = ft.Image(width=1024, height=1024, border_radius=ft.border_radius.all(4))

    # Build parameter input fields
    fields: Dict[str, ft.TextField] = {}
    for key, val in DEFAULT_PARAMS.items():
        fields[key] = ft.TextField(
            label=key,
            value=str(val),
            width=120,
            dense=True,
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=key in DISPLAY_PARAMS,
        )

    apply_btn = ft.ElevatedButton(text="Apply / Restart", icon=ft.Icons.PLAY_ARROW)

    row = ft.Column(
        [
            ft.Text("", size=20),
            ft.Row(list(fields.values()) + [apply_btn], wrap=True, expand=False),
            ft.Text("M: モビリティー, kappa: 界面エネルギー, c0: 初期濃度"),
        ]
    )
    page.add(row, img)

    # Start first simulation
    stop_event = threading.Event()
    sim_params = DEFAULT_PARAMS.copy()
    page.run_thread(simulation_worker, page, img, sim_params, stop_event)

    # --------------------------------------------------
    #   Event handlers
    # --------------------------------------------------
    def apply_clicked(e):
        nonlocal stop_event, sim_params
        # Request current simulation to stop
        stop_event.set()

        # Parse updated params from UI
        new_params: Dict = {}
        for k, tf in fields.items():
            try:
                if k in ("N", "steps", "display_interval"):
                    new_params[k] = int(tf.value)
                else:
                    new_params[k] = float(tf.value)
                tf.error_text = None
            except ValueError:
                tf.error_text = "Invalid"
                tf.update()
                return

        # Merge with any hidden parameters that weren’t in UI
        for k in DEFAULT_PARAMS.keys():
            if k not in new_params:
                new_params[k] = sim_params.get(k, DEFAULT_PARAMS[k])

        sim_params = new_params

        # Launch a new simulation thread with fresh stop_event
        stop_event = threading.Event()
        page.run_thread(simulation_worker, page, img, sim_params, stop_event)

    apply_btn.on_click = apply_clicked


if __name__ == "__main__":
    ft.app(target=main)
