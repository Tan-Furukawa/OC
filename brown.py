import threading
import time
import io
import base64
from typing import Dict

import numpy as np
import matplotlib

# Use non-GUI backend for background thread
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flet as ft

# --------------------------------------------------
# Utility functions
# --------------------------------------------------


def generate_initial_positions(N: int, L: float) -> np.ndarray:
    """Initialize N particle positions uniformly in a square box of size L."""
    return np.random.rand(N, 2) * L


def positions_and_trajectories_to_base64(
    positions: np.ndarray, traj: np.ndarray, L: float
) -> str:
    """Render current positions and trajectories to PNG -> base64 string."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    # Plot trajectories
    for i in range(traj.shape[0]):
        ax.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.6)
    # Plot current positions
    ax.scatter(positions[:, 0], positions[:, 1], c="red", s=10)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------
# Simulation worker
# --------------------------------------------------


def simulation_worker(
    page: ft.Page, img_ctl: ft.Image, params: Dict, stop_evt: threading.Event
):
    """Run Brownian motion simulation in background thread."""
    p = params.copy()
    N = p["N_particles"]
    D = p["D"]
    dt = p["dt"]
    steps = p["steps"]
    disp_int = p["display_interval"]
    L = p["box_size"]
    sigma = np.sqrt(2 * D * dt)

    # Initialize
    positions = generate_initial_positions(N, L)
    # Trajectory history: shape (N, max_points, 2)
    # Estimate max_points = steps//disp_int + 1
    max_pts = steps // disp_int + 1
    traj = np.zeros((N, max_pts, 2), dtype=float)
    traj[:, 0, :] = positions
    t_idx = 1

    for n in range(steps):
        if stop_evt.is_set():
            break
        # Brownian update
        displ = sigma * np.random.randn(N, 2)
        positions += displ

        # ← ここを反射境界に変更
        # positions = np.mod(positions, L)  # ← これをコメントアウト or 削除
        # 反射境界条件
        # まず負の座標を反転
        positions = np.where(positions < 0, -positions, positions)
        # 次に座標がLを超えた分を反転
        positions = np.where(positions > L, 2 * L - positions, positions)

        # Record trajectory at display steps
        if n % disp_int == 0 and t_idx < max_pts:
            traj[:, t_idx, :] = positions
            t_idx += 1
            # Update image
            img_ctl.src_base64 = positions_and_trajectories_to_base64(
                positions, traj[:, :t_idx, :], L
            )
            img_ctl.update()
        time.sleep(0.001)


# --------------------------------------------------
# Flet UI
# --------------------------------------------------

DEFAULT_PARAMS: Dict = {
    "N_particles": 1,
    "D": 0.01,  # Diffusion coefficient
    "dt": 0.1,  # Time step
    "steps": 100000,
    "display_interval": 10,
    "box_size": 10.0,
}
# UI-exposed parameters
DISPLAY_PARAMS = ["N_particles", "D", "dt"]


def main(page: ft.Page):
    page.title = "Brownian Motion Simulator"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Image placeholder
    img = ft.Image(width=800, height=800, border_radius=ft.border_radius.all(4))

    # Parameter fields
    fields: Dict[str, ft.TextField] = {}
    for key, val in DEFAULT_PARAMS.items():
        fields[key] = ft.TextField(
            label=key,
            value=str(val),
            width=120,
            dense=True,
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=(key in DISPLAY_PARAMS),
        )
    apply_btn = ft.ElevatedButton(text="Apply / Restart", icon=ft.Icons.PLAY_ARROW)

    # Layout
    controls = ft.Column(
        [
            ft.Row(list(fields.values()) + [apply_btn], wrap=True),
            ft.Text("Box size, display_interval は隠しパラメータです。"),
        ]
    )
    page.add(controls, img)

    # Start first simulation
    stop_event = threading.Event()
    sim_params = DEFAULT_PARAMS.copy()
    page.run_thread(simulation_worker, page, img, sim_params, stop_event)

    # Event handler
    def apply_clicked(e):
        nonlocal stop_event, sim_params
        stop_event.set()
        # Parse inputs
        new_params: Dict = {}
        for k, tf in fields.items():
            try:
                if k in ("N_particles", "steps", "display_interval"):
                    new_params[k] = int(tf.value)
                else:
                    new_params[k] = float(tf.value)
                tf.error_text = None
            except ValueError:
                tf.error_text = "Invalid"
                tf.update()
                return
        # Merge hidden
        for k in DEFAULT_PARAMS:
            if k not in new_params:
                new_params[k] = sim_params.get(k, DEFAULT_PARAMS[k])
        sim_params = new_params
        stop_event = threading.Event()
        page.run_thread(simulation_worker, page, img, sim_params, stop_event)

    apply_btn.on_click = apply_clicked


if __name__ == "__main__":
    ft.app(target=main)
