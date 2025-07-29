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


def time_x_plot_to_base64(times: np.ndarray, x_values: np.ndarray) -> str:
    """Render time vs x plot to base64 PNG."""
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    ax.plot(times, x_values, linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("x position")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------
# Simulation worker
# --------------------------------------------------


def simulation_worker(
    page: ft.Page,
    img_ctl: ft.Image,
    tx_ctl: ft.Image,
    params: Dict,
    stop_evt: threading.Event,
    pause_evt: threading.Event,
    buffers: Dict[str, np.ndarray],
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

    positions = generate_initial_positions(N, L)

    max_pts = steps // disp_int + 1
    traj = np.zeros((N, max_pts, 2), dtype=float)
    traj[:, 0, :] = positions
    t_idx = 1

    times = [0.0]
    x_series = [positions[0, 0]]

    for n in range(1, steps + 1):
        if stop_evt.is_set():
            break

        # 一時停止中は待機
        while pause_evt.is_set() and not stop_evt.is_set():
            time.sleep(0.05)

        displ = sigma * np.random.randn(N, 2)
        positions += displ

        # Reflective boundaries
        positions = np.where(positions < 0, -positions, positions)
        positions = np.where(positions > L, 2 * L - positions, positions)

        times.append(n * dt)
        x_series.append(positions[0, 0])

        if n % disp_int == 0 and t_idx < max_pts:
            traj[:, t_idx, :] = positions
            t_idx += 1

            img_ctl.src_base64 = positions_and_trajectories_to_base64(
                positions, traj[:, :t_idx, :], L
            )
            img_ctl.update()

            tx_ctl.src_base64 = time_x_plot_to_base64(
                np.array(times), np.array(x_series)
            )
            tx_ctl.update()

        time.sleep(0.001)

    buffers["times"] = np.array(times)
    buffers["x"] = np.array(x_series)


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
DISPLAY_PARAMS = ["N_particles", "D", "dt"]


def main(page: ft.Page):
    page.title = "Brownian Motion Simulator"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO

    img = ft.Image(width=500, height=500, border_radius=ft.border_radius.all(4))
    tx_img = ft.Image(width=500, height=250, border_radius=ft.border_radius.all(4))

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
    stop_btn = ft.ElevatedButton(text="停止", icon=ft.Icons.STOP)
    resume_btn = ft.ElevatedButton(text="再開", icon=ft.Icons.PLAY_ARROW, disabled=True)

    controls = ft.Column(
        [
            ft.Row(
                list(fields.values()) + [apply_btn, stop_btn, resume_btn], wrap=True
            ),
            ft.Text("Box size, display_interval は隠しパラメータです。"),
        ]
    )
    plots_row = ft.Row([img, tx_img], alignment=ft.MainAxisAlignment.CENTER)
    page.add(controls, plots_row)

    buffers: Dict[str, np.ndarray] = {}

    stop_event = threading.Event()
    pause_event = threading.Event()
    sim_params = DEFAULT_PARAMS.copy()

    def start_thread():
        page.run_thread(
            simulation_worker,
            page,
            img,
            tx_img,
            sim_params,
            stop_event,
            pause_event,
            buffers,
        )

    # 初回開始
    start_thread()

    def apply_clicked(e):
        nonlocal stop_event, pause_event, sim_params, buffers
        # 旧スレッド停止
        stop_event.set()

        # 入力検証
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

        # 隠しパラメータは維持
        for k in DEFAULT_PARAMS:
            if k not in new_params:
                new_params[k] = sim_params.get(k, DEFAULT_PARAMS[k])
        sim_params = new_params

        # 状態をリセット
        buffers = {}
        stop_event = threading.Event()
        pause_event = threading.Event()
        stop_btn.disabled = False
        resume_btn.disabled = True
        stop_btn.update()
        resume_btn.update()

        start_thread()

    def stop_clicked(e):
        # 一時停止
        pause_event.set()
        stop_btn.disabled = True
        resume_btn.disabled = False
        stop_btn.update()
        resume_btn.update()

    def resume_clicked(e):
        pause_event.clear()
        stop_btn.disabled = False
        resume_btn.disabled = True
        stop_btn.update()
        resume_btn.update()

    apply_btn.on_click = apply_clicked
    stop_btn.on_click = stop_clicked
    resume_btn.on_click = resume_clicked


if __name__ == "__main__":
    ft.app(target=main)
