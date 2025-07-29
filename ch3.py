import threading
import time
import io
import base64
from typing import Dict

import numpy as np
import matplotlib

# Non-GUI backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fft2, ifft2
import flet as ft

# --------------------------------------------------
# Utility functions
# --------------------------------------------------


def generate_initial_field(N: int, c0: float) -> np.ndarray:
    field = c0 + 0.01 * (np.random.rand(N, N) - 0.5)
    return np.clip(field, 0.0, 1.0)


def field_to_base64(c: np.ndarray) -> str:
    fig = plt.figure(figsize=(7, 7), dpi=150)
    plt.imshow(c, cmap="viridis", origin="lower", vmax=1, vmin=0)
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_gc_with_samples_to_base64(c: np.ndarray) -> str:
    # g(c) = c^2 (1-c)^2
    c_vals = np.linspace(0, 1, 500)
    g_vals = c_vals**2 * (1 - c_vals) ** 2

    # ランダムに100点サンプリング
    flat = c.ravel()
    idx = np.random.choice(flat.size, size=min(100, flat.size), replace=False)
    a_samp = flat[idx]
    g_samp = a_samp**2 * (1 - a_samp) ** 2

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    # 黒背景＋白線
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.plot(c_vals, g_vals, color="white", linewidth=2)
    r = np.random.random(len(g_samp)) * 0.003
    ax.scatter(a_samp, g_samp + 0.001 + r, c="blue", s=10)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, g_vals.max() * 2)
    ax.set_xlabel("c", color="white")
    ax.set_ylabel("g(c)", color="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.tick_params(colors="white")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_free_energy(c: np.ndarray, dx: float, kappa: float):
    # ポテンシャル項 g(c)=c^2(1−c)^2
    g = c**2 * (1 - c) ** 2
    F_pot = np.sum(g) * dx * dx
    # 界面項 kappa/2 |∇c|^2
    dc_dx, dc_dy = np.gradient(c, dx)
    grad2 = dc_dx**2 + dc_dy**2
    F_int = 0.5 * kappa * np.sum(grad2) * dx * dx
    return F_pot, F_int, F_pot + F_int


def plot_energy_to_base64(
    F_pot: np.ndarray, F_int: np.ndarray, F_tot: np.ndarray, total_steps: int
) -> str:
    steps = np.arange(len(F_tot))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    # 黒背景
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 各項プロット（色は任意で視認性優先）
    ax.plot(steps, F_pot, label="Potential", color="cyan", linewidth=2)
    ax.plot(steps, F_int, label="Interface", color="magenta", linewidth=2)
    ax.plot(steps, F_tot, label="Total", color="yellow", linewidth=2)

    # 軸ラベル・目盛り白、スパイン白
    ax.set_xlabel("Step", color="white")
    ax.set_ylabel("Free Energy", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

    # 横軸を 0 から total_steps で固定
    ax.set_xlim(0, total_steps)

    # 凡例も白文字
    ax.legend(facecolor="black", labelcolor="white", loc="upper right")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------
# Simulation worker
# --------------------------------------------------


def simulation_worker(
    page: ft.Page,
    img_field: ft.Image,
    img_cg: ft.Image,
    img_energy: ft.Image,
    params: Dict,
    stop_evt: threading.Event,
):
    p = params.copy()
    N, dx, dt = p["N"], p["dx"], p["dt"]
    steps, M, kappa = p["steps"], p["M"], p["kappa"]
    disp_int, c0 = p["display_interval"], p["c0"]

    # 初期場
    c = generate_initial_field(N, c0)
    k = 2 * np.pi * fftfreq(N, dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2, K4 = KX**2 + KY**2, (KX**2 + KY**2) ** 2

    # エネルギー記録配列
    F_pot = np.zeros(steps)
    F_int = np.zeros(steps)
    F_tot = np.zeros(steps)

    for n in range(steps):
        if stop_evt.is_set():
            break

        # Cahn–Hilliard 更新
        f_prime = 2 * c * (1 - 3 * c + 2 * c**2)
        c_k = fft2(c)
        f_k = fft2(f_prime)
        c_k = (c_k - dt * M * K2 * f_k) / (1 + 2 * dt * M * kappa * K4)
        c = np.real(ifft2(c_k))

        # エネルギー計算
        ep, ei, et = compute_free_energy(c, dx, kappa)
        F_pot[n] = ep
        F_int[n] = ei
        F_tot[n] = et

        if n % disp_int == 0 or n == steps - 1:
            # [1] 濃度場更新
            img_field.src_base64 = field_to_base64(c)
            img_field.update()
            # [2] c vs g(c) 更新
            img_cg.src_base64 = plot_gc_with_samples_to_base64(c)
            img_cg.update()
            # [3] エネルギー変化更新（常に 0～steps で横軸固定）
            img_energy.src_base64 = plot_energy_to_base64(
                F_pot[: n + 1], F_int[: n + 1], F_tot[: n + 1], steps
            )
            img_energy.update()

        time.sleep(0.001)


# --------------------------------------------------
# Flet UI
# --------------------------------------------------

DEFAULT_PARAMS: Dict = {
    "N": 256,
    "dx": 1.0,
    "dt": 0.3,
    "steps": 6000,
    "M": 1.0,
    "kappa": 0.2,
    "display_interval": 10,
    "c0": 0.5,
}
DISPLAY_PARAMS = ["M", "kappa", "c0"]


def main(page: ft.Page):
    page.title = "Cahn–Hilliard Simulator (3-panel)"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # ３つの Image コントロールを横並び
    img_field = ft.Image(width=500, height=500, border_radius=4)
    img_cg = ft.Image(width=500, height=500, border_radius=4)
    img_energy = ft.Image(width=500, height=500, border_radius=4)

    # パラメータ入力
    fields: Dict[str, ft.TextField] = {}
    for key, val in DEFAULT_PARAMS.items():
        fields[key] = ft.TextField(
            label=key,
            value=str(val),
            width=100,
            dense=True,
            keyboard_type=ft.KeyboardType.NUMBER,
            visible=(key in DISPLAY_PARAMS),
        )
    apply_btn = ft.ElevatedButton(text="Apply / Restart", icon=ft.Icons.PLAY_ARROW)
    controls = ft.Column(
        [
            ft.Text(""),
            ft.Text(""),
            ft.Text(""),
            ft.Row(list(fields.values()) + [apply_btn], wrap=True),
            ft.Text("M: mobility, kappa: interface energy, c0: initial concentration"),
        ]
    )

    page.add(
        controls,
        ft.Row(
            [img_field, img_cg, img_energy],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        ),
    )

    # シミュレーション開始
    stop_event = threading.Event()
    sim_params = DEFAULT_PARAMS.copy()
    page.run_thread(
        simulation_worker, page, img_field, img_cg, img_energy, sim_params, stop_event
    )

    def apply_clicked(e):
        nonlocal stop_event, sim_params
        stop_event.set()
        new_p: Dict = {}
        for k, tf in fields.items():
            try:
                new_p[k] = (
                    int(tf.value)
                    if k in ("N", "steps", "display_interval")
                    else float(tf.value)
                )
                tf.error_text = None
            except ValueError:
                tf.error_text = "Invalid"
                tf.update()
                return
        for k in DEFAULT_PARAMS:
            if k not in new_p:
                new_p[k] = sim_params.get(k, DEFAULT_PARAMS[k])
        sim_params = new_p
        stop_event = threading.Event()
        page.run_thread(
            simulation_worker,
            page,
            img_field,
            img_cg,
            img_energy,
            sim_params,
            stop_event,
        )

    apply_btn.on_click = apply_clicked


if __name__ == "__main__":
    ft.app(target=main)
