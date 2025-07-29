# type: ignore
import threading
import time
import io
import base64
from typing import Dict

import numpy as np
import matplotlib

# Use a non‑GUI backend because the simulation runs in a background thread
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

# --- external deps ---
# NOTE: the algorithm requires SciPy for sparse matrices. Make sure it is installed.
import scipy.sparse as sp
import flet as ft

# --------------------------------------------------
#   Utility functions
# --------------------------------------------------


def field_to_base64(field: np.ndarray) -> str:
    """Render field (phi) to PNG -> base64 string for Flet Image.src_base64."""
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(field, cmap="viridis", origin="lower", vmax=1, vmin=0)
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------
#   Phase‑Field (Solidification) helper routines
# --------------------------------------------------


def nucleus(Nx: int, Ny: int, seed: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a circular solid nucleus of radius sqrt(seed) centred in the domain."""
    phi = np.zeros((Nx, Ny), dtype=float)
    tempr = np.zeros((Nx, Ny), dtype=float)

    cx, cy = Nx / 2, Ny / 2
    r2 = seed  # actually radius^2 in the original reference implementation
    for i in range(Nx):
        dx_i2 = (i + 1 - cx) ** 2
        for j in range(Ny):
            dy_j2 = (j + 1 - cy) ** 2
            if dx_i2 + dy_j2 < r2:
                phi[i, j] = 1.0
    return phi, tempr


def build_laplacian(Nx: int, Ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """Return 2‑D Laplacian with periodic BCs as a sparse CSR matrix."""
    # 1‑D second‑difference (non‑periodic)
    e = np.ones(Nx, dtype=float)
    diagonals = [2 * e, -1 * e[:-1], -1 * e[:-1]]
    offsets = [0, -1, 1]
    T = sp.diags(diagonals, offsets, shape=(Nx, Nx), format="lil")
    I = sp.eye(Nx, format="lil")
    L = -(sp.kron(T, I, format="lil") + sp.kron(I, T, format="lil"))

    # manually stitch periodic boundaries
    for j in range(Ny):
        k_left = 0 + j * Nx
        k_right = (Nx - 1) + j * Nx
        L[k_left, k_right] = 1.0
        L[k_right, k_left] = 1.0
    for i in range(Nx):
        k_bottom = i + 0 * Nx
        k_top = i + (Ny - 1) * Nx
        L[k_bottom, k_top] = 1.0
        L[k_top, k_bottom] = 1.0
    L = L.tocsr()
    return L / (dx * dy)


def vec2matx(V: np.ndarray, Nx: int) -> np.ndarray:
    """Reshape a flattened (row‑major) vector back into (Nx, Ny)."""
    Ny = V.size // Nx
    return V.reshape((Nx, Ny), order="C")


def gradient_mat(
    matx: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return x‑ and y‑ gradients with periodic BC (ndarray, ndarray)."""
    # np.gradient returns (d/dy, d/dx) by default
    drow, dcol = np.gradient(matx, edge_order=2)
    return dcol / dx, drow / dy


# --------------------------------------------------
#   Simulation worker (runs in background thread)
# --------------------------------------------------


def simulation_worker(
    page: ft.Page, img_ctl: ft.Image, params: Dict, stop_evt: threading.Event
):
    """Run the phase‑field solidification simulation."""
    p = params.copy()

    # extract parameters
    N: int = p["N"]
    Nx = Ny = N
    dx = p["dx"]
    dy = p["dy"]
    nstep = p["steps"]
    dtime = p["dtime"]
    nprint = p["display_interval"]
    # material parameters
    tau = p["tau"]
    epsilonb = p["epsilonb"]
    mu = p["mu"]
    kappa = p["kappa"]
    delta = p["delta"]
    aniso = p["aniso"]
    alpha = p["alpha"]
    gamma = p["gamma"]
    teq = p["teq"]
    theta0 = p["theta0"]
    seed = p["seed"]
    pix = np.pi  # constant

    # initialise field variables
    phi, tempr = nucleus(Nx, Ny, seed)
    laplacian = build_laplacian(Nx, Ny, dx, dy)

    for istep in range(1, nstep + 1):
        if stop_evt.is_set():
            break

        phiold = phi.copy()

        # laplacians
        lap_phi = vec2matx(laplacian.dot(phi.flatten(order="C")), Nx)
        lap_tempr = vec2matx(laplacian.dot(tempr.flatten(order="C")), Nx)

        # gradients
        phidy, phidx = gradient_mat(phi, dx, dy)

        theta = np.arctan2(phidy, phidx)

        epsilon = epsilonb * (1.0 + delta * np.cos(aniso * (-theta - theta0)))
        epsilon_deriv = -epsilonb * aniso * delta * np.sin(aniso * (theta - theta0))

        # term1 and term2 constitute the divergence of (epsilon^2 * grad(phi)) with anisotropy
        term1, _ = gradient_mat(epsilon * epsilon_deriv * phidx, dx, dy)
        _, term2 = gradient_mat(-epsilon * epsilon_deriv * phidy, dx, dy)

        m = (alpha / pix) * np.arctan(gamma * (teq - tempr))

        # update phi
        phi = phi + (dtime / tau) * (
            term1
            + term2
            + (epsilon**2) * lap_phi
            + phiold * (1.0 - phiold) * (phiold - 0.5 + m)
        )
        # update temperature
        tempr = tempr + dtime * lap_tempr + kappa * (phi - phiold)

        # visualise every nprint steps or last step
        if istep % nprint == 0 or istep == nstep:
            img_ctl.src_base64 = field_to_base64(phi)
            img_ctl.update()

        time.sleep(0.001)  # keep UI responsive


# --------------------------------------------------
#   Flet UI
# --------------------------------------------------

DEFAULT_PARAMS: Dict = {
    "N": 256,
    "dx": 0.03,
    "dy": 0.03,
    "steps": 3000,
    "dtime": 1e-4,
    "display_interval": 50,
    # material params
    "tau": 0.0003,
    "epsilonb": 0.01,
    "mu": 1.0,
    "kappa": 1.8,
    "delta": 0.02,
    "aniso": 6.0,
    "alpha": 0.9,
    "gamma": 10.0,
    "teq": 1.0,
    "theta0": 0.0,
    "seed": 5.0,
}


# control bar shows only a few intuitive parameters (others are hidden but preserved)
DISPLAY_PARAMS = [
    "kappa",
    "aniso",
    "gamma",
]


def main(page: ft.Page):
    page.title = "Phase‑Field Solidification Simulator"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    img = ft.Image(width=1024, height=1024, border_radius=ft.border_radius.all(4))

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
            ft.Text("kappa: 相転移のときの潜熱, aniso: 異方性, gamma: 過飽和度"),
        ]
    )
    page.add(row, img)

    # initial simulation
    stop_event = threading.Event()
    sim_params = DEFAULT_PARAMS.copy()
    page.run_thread(simulation_worker, page, img, sim_params, stop_event)

    # --------------------------------------------------
    #   Handlers
    # --------------------------------------------------
    def apply_clicked(e):
        nonlocal stop_event, sim_params
        stop_event.set()

        # parse UI fields
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

        # fill hidden params
        for k in DEFAULT_PARAMS.keys():
            if k not in new_params:
                new_params[k] = sim_params.get(k, DEFAULT_PARAMS[k])

        sim_params = new_params

        # restart thread
        stop_event = threading.Event()
        page.run_thread(simulation_worker, page, img, sim_params, stop_event)

    apply_btn.on_click = apply_clicked


if __name__ == "__main__":
    ft.app(target=main)
