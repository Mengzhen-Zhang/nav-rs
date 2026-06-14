#!/usr/bin/env python3
"""Regenerate the three README figures for nav-attitude.

Two are plotted from numbers printed by the crate's own tests (run with
`-- --nocapture`); the third is computed here with scipy, because no test
currently emits the rotation-vector/BCH curve.

    naive_log_degrades_near_pi_quat_path_does_not  -> log_near_pi.png
    conditioning_sweep                             -> euler_conditioning.png
    (computed: BCH residual vs angle)              -> bch_residual.png

Usage:  python3 docs/make_plots.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. log near theta = pi: quaternion-routed log vs the naive trace formula.
#    Numbers copied verbatim from `cargo test naive_log_degrades_near_pi_
#    quat_path_does_not -- --nocapture`.
# ---------------------------------------------------------------------------
def plot_log_near_pi():
    gap = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])  # pi - theta
    quat = np.array([2.483e-16, 4.965e-16, 5.088e-16, 5.088e-16,
                     2.483e-16, 5.088e-16, 2.483e-16])
    naive = np.array([8.244e-9, 7.106e-6, 1.256e-3, 3.309e-2,
                      2.565e8, 2.565e7, 2.565e6])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.loglog(gap, naive, "o-", color="#c0392b", label="naive trace/sin log")
    ax.loglog(gap, quat, "o-", color="#2471a3",
              label="quaternion-routed log (Shepperd)")
    ax.axhline(2.2e-16, ls=":", color="gray", lw=1, label="f64 epsilon")
    ax.invert_xaxis()  # approach pi as we move right
    ax.set_xlabel(r"$\pi - \theta$  (closer to $\pi$ $\rightarrow$)")
    ax.set_ylabel("error in recovered rotation vector  (rad)")
    ax.set_title(r"log stability near $\theta=\pi$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "log_near_pi.png")


# ---------------------------------------------------------------------------
# 2. Euler 321 conditioning sweep toward gimbal lock. Numbers from
#    `cargo test conditioning_sweep -- --nocapture`.
# ---------------------------------------------------------------------------
def plot_conditioning():
    gap = 10.0 ** -np.arange(1, 13)  # pi/2 - theta
    err = np.array([1.776e-15, 9.770e-15, 1.188e-13, 1.372e-12, 1.478e-11,
                    7.094e-11, 1.157e-9, 1.000e-8, 5.205e-8,
                    3.000e-1, 3.000e-1, 3.000e-1])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.loglog(gap, err, "o-", color="#8e44ad",
              label="worst Euler-angle error after round trip")
    # 1/cos(theta) ~ 1/(pi/2 - theta) reference slope, anchored to the run.
    ref = err[2] * (gap[2] / gap)
    ax.loglog(gap, ref, ls="--", color="gray", lw=1,
              label=r"$\propto 1/\cos\theta$ reference")
    ax.axvline(1e-9, ls=":", color="#c0392b", lw=1,
               label="gimbal-lock gate (1e-9)")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\pi/2 - \theta$  (toward gimbal lock $\rightarrow$)")
    ax.set_ylabel("angle error  (rad)")
    ax.set_title("Euler-321 angle conditioning: rotation exact, angles not")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "euler_conditioning.png")


# ---------------------------------------------------------------------------
# 3. Rotation vectors do not add. Compose exp(a) . exp(b) for orthogonal
#    generators a = theta*x, b = theta*y, and compare the true log against
#    naive summation (a+b) and second-order BCH (a+b+0.5 a x b).
#    Computed with scipy so the curve matches the crate's `bch_second_order`
#    and `rotation_vectors_do_not_add` tests at their sample points.
# ---------------------------------------------------------------------------
def plot_bch():
    from scipy.spatial.transform import Rotation as Rot

    thetas = np.logspace(-3, 0, 50)  # per-axis angle, up to 1 rad
    naive_err, bch_err = [], []
    for t in thetas:
        a = np.array([t, 0.0, 0.0])
        b = np.array([0.0, t, 0.0])
        # scipy p*q applies q first, matching compose(exp(a), exp(b)) = Ra.Rb
        z = (Rot.from_rotvec(a) * Rot.from_rotvec(b)).as_rotvec()
        naive_err.append(np.linalg.norm(z - (a + b)))
        bch_err.append(np.linalg.norm(z - (a + b + 0.5 * np.cross(a, b))))
    naive_err = np.array(naive_err)
    bch_err = np.array(bch_err)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.loglog(thetas, naive_err, "o-", color="#c0392b",
              label=r"naive sum  $a+b$   (slope 2: the $\frac{1}{2} a\times b$ term)")
    ax.loglog(thetas, bch_err, "o-", color="#2471a3",
              label=r"2nd-order BCH  $a+b+\frac{1}{2} a\times b$   (slope 3)")
    ax.loglog(thetas, 0.5 * thetas ** 2, ls="--", color="gray", lw=1,
              label=r"$\frac{1}{2}\theta^2$ reference")
    ax.set_xlabel(r"generator angle  $\|a\|=\|b\|=\theta$  (rad)")
    ax.set_ylabel(r"error vs true $\log(e^a e^b)$  (rad)")
    ax.set_title("Rotation vectors do not add; the bracket is the correction")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "bch_residual.png")


if __name__ == "__main__":
    plot_log_near_pi()
    plot_conditioning()
    plot_bch()
