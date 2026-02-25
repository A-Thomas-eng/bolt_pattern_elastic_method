"""
Bolt Pattern Force Distribution
================================
Implements the methodology from:
https://mechanicalc.com/reference/bolt-pattern-force-distribution

Given a bolt pattern (positions + areas) and applied loads at arbitrary
locations, this module computes the axial and shear force on each bolt.

Coordinate system
-----------------
  X, Y  - in-plane axes (the plane of the bolt pattern)
  Z      - out-of-plane axis (bolt axis direction)

Applied load / moment sign convention follows the right-hand rule.

Usage example at the bottom of this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – registers 3-D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_bolt_counter = 0

@dataclass
class Bolt:
    """A single bolt in the pattern.

    Parameters
    ----------
    x, y : float
        In-plane position of the bolt.
    area : float
        Tensile stress area of the bolt (default 1.0 for equal-size bolts).
    label : str
        Human-readable name. Auto-assigned as B1, B2, ... if not provided.
    """
    x: float
    y: float
    area: float = 1.0
    label: str = ""

    def __post_init__(self):
        global _bolt_counter
        if not self.label:
            _bolt_counter += 1
            self.label = f"B{_bolt_counter}"


@dataclass
class AppliedLoad:
    """A force/moment applied at a specific point.

    Parameters
    ----------
    Fx, Fy, Fz : float
        Force components (Fz is axial / out-of-plane).
    Mx, My, Mz : float
        Moment components about each axis (right-hand rule).
    x, y, z : float
        Location at which the load is applied.
        The moments are transferred to the pattern centroid automatically.
    """
    Fx: float = 0.0
    Fy: float = 0.0
    Fz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class BoltResult:
    """Forces resolved onto a single bolt."""
    bolt: Bolt
    # Axial (Z-direction) components
    Fz_direct: float = 0.0      # due to direct Fz at centroid
    Fz_Mcx: float = 0.0         # due to centroidal moment about X
    Fz_Mcy: float = 0.0         # due to centroidal moment about Y
    # Shear components
    Fxy_direct_x: float = 0.0   # direct Fx distributed by area
    Fxy_direct_y: float = 0.0   # direct Fy distributed by area
    Fxy_Mcz_x: float = 0.0      # torsional moment – X component
    Fxy_Mcz_y: float = 0.0      # torsional moment – Y component

    @property
    def Fz_total(self) -> float:
        """Total axial force on this bolt."""
        return self.Fz_direct + self.Fz_Mcx + self.Fz_Mcy

    @property
    def Fx_total(self) -> float:
        return self.Fxy_direct_x + self.Fxy_Mcz_x

    @property
    def Fy_total(self) -> float:
        return self.Fxy_direct_y + self.Fxy_Mcz_y

    @property
    def F_shear(self) -> float:
        """Resultant shear magnitude."""
        return math.hypot(self.Fx_total, self.Fy_total)

    @property
    def F_total(self) -> float:
        """Total resultant force magnitude."""
        return math.sqrt(self.Fx_total**2 + self.Fy_total**2 + self.Fz_total**2)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

@dataclass
class BoltPatternAnalysis:
    """Analyse force distribution over a bolt pattern.

    Parameters
    ----------
    bolts : list[Bolt]
        The bolt pattern.
    loads : list[AppliedLoad]
        All applied loads (can be at arbitrary locations).
    """
    bolts: list[Bolt]
    loads: list[AppliedLoad] = field(default_factory=list)

    # ---- Pattern properties (computed lazily) ----

    @property
    def total_area(self) -> float:
        """Sum of bolt areas."""
        return sum(b.area for b in self.bolts)

    @property
    def centroid(self) -> tuple[float, float]:
        """(xc, yc) – area-weighted centroid of the bolt pattern."""
        A = self.total_area
        xc = sum(b.area * b.x for b in self.bolts) / A
        yc = sum(b.area * b.y for b in self.bolts) / A
        return xc, yc

    @property
    def Ic_x(self) -> float:
        """Second moment of area about the centroidal X-axis (bending about X)."""
        xc, yc = self.centroid
        return sum(b.area * (b.y - yc)**2 for b in self.bolts)

    @property
    def Ic_y(self) -> float:
        """Second moment of area about the centroidal Y-axis (bending about Y)."""
        xc, yc = self.centroid
        return sum(b.area * (b.x - xc)**2 for b in self.bolts)

    @property
    def Ic_p(self) -> float:
        """Polar moment of area about the centroid (torsion about Z)."""
        xc, yc = self.centroid
        return sum(b.area * ((b.x - xc)**2 + (b.y - yc)**2) for b in self.bolts)

    # ---- Translate all loads to the centroid ----

    def centroidal_loads(self) -> tuple[float, float, float, float, float, float]:
        """Return (Fc_x, Fc_y, Fc_z, Mc_x, Mc_y, Mc_z) at the pattern centroid.

        Each applied load is moved to the centroid using the cross-product
        transfer rule:  M_new = M_applied + R × F
        where R is the vector from the centroid to the load application point.
        """
        xc, yc = self.centroid

        Fc_x = Fc_y = Fc_z = 0.0
        Mc_x = Mc_y = Mc_z = 0.0

        for load in self.loads:
            # Direct force sums
            Fc_x += load.Fx
            Fc_y += load.Fy
            Fc_z += load.Fz

            # Moment transfer: R = (load.x - xc, load.y - yc, load.z - 0)
            rx = load.x - xc
            ry = load.y - yc
            rz = load.z  # centroid is at z=0 by convention

            # Cross product  R × F  (right-hand rule)
            cross_x = ry * load.Fz - rz * load.Fy
            cross_y = rz * load.Fx - rx * load.Fz
            cross_z = rx * load.Fy - ry * load.Fx

            Mc_x += load.Mx + cross_x
            Mc_y += load.My + cross_y
            Mc_z += load.Mz + cross_z

        return Fc_x, Fc_y, Fc_z, Mc_x, Mc_y, Mc_z

    # ---- Distribute to individual bolts ----

    def solve(self) -> list[BoltResult]:
        """Compute and return the force on each bolt."""
        xc, yc = self.centroid
        A = self.total_area
        Ic_x = self.Ic_x
        Ic_y = self.Ic_y
        Ic_p = self.Ic_p

        Fc_x, Fc_y, Fc_z, Mc_x, Mc_y, Mc_z = self.centroidal_loads()

        results = []
        for b in self.bolts:
            r = BoltResult(bolt=b)

            # Distance of bolt from centroid
            dx = b.x - xc   # rc_y component (distance in X from centroid)
            dy = b.y - yc   # rc_x component (distance in Y from centroid)

            # ------------------------------------------------------------------
            # Axial forces  (Z direction)
            # ------------------------------------------------------------------
            # 1. Direct Fz distributed proportional to bolt area
            r.Fz_direct = (b.area / A) * Fc_z

            # 2. Moment Mc_x about X-axis → tensile/compressive along Z
            #    F = (Mc_x * rc_x,i / Ic_x) * A_i
            #    rc_x,i = dy  (distance from centroid in Y direction)
            if abs(Ic_x) > 0:
                r.Fz_Mcx = (Mc_x * dy / Ic_x) * b.area
            else:
                r.Fz_Mcx = 0.0

            # 3. Moment Mc_y about Y-axis → tensile/compressive along Z
            #    rc_y,i = dx  (distance from centroid in X direction)
            #    Note: sign convention – positive Mc_y acts in -Z for bolts at +X
            if abs(Ic_y) > 0:
                r.Fz_Mcy = -(Mc_y * dx / Ic_y) * b.area
            else:
                r.Fz_Mcy = 0.0

            # ------------------------------------------------------------------
            # Shear forces  (X-Y plane)
            # ------------------------------------------------------------------
            # 1. Direct shear distributed proportional to bolt area
            r.Fxy_direct_x = (b.area / A) * Fc_x
            r.Fxy_direct_y = (b.area / A) * Fc_y

            # 2. Torsional moment Mc_z – shear perpendicular to radius vector
            #    F_i = (Mc_z * r_i / Ic_p) * A_i  (magnitude)
            #    Direction: perpendicular to (dx, dy), i.e. (-dy, dx) normalised
            if abs(Ic_p) > 0:
                scale = (Mc_z / Ic_p) * b.area
                r.Fxy_Mcz_x = -scale * dy
                r.Fxy_Mcz_y =  scale * dx
            else:
                r.Fxy_Mcz_x = 0.0
                r.Fxy_Mcz_y = 0.0

            results.append(r)

        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def bolt_pattern_force_distribution(
    bolt_positions: Sequence[tuple[float, float]],
    applied_loads: Sequence[dict],
    bolt_areas: Sequence[float] | None = None,
    bolt_labels: Sequence[str] | None = None,
) -> list[BoltResult]:
    """High-level helper function.

    Parameters
    ----------
    bolt_positions : list of (x, y)
        In-plane coordinates of each bolt.
    applied_loads : list of dicts
        Each dict may contain keys: Fx, Fy, Fz, Mx, My, Mz, x, y, z.
        Missing keys default to 0.
    bolt_areas : list of float, optional
        Tensile stress area of each bolt. Defaults to 1.0 for all bolts
        (equal-size bolts, which simplifies area-weighted sums).
    bolt_labels : list of str, optional
        Human-readable names for the bolts.

    Returns
    -------
    list[BoltResult]
        One entry per bolt with decomposed and total forces.
    """
    n = len(bolt_positions)
    if bolt_areas is None:
        bolt_areas = [1.0] * n
    if bolt_labels is None:
        bolt_labels = [f"Bolt {i+1}" for i in range(n)]

    bolts = [
        Bolt(x=pos[0], y=pos[1], area=a, label=lbl)
        for pos, a, lbl in zip(bolt_positions, bolt_areas, bolt_labels)
    ]

    loads = [
        AppliedLoad(
            Fx=ld.get("Fx", 0.0),
            Fy=ld.get("Fy", 0.0),
            Fz=ld.get("Fz", 0.0),
            Mx=ld.get("Mx", 0.0),
            My=ld.get("My", 0.0),
            Mz=ld.get("Mz", 0.0),
            x=ld.get("x", 0.0),
            y=ld.get("y", 0.0),
            z=ld.get("z", 0.0),
        )
        for ld in applied_loads
    ]

    analysis = BoltPatternAnalysis(bolts=bolts, loads=loads)
    return analysis.solve()

# ---------------------------------------------------------------------------
# 3-D Visualisation
# ---------------------------------------------------------------------------

def plot_bolt_pattern_3d(
    analysis: BoltPatternAnalysis,
    results: list[BoltResult],
    title: str = "Bolt Pattern Force Distribution",
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a 3-D visualisation of the bolt pattern and its force distribution.

    The bolt pattern sits in the Z=0 plane.  For each bolt three arrows are drawn:

    * **Blue**  - total shear force vector (in the X-Y plane, originating at bolt)
    * **Red**   - total axial force (along Z, upward = tension, downward = compression)
    * **Green** - resultant total force vector

    The centroid is marked with a cross, and each applied load is shown as a
    magenta arrow originating from its application point.

    Parameters
    ----------
    analysis : BoltPatternAnalysis
        The analysis object (used for centroid, centroidal loads, etc.).
    results : list[BoltResult]
        Per-bolt results from ``analysis.solve()``.
    title : str
        Figure title.
    show : bool
        If True, call ``plt.show()`` at the end.
    save_path : str or None
        If given, save the figure to this path before showing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    matplotlib.rcParams.update({
        "font.family": "monospace",
        "axes.facecolor": "#0d1117",
        "figure.facecolor": "#0d1117",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "axes.edgecolor": "#30363d",
    })

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0d1117")

    # ── Main 3-D axis ────────────────────────────────────────────────────────
    ax3d = fig.add_axes([0.02, 0.08, 0.62, 0.88], projection="3d")
    ax3d.set_facecolor("#0d1117")
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#21262d")

    xc, yc = analysis.centroid
    Fc_x, Fc_y, Fc_z, Mc_x, Mc_y, Mc_z = analysis.centroidal_loads()

    # Normalisation scale so arrows are legible regardless of unit magnitudes
    all_magnitudes = [r.F_total for r in results if r.F_total > 0]
    if not all_magnitudes:
        all_magnitudes = [1.0]
    max_mag = max(all_magnitudes)

    bolt_xs = [b.bolt.x for b in results]
    bolt_ys = [b.bolt.y for b in results]
    span = max(
        max(bolt_xs) - min(bolt_xs),
        max(bolt_ys) - min(bolt_ys),
        1e-6,
    )
    arrow_scale = span * 0.45 / max_mag   # arrows reach ~45 % of the pattern span

    # ── Colour map: map |F_total| → colour ───────────────────────────────────
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=max_mag)

    # ── Draw bolt plate (semi-transparent polygon in Z=0 plane) ──────────────
    # convex hull of bolt positions
    from functools import reduce as _reduce
    import operator as _op

    pad = span * 0.18
    plate_xs = [min(bolt_xs) - pad, max(bolt_xs) + pad,
                max(bolt_xs) + pad, min(bolt_xs) - pad]
    plate_ys = [min(bolt_ys) - pad, min(bolt_ys) - pad,
                max(bolt_ys) + pad, max(bolt_ys) + pad]
    plate_zs = [0, 0, 0, 0]
    verts = [list(zip(plate_xs, plate_ys, plate_zs))]
    plate = Poly3DCollection(verts, alpha=0.12, facecolor="#58a6ff", edgecolor="#30363d", linewidth=0.8)
    ax3d.add_collection3d(plate)

    # ── Draw bolts as cylinders ───────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 32)
    bolt_r = span * 0.04

    for r in results:
        bx, by = r.bolt.x, r.bolt.y
        color = cmap(norm(r.F_total))

        # Cylinder body
        cyl_h = span * 0.08
        cx = bx + bolt_r * np.cos(theta)
        cy = by + bolt_r * np.sin(theta)
        z_top = np.full_like(theta, cyl_h)
        z_bot = np.full_like(theta, -cyl_h)
        ax3d.plot_surface(
            np.array([cx, cx]), np.array([cy, cy]),
            np.array([z_bot, z_top]),
            color=color, alpha=0.85, linewidth=0,
        )
        # Top cap
        ax3d.plot_surface(
            np.array([[bx + bolt_r * np.cos(t) for t in theta]]),
            np.array([[by + bolt_r * np.sin(t) for t in theta]]),
            np.full((1, 32), cyl_h),
            color=color, alpha=0.95, linewidth=0,
        )

        # ── Shear arrow (in-plane) ─────────────────────────────────────────
        fs = r.F_shear
        if fs > 1e-9:
            us = r.Fx_total / fs * fs * arrow_scale
            vs = r.Fy_total / fs * fs * arrow_scale
            ax3d.quiver(bx, by, cyl_h, us, vs, 0,
                        color="#388bfd", linewidth=1.8, arrow_length_ratio=0.25)

        # ── Axial arrow (Z) ────────────────────────────────────────────────
        fz = r.Fz_total
        if abs(fz) > 1e-9:
            wz = fz * arrow_scale
            ax3d.quiver(bx, by, cyl_h, 0, 0, wz,
                        color="#f85149" if fz < 0 else "#3fb950",
                        linewidth=1.8, arrow_length_ratio=0.25)

        # ── Resultant arrow ────────────────────────────────────────────────
        ft = r.F_total
        if ft > 1e-9:
            ux = r.Fx_total / ft * ft * arrow_scale
            uy = r.Fy_total / ft * ft * arrow_scale
            uz = r.Fz_total / ft * ft * arrow_scale
            ax3d.quiver(bx, by, cyl_h, ux, uy, uz,
                        color="#ffa657", linewidth=1.2,
                        arrow_length_ratio=0.2, linestyle="dashed", alpha=0.7)

        # Label
        lbl = r.bolt.label or f"({bx:.1f},{by:.1f})"
        ax3d.text(bx, by, cyl_h + span * 0.06, lbl,
                  color="#e6edf3", fontsize=7, ha="center", va="bottom",
                  fontweight="bold")

    # ── Centroid marker ───────────────────────────────────────────────────────
    ax3d.scatter([xc], [yc], [0], color="#d2a8ff", s=80, zorder=10, marker="+")
    ax3d.text(xc, yc, span * 0.05, "centroid",
              color="#d2a8ff", fontsize=7, ha="center")

    # ── Applied load arrows (magenta, from their 3-D application point) ──────
    if analysis.loads:
        app_mag = max(
            math.sqrt(l.Fx**2 + l.Fy**2 + l.Fz**2) for l in analysis.loads
        ) or 1.0
        app_scale = span * 0.35 / app_mag
        for ld in analysis.loads:
            fm = math.sqrt(ld.Fx**2 + ld.Fy**2 + ld.Fz**2)
            if fm < 1e-9:
                continue
            ax3d.quiver(ld.x, ld.y, ld.z,
                        ld.Fx * app_scale, ld.Fy * app_scale, ld.Fz * app_scale,
                        color="#bc8cff", linewidth=2.2, arrow_length_ratio=0.2)
            ax3d.scatter([ld.x], [ld.y], [ld.z], color="#bc8cff", s=40, marker="*")

    ax3d.set_xlabel("X", labelpad=4, fontsize=9, color="#8b949e")
    ax3d.set_ylabel("Y", labelpad=4, fontsize=9, color="#8b949e")
    ax3d.set_zlabel("Z  (axial)", labelpad=4, fontsize=9, color="#8b949e")
    ax3d.set_title(title, color="#e6edf3", fontsize=12, pad=10, fontweight="bold")
    ax3d.view_init(elev=28, azim=-55)
    ax3d.tick_params(colors="#8b949e", labelsize=7)

    # ── Colour bar ────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.65, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("|F_total| per bolt", color="#e6edf3", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#8b949e", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e6edf3")
    cbar.outline.set_edgecolor("#30363d")

    # ── Bar chart panel (right) ───────────────────────────────────────────────
    ax_bar = fig.add_axes([0.70, 0.55, 0.28, 0.38])
    ax_bar.set_facecolor("#161b22")
    labels   = [r.bolt.label or f"B{i+1}" for i, r in enumerate(results)]
    fz_vals  = [r.Fz_total  for r in results]
    fsh_vals = [r.F_shear   for r in results]
    ft_vals  = [r.F_total   for r in results]
    x_idx = np.arange(len(results))
    w = 0.26
    b1 = ax_bar.bar(x_idx - w, fz_vals,  width=w, label="Fz (axial)",   color="#3fb950", alpha=0.85)
    b2 = ax_bar.bar(x_idx,     fsh_vals, width=w, label="|F_shear|",    color="#388bfd", alpha=0.85)
    b3 = ax_bar.bar(x_idx + w, ft_vals,  width=w, label="|F_total|",    color="#ffa657", alpha=0.85)
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels(labels, fontsize=7, color="#e6edf3")
    ax_bar.set_ylabel("Force", fontsize=8, color="#8b949e")
    ax_bar.set_title("Per-bolt forces", fontsize=9, color="#e6edf3", fontweight="bold")
    ax_bar.tick_params(colors="#8b949e", labelsize=7)
    ax_bar.spines[:].set_edgecolor("#30363d")
    ax_bar.legend(fontsize=7, facecolor="#0d1117", edgecolor="#30363d",
                  labelcolor="#e6edf3", loc="upper left")
    ax_bar.axhline(0, color="#30363d", linewidth=0.8)

    # ── Centroidal loads table ────────────────────────────────────────────────
    ax_tbl = fig.add_axes([0.70, 0.08, 0.28, 0.40])
    ax_tbl.set_facecolor("#161b22")
    ax_tbl.axis("off")
    col_labels = ["Bolt", "Fz", "Fx", "Fy", "|Fsh|", "|Ft|"]
    rows = [
        [
            r.bolt.label or f"B{i+1}",
            f"{r.Fz_total:+.1f}",
            f"{r.Fx_total:+.1f}",
            f"{r.Fy_total:+.1f}",
            f"{r.F_shear:.1f}",
            f"{r.F_total:.1f}",
        ]
        for i, r in enumerate(results)
    ]
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d1117" if row % 2 == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#79c0ff", fontweight="bold")
    ax_tbl.set_title(
        f"Centroid ({xc:.2f}, {yc:.2f})   "
        f"Fc=({Fc_x:.1f}, {Fc_y:.1f}, {Fc_z:.1f})   "
        f"Mc=({Mc_x:.1f}, {Mc_y:.1f}, {Mc_z:.1f})",
        fontsize=7, color="#8b949e", pad=4,
    )

    # ── Legend for arrows ─────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color="#388bfd", label="Shear (in-plane)"),
        mpatches.Patch(color="#3fb950", label="Axial +Z (tension)"),
        mpatches.Patch(color="#f85149", label="Axial −Z (compression)"),
        mpatches.Patch(color="#ffa657", label="Resultant"),
        mpatches.Patch(color="#bc8cff", label="Applied load"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=5,
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#e6edf3", fontsize=8,
        bbox_to_anchor=(0.35, 0.01),
    )

    plt.suptitle(title, color="#e6edf3", fontsize=13, fontweight="bold", y=0.99)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()

    return fig

def print_results(results: list[BoltResult], analysis: BoltPatternAnalysis) -> None:
    xc, yc = analysis.centroid
    Fc_x, Fc_y, Fc_z, Mc_x, Mc_y, Mc_z = analysis.centroidal_loads()

    print("=" * 60)
    print("BOLT PATTERN PROPERTIES")
    print(f"  Total area  A    = {analysis.total_area:.4f}")
    print(f"  Centroid         = ({xc:.4f}, {yc:.4f})")
    print(f"  Ic_x             = {analysis.Ic_x:.4f}")
    print(f"  Ic_y             = {analysis.Ic_y:.4f}")
    print(f"  Ic_p (polar)     = {analysis.Ic_p:.4f}")
    print()
    print("CENTROIDAL LOADS")
    print(f"  Fc_x={Fc_x:.3f}  Fc_y={Fc_y:.3f}  Fc_z={Fc_z:.3f}")
    print(f"  Mc_x={Mc_x:.3f}  Mc_y={Mc_y:.3f}  Mc_z={Mc_z:.3f}")
    print()
    print(f"SUMMARY:")
    print(f"Max fastener tensile load: {max(r.Fz_total for r in results):.1f}.")
    print(f"Max fastener shear load: {max(r.F_shear for r in results):.1f}.")
    print()
    print("BOLT FORCES")
    header = f"{'Bolt':<10} {'Fz_total':>10} {'Fx_total':>10} {'Fy_total':>10} {'F_shear':>10} {'F_total':>10}"
    print(header)
    print("-" * 60)
    for r in results:
        label = r.bolt.label or f"({r.bolt.x},{r.bolt.y})"
        print(
            f"{label:<10} "
            f"{r.Fz_total:>10.4f} "
            f"{r.Fx_total:>10.4f} "
            f"{r.Fy_total:>10.4f} "
            f"{r.F_shear:>10.4f} "
            f"{r.F_total:>10.4f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------



# if __name__ == "__main__":
#     import os
#     HERE = os.path.dirname(os.path.abspath(__file__))

#     positions_1 = [(-3.5, 15), (3.5, 15), (-3.5, -15), (3.5, -15)
#                    , (10, -15), (18.82, -12.14), (24.27, -4.635)
#                    , (24.27, 4.635), (18.82, 12.14), (10, 15)
#                    , (-10, 15), (-18.82, 12.14), (-24.27, 4.635)
#                    , (-24.27, -4.635), (-18.82, -12.14), (-10, -15)] #[x,y]

#     # ------------------------------------------------------------------
#     # Example 1: 4-bolt rectangular pattern, in-plane eccentric shear
#     # A 1000 N force applied in X at (y=5) – creates torsion about Z
#     # ------------------------------------------------------------------
#     print("\nEXAMPLE 1 – General 3-D loading")
#     save_path="bolt_pattern_ex1.png"
#     analysis_1 = BoltPatternAnalysis(
#         bolts=[Bolt(x, y) for (x, y) in positions_1],
#         loads=[AppliedLoad(Fx=1500.0, Fy=500.0, Fz=5000.0, z=15.0),
#                AppliedLoad(Mz=1000.0)],
#     )
#     results_1 = analysis_1.solve()
#     print_results(results_1, analysis_1)
#     plot_bolt_pattern_3d(
#         analysis_1, results_1,
#         title="Example 1 - General 3-D loading",
#         show=False,
#         save_path=os.path.join(HERE, "save_path"),
#     )
#     print(f"  → saved {save_path}")


#     # ------------------------------------------------------------------
#     # Example 2: general 3-D loading using the convenience function
#     # ------------------------------------------------------------------
#     print("\nEXAMPLE 2 - General 3-D loading (convenience function)")
#     analysis_2 = BoltPatternAnalysis(
#         bolts=[Bolt(x, y, label=lbl) for (x, y), lbl in zip(
#             [(-3, -3), (3, -3), (3, 3), (-3, 3)], ["TL", "TR", "BR", "BL"]
#         )],
#         loads=[AppliedLoad(Fx=200, Fy=-150, Fz=300,
#                            Mx=500, My=-400, Mz=1000,
#                            x=1.0, y=2.0, z=50.0)],
#     )
#     results_2 = analysis_2.solve()
#     print_results(results_2, analysis_2)
#     plot_bolt_pattern_3d(
#         analysis_2, results_2,
#         title="Example 2 - General 3-D Loading",
#         show=False,
#         save_path=os.path.join(HERE, "bolt_pattern_ex2.png"),
#     )
#     print("  → saved bolt_pattern_ex2.png")