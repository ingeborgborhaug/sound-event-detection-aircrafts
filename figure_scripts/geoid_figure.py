import numpy as np
import matplotlib.pyplot as plt


def smooth_topography(x: np.ndarray) -> np.ndarray:
    """Create a smooth stylized terrain profile."""
    return (
        1.55
        + 0.18 * np.sin(0.9 * x - 0.7)
        + 0.08 * np.sin(2.6 * x + 0.8)
        + 0.04 * np.sin(5.0 * x - 1.3)
    )


def geoid_curve(x: np.ndarray) -> np.ndarray:
    """Stylized geoid profile."""
    return 1.15 + 0.11 * np.sin(0.55 * x - 0.4) + 0.03 * np.sin(1.5 * x + 0.2)


def ellipsoid_curve(x: np.ndarray) -> np.ndarray:
    """Stylized reference ellipsoid profile."""
    return 0.78 + 0.07 * np.sin(0.42 * x - 0.9) - 0.02 * np.sin(1.1 * x)


def main() -> None:
    x = np.linspace(0, 10, 1000)

    topo = smooth_topography(x)
    geoid = geoid_curve(x)
    ellipsoid = ellipsoid_curve(x)

    # Point where the geoid height N is shown
    x0 = 5.8
    y_geoid = np.interp(x0, x, geoid)
    y_ellipsoid = np.interp(x0, x, ellipsoid)
    y_topo = np.interp(x0, x, topo)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Ground / topography fill
    ax.fill_between(x, 0, topo, color="#b9d7a8", zorder=1)

    # Light blue fill between geoid and ellipsoid
    ax.fill_between(x, ellipsoid, geoid, where=geoid >= ellipsoid, color="#dcecf7", zorder=2)

    # Curves
    ax.plot(x, topo, color="#2ca02c", lw=1.8, zorder=3)
    ax.plot(x, geoid, color="#1f4ed8", lw=2.0, zorder=4)
    ax.plot(x, ellipsoid, color="black", lw=1.5, zorder=4)

    # Chosen point on geoid
    ax.plot(x0, y_geoid, "o", color="red", ms=5, zorder=6, mec="black", mew=0.6)

    # Vertical line showing geoid height N
    ax.plot([x0, x0], [y_ellipsoid, y_geoid], color="black", lw=1.1, zorder=5)

    # Arrow for N
    ax.annotate(
        "",
        xy=(x0 + 0.12, y_geoid),
        xytext=(x0 + 0.12, y_ellipsoid),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="black"),
        zorder=6,
    )
    ax.text(x0 + 0.18, 0.5 * (y_geoid + y_ellipsoid), "Geoid height $N$", va="center", fontsize=12)

    # Plumb-line deflection (dashed blue tilted line)
    dx = 0.18
    ax.plot(
        [x0 - dx, x0 + dx],
        [y_topo + 0.75, y_ellipsoid - 0.12],
        ls=(0, (4, 3)),
        color="#2563eb",
        lw=1.4,
        zorder=5,
    )

    # Small dotted arc near the top of dashed line
    arc_theta = np.linspace(np.deg2rad(210), np.deg2rad(310), 80)
    arc_r = 0.55
    arc_xc = x0 - 1.1
    arc_yc = y_topo + 0.35
    arc_x = arc_xc + arc_r * np.cos(arc_theta)
    arc_y = arc_yc + arc_r * np.sin(arc_theta)
    ax.plot(arc_x, arc_y, color="0.4", ls=":", lw=1.2, zorder=5)

    ax.annotate(
        "",
        xy=(arc_x[-1], arc_y[-1]),
        xytext=(arc_x[-8], arc_y[-8]),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.4"),
        zorder=6,
    )

    # Labels
    ax.text(0.45, 2.62, "Plumb-line\ndeflections $(\\xi, \\eta)$", fontsize=13)
    ax.text(7.2, 2.15, "Topography", fontsize=13)
    ax.text(1.0, 1.18, "Geoid", color="#1f4ed8", fontsize=13)
    ax.text(6.1, 0.88, "Reference ellipsoid", fontsize=13, rotation=3)

    # Small right-angle marker near point
    s = 0.08
    ax.plot([x0, x0 + s], [y_geoid, y_geoid], color="black", lw=1.0, zorder=6)
    ax.plot([x0 + s, x0 + s], [y_geoid, y_geoid - s], color="black", lw=1.0, zorder=6)

    # Style
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 2.9)
    ax.axis("off")

    # Save as vector PDF
    plt.savefig("geoid_model_figure.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()