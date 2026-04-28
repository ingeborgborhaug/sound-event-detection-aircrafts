import numpy as np
import matplotlib.pyplot as plt

"Figure inspired by Figure 5.1 and 7.7. in Physical Geodesy (see notebook lm or zotero for link to book)"

plt.rcParams["font.family"] = "Times New Roman"


def topography(x: np.ndarray) -> np.ndarray:
    """Stylized terrain profile."""
    return (
        1.55
        + 0.18 * np.sin(0.9 * x - 0.8)
        + 0.07 * np.sin(2.6 * x + 0.4)
        + 0.03 * np.sin(5.2 * x - 0.7)
    )


def geoid(x: np.ndarray) -> np.ndarray:
    """Smooth geoid profile."""
    return 0.92 + 0.05 * np.sin(0.45 * x + 0.6) + 0.015 * np.sin(1.4 * x)


def ellipsoid(x: np.ndarray) -> np.ndarray:
    """Reference ellipsoid profile: small curved arc of a very large ellipsoid."""
    return 0.78 - 0.008 * x - 0.0006 * (x - 5.0) ** 2


def main() -> None:
    x = np.linspace(0, 10, 1200)

    y_topo = topography(x)
    y_geoid = geoid(x)
    y_ellipsoid = ellipsoid(x)

    # Point where the three heights are illustrated
    x0 = 4.0
    yt = np.interp(x0, x, y_topo)
    yg = np.interp(x0, x, y_geoid)
    ye = np.interp(x0, x, y_ellipsoid)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Fill terrain
    ax.fill_between(x, y_ellipsoid - 0.02, y_topo, color="#b8d3b2", zorder=1)

    # Curves
    ax.plot(x, y_topo, color="green", lw=1.1, zorder=3)
    ax.plot(x, y_geoid, color="blue", lw=1.5, zorder=4)
    ax.plot(x, y_ellipsoid, color="black", lw=1.0, zorder=4)

    # Vertical reference line through the chosen point (exact intersections)
    # ax.plot([x0, x0], [ye, yt], color="black", lw=1.0, zorder=5)

    # Orthometric height H: from geoid to terrain
    x_H = x0 - 0.22
    yg_H = np.interp(x_H, x, y_geoid)
    yt_H = np.interp(x_H, x, y_topo)
    ax.annotate(
        "",
        xy=(x_H, yt_H),
        xytext=(x_H, yg_H),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="blue", shrinkA=0, shrinkB=0),
        zorder=6,
    )
    ax.text(x_H, yt_H + 0.05, r"$H$", fontsize=14, ha="center", va="bottom", color="blue")

    # Ellipsoidal height h: from ellipsoid to terrain
    x_h = x0 + 0.20
    ye_h = np.interp(x_h, x, y_ellipsoid)
    yt_h = np.interp(x_h, x, y_topo)
    ax.annotate(
        "",
        xy=(x_h, yt_h),
        xytext=(x_h, ye_h),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="black", shrinkA=0, shrinkB=0),
        zorder=6,
    )
    ax.text(x_h, yt_h + 0.05, r"$h$", fontsize=14, ha="center", va="bottom", color="black")

    # Geoid height N: from ellipsoid to geoid
    x_N = 6.2
    yg2 = np.interp(x_N, x, y_geoid)
    ye2 = np.interp(x_N, x, y_ellipsoid)

    ax.annotate(
        "",
        xy=(x_N, yg2),
        xytext=(x_N, ye2),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="blue"),
        zorder=6,
    )
    ax.text(x_N, ye2 - 0.04, r"$N$", fontsize=14, ha="center", va="top", color="blue")

    # Labels
    ax.text(0.9, 2.0, "Topography", fontsize=14)
    ax.text(0.4, 1.08, "Geoid", fontsize=14, color="blue")
    ax.text(4.35, 0.42, "Reference ellipsoid", fontsize=13, ha="center")

    # Dotted leader lines from labels to corresponding curves
    x_topo_lead = 1.8
    y_topo_lead = np.interp(x_topo_lead, x, y_topo)
    ax.plot([1.55, x_topo_lead], [1.94, y_topo_lead], ls=":", lw=1.0, color="green", zorder=5)

    x_geoid_lead = 1.2
    y_geoid_lead = np.interp(x_geoid_lead, x, y_geoid)
    ax.plot([0.95, x_geoid_lead], [1.06, y_geoid_lead], ls=":", lw=1.0, color="blue", zorder=5)

    x_ellipsoid_lead = 4.6
    y_ellipsoid_lead = np.interp(x_ellipsoid_lead, x, y_ellipsoid)
    ax.plot([4.35, x_ellipsoid_lead], [0.50, y_ellipsoid_lead], ls=":", lw=1.0, color="black", zorder=5)

    # Clean look
    ax.set_xlim(0, 10)
    ax.set_ylim(0.34, 2.06)
    ax.axis("off")

    # Vector PDF output (save only)
    plt.savefig("geoid_heights.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()