import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

common_font_size = 10

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.titlesize": common_font_size,
    "axes.labelsize": common_font_size,
    "xtick.labelsize": common_font_size,
    "ytick.labelsize": common_font_size,
})

# ============================================================
# INPUT: Geodetic coordinates (lon, lat, height) in WGS84
# ============================================================
P1_latlong = (10.814295, 63.472832, 2.6)
P2_latlong = (10.814295, 63.472832, 10000.0)
P3_latlong = (10.633925, 63.434891, 20000.0)

R = 15_000.0  # meters

OUT_PDF = "ecef_sphere_with_ground_plane.pdf"
SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "pad_inches": 0.02,
}

# ============================================================
# Convert geodetic → ECEF
# ============================================================
geo_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
P1 = np.array(geo_to_ecef.transform(*P1_latlong), dtype=float)
P2 = np.array(geo_to_ecef.transform(*P2_latlong), dtype=float)
P3 = np.array(geo_to_ecef.transform(*P3_latlong), dtype=float)

# ============================================================
# Distance checks (ECEF Euclidean)
# ============================================================
d2 = np.linalg.norm(P2 - P1)
d3 = np.linalg.norm(P3 - P1)

inside2 = d2 <= R
inside3 = d3 <= R
status2 = "INSIDE" if inside2 else "OUTSIDE"
status3 = "INSIDE" if inside3 else "OUTSIDE"

p2_color = "tab:green" if inside2 else "tab:red"
p3_color = "tab:green" if inside3 else "tab:red"

# ============================================================
# Sphere mesh centered at P1 (ABSOLUTE ECEF)
# ============================================================
nu, nv = 70, 140
u = np.linspace(0, np.pi, nu)
v = np.linspace(0, 2*np.pi, nv)
U, V = np.meshgrid(u, v)

Xs_full = P1[0] + R * np.sin(U) * np.cos(V)
Ys_full = P1[1] + R * np.sin(U) * np.sin(V)
Zs_full = P1[2] + R * np.cos(U)

# ============================================================
# Local ground plane at P1 (tangent plane to ellipsoid)
# We'll use ENU basis vectors at (lat,lon) but the plot remains in ECEF.
# ============================================================
lon0 = np.deg2rad(P1_latlong[0])
lat0 = np.deg2rad(P1_latlong[1])

east_hat  = np.array([-np.sin(lon0),  np.cos(lon0), 0.0])
north_hat = np.array([-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)])
up_hat    = np.array([ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)])

# --- Build a square patch in the tangent plane (centered at P1)
PLANE_HALFSPAN = 30_000.0  # meters; adjust for visibility

g = np.linspace(-PLANE_HALFSPAN, PLANE_HALFSPAN, 2)
GGx, GGy = np.meshgrid(g, g)

plane = (P1.reshape(3, 1, 1)
         + east_hat.reshape(3, 1, 1)  * GGx
         + north_hat.reshape(3, 1, 1) * GGy)

Xp, Yp, Zp = plane[0], plane[1], plane[2]

# --- Intersection ring between sphere and plane (circle in the plane)
t = np.linspace(0, 2*np.pi, 400)
ring = (P1.reshape(3, 1)
        + np.outer(east_hat,  R * np.cos(t))
        + np.outer(north_hat, R * np.sin(t)))

# ============================================================
# OPTIONAL: show only "above ground" hemisphere
# (makes the annotation interpretation very clear)
# ============================================================
dot_up = (Xs_full - P1[0]) * up_hat[0] + (Ys_full - P1[1]) * up_hat[1] + (Zs_full - P1[2]) * up_hat[2]
mask = dot_up >= 0

SHOW_ONLY_ABOVE_GROUND = False
if SHOW_ONLY_ABOVE_GROUND:
    Xs = np.where(mask, Xs_full, np.nan)
    Ys = np.where(mask, Ys_full, np.nan)
    Zs = np.where(mask, Zs_full, np.nan)
else:
    Xs, Ys, Zs = Xs_full, Ys_full, Zs_full

# ============================================================
# Plot (ABSOLUTE ECEF)
# ============================================================
plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(9.0, 6.8))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
ax = fig.add_subplot(111, projection="3d")

# --- Ground plane (local Earth tangent plane)
ax.plot_surface(
    Xp, Yp, Zp,
    color="#a1a1a1",
    alpha=0.35,          # "vague plane" look
    linewidth=0,
    edgecolors="none",
    shade=False
)

# --- Sphere (or hemisphere)
ax.plot_surface(
    Xs, Ys, Zs,
    linewidth=0,
    antialiased=True,
    alpha=0.22,
    color="royalblue"
)

ax.plot_wireframe(
    Xs, Ys, Zs,
    rstride=10,   # change for more/fewer "latitude" lines
    cstride=10,  # change for more/fewer "longitude" lines
    linewidth=0.8,
    alpha=0.6
)

# --- Intersection ring
ax.plot(ring[0], ring[1], ring[2], linewidth=2.0, color="tab:gray", alpha=0.9)

# --- Connecting lines
ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], linewidth=2.5, color=p2_color)
ax.plot([P1[0], P3[0]], [P1[1], P3[1]], [P1[2], P3[2]], linewidth=2.5, color=p3_color)

# --- Points
ax.scatter(P1[0], P1[1], P1[2], s=90, color="black", label="Microphone location")
ax.scatter(P2[0], P2[1], P2[2], s=90, color=p2_color, label=f"Aircraft 1 location")
ax.scatter(P3[0], P3[1], P3[2], s=90, color=p3_color, label=f"Aircraft 2 location")

# --- Labels
ax.set_xlabel("X (m)", fontsize=common_font_size)
ax.set_ylabel("Y (m)", fontsize=common_font_size)
ax.set_zlabel("Z (m)", fontsize=common_font_size)

# Move text toward right edge of plane
text_position = (
    P1
    + 0.7 * PLANE_HALFSPAN * east_hat   # move toward right side of plane
    - 0.2 * PLANE_HALFSPAN * north_hat  # slight backward shift
    - 0.5 * R * up_hat                 # lift slightly above plane
)



info = (
    f"Aircraft 1 distance = {d2/1000:.3f} km\n"
    f"Aircraft 2 distance = {d3/1000:.3f} km\n"
)

distance_label = ax.text2D(0.10, 0.85, info, fontsize=common_font_size, transform=ax.transAxes, verticalalignment="top")

# --- Zoom around P1 (keeps sphere large)
L = max(1.35 * R, d2 * 1.05, d3 * 1.05, PLANE_HALFSPAN * 1.05)
ax.set_xlim(P1[0] - L, P1[0] + L)
ax.set_ylim(P1[1] - L, P1[1] + L)
ax.set_zlim(P1[2] - L, P1[2] + L)
ax.set_box_aspect((1, 1, 1))

# Fewer ticks
ax.xaxis.set_major_locator(MaxNLocator(3))
ax.yaxis.set_major_locator(MaxNLocator(3))
ax.zaxis.set_major_locator(MaxNLocator(3))
ax.ticklabel_format(style='plain', useOffset=False)
ax.xaxis.set_tick_params(pad=6)
ax.yaxis.set_tick_params(pad=6)
ax.zaxis.set_tick_params(pad=6)

#ax.view_init(elev=10, azim=35)
ax.grid(True)
legend_obj = ax.legend(loc="lower right", bbox_to_anchor=(0.94, 0.06), fontsize=common_font_size, framealpha=0.9)

plane_label = ax.text2D(
    0.72, 0.42,  # (x,y) in axes fraction; tune these
    "Local Earth surface\n(tangent plane)",
    transform=ax.transAxes,
    fontsize=common_font_size,
    color="black",
    rotation=0,
    ha="left",
    va="center"
)

# ---- View 1 ----
plane_label.set_position((0.70, 0.44))
distance_label.set_verticalalignment("top")

ax.view_init(elev=6, azim=-63)

# Prefer this over tight_layout for 3D stability:
# fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
fig.savefig(f"1_{OUT_PDF}", **SAVEFIG_KWARGS)

# ---- View 2 ----
plane_label.set_position((0.70, 0.53))
distance_label.set_position((0.10, 0.08))  # Move to bottom of axes
distance_label.set_verticalalignment("bottom")
#
#ax.legend(loc="lower right", fontsize=common_font_size)
ax.view_init(elev=-8, azim=40)

fig.savefig(f"2_{OUT_PDF}", **SAVEFIG_KWARGS)

plt.show()