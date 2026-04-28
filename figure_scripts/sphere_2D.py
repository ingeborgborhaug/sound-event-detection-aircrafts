import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer, Geod

# ============================================================
# Thesis figure: aircraft annotation geofence (technical + clean)
# Rule: "aircraft" active in audio if 3D distance to microphone <= 15 km
# d = sqrt(d_h^2 + (Δh)^2)
# ============================================================

# -------------------------
# INPUT (edit these)
# -------------------------
# Microphone (P1) location (lon, lat, height in meters, WGS84)
P1_lon, P1_lat, P1_h = 10.814295, 63.472832, 2.6

# Geofence radius
R = 15_000.0  # meters (15 km)

# Aircraft examples (lon, lat, height in meters, WGS84)
# A: inside example (adjust as needed)
AC_A = dict(name="Aircraft A", lon=10.9200, lat=63.5200, h=1200.0)

# B: directly above microphone but too high (outside by altitude)
AC_B = dict(name="Aircraft B", lon=P1_lon, lat=P1_lat, h=16_000.0)

OUT_PDF = "annotation_geofence_3d_rule_clean.pdf"
OUT_PNG = "annotation_geofence_3d_rule_clean.png"

# -------------------------
# Geodesy helpers
# -------------------------
geod = Geod(ellps="WGS84")

def horiz_distance_m(lon1, lat1, lon2, lat2) -> float:
    _, _, d = geod.inv(lon1, lat1, lon2, lat2)
    return float(d)

def geodesic_circle_lonlat(lon0, lat0, radius_m, n=360):
    az = np.linspace(0, 360, n, endpoint=False)
    lons, lats, _ = geod.fwd(
        np.full_like(az, lon0, dtype=float),
        np.full_like(az, lat0, dtype=float),
        az,
        np.full_like(az, radius_m, dtype=float),
    )
    return lons, lats

# Local Azimuthal Equidistant projection centered at P1 (distance-preserving from P1)
crs_ll = CRS.from_epsg(4326)
crs_local = CRS.from_proj4(
    f"+proj=aeqd +lat_0={P1_lat} +lon_0={P1_lon} +datum=WGS84 +units=m +no_defs"
)
to_local = Transformer.from_crs(crs_ll, crs_local, always_xy=True).transform

def ll_to_xy(lon, lat):
    x, y = to_local(lon, lat)
    return float(x), float(y)

# -------------------------
# Compute distances and classification
# -------------------------
def classify_aircraft(ac):
    d_h = horiz_distance_m(P1_lon, P1_lat, ac["lon"], ac["lat"])
    d_v = abs(ac["h"] - P1_h)
    d_3d = np.sqrt(d_h**2 + d_v**2)
    inside = d_3d <= R
    return d_h, d_v, d_3d, inside

dA_h, dA_v, dA_3d, A_inside = classify_aircraft(AC_A)
dB_h, dB_v, dB_3d, B_inside = classify_aircraft(AC_B)

# Styling (clean + consistent)
boundary_color = "royalblue"
inside_color = "tab:green"
outside_color = "tab:red"

A_color = inside_color if A_inside else outside_color
B_color = inside_color if B_inside else outside_color

# -------------------------
# Build top-down geofence circle in local XY
# -------------------------
circle_lons, circle_lats = geodesic_circle_lonlat(P1_lon, P1_lat, R, n=360)
circle_xy = np.array([ll_to_xy(lon, lat) for lon, lat in zip(circle_lons, circle_lats)])
cx, cy = circle_xy[:, 0], circle_xy[:, 1]

# Microphone + aircraft local XY
p1x, p1y = ll_to_xy(P1_lon, P1_lat)
a_x, a_y = ll_to_xy(AC_A["lon"], AC_A["lat"])
b_x, b_y = ll_to_xy(AC_B["lon"], AC_B["lat"])

# -------------------------
# Figure: 2 panels (top-down + side view)
# -------------------------
plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(10.2, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.25)

# ============================================================
# Panel 1: Top-down (horizontal distance)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Horizontal proximity (top-down)")

# Geofence boundary
ax1.plot(cx, cy, color=boundary_color, linewidth=2.0)
ax1.fill(cx, cy, color=boundary_color, alpha=0.06)

# Microphone
ax1.scatter([p1x], [p1y], s=110, color="black", edgecolor="white", linewidth=1.4, zorder=5)
ax1.text(p1x + 1500, p1y + 1500, "P1 (microphone)", ha="left", va="bottom")

# Aircraft A
ax1.scatter([a_x], [a_y], s=110, color=A_color, edgecolor="black", linewidth=1.2, zorder=6)
ax1.plot([p1x, a_x], [p1y, a_y], linewidth=2.0, color=A_color)
ax1.text(a_x + 1500, a_y, f"{AC_A['name']}\n$d_h$={dA_h/1000:.1f} km",
         ha="left", va="center")

# Aircraft B (above microphone: same lon/lat -> same XY)
ax1.scatter([b_x], [b_y], s=110, color=B_color, edgecolor="black", linewidth=1.2, zorder=6)
ax1.plot([p1x, b_x], [p1y, b_y], linewidth=2.0, color=B_color)
ax1.text(b_x + 1500, b_y - 1500, f"{AC_B['name']}\n$d_h$={dB_h/1000:.1f} km",
         ha="left", va="top")

ax1.text(p1x, p1y + R + 800, "15 km horizontal radius", color=boundary_color,
         ha="center", va="bottom")

ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel("Local Easting (m) [AEQD @ P1]")
ax1.set_ylabel("Local Northing (m) [AEQD @ P1]")

pad = 4500
ax1.set_xlim(cx.min() - pad, cx.max() + pad)
ax1.set_ylim(cy.min() - pad, cy.max() + pad)
ax1.grid(True, alpha=0.25)

# ============================================================
# Panel 2: Side view (3D rule)
# x-axis: horizontal distance (km), y-axis: altitude (km)
# 3D sphere => circle in this cross-section
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("3D distance rule (side view)")

t = np.linspace(0, 2 * np.pi, 360)
P1_alt_km = P1_h / 1000.0

circle_x = (R/1000.0) * np.cos(t)                    # km
circle_y = P1_alt_km + (R/1000.0) * np.sin(t)        # km
ax2.plot(circle_x, circle_y, color=boundary_color, linewidth=2.0)
ax2.fill(circle_x, circle_y, color=boundary_color, alpha=0.06)

# Ground line
ax2.axhline(0, linewidth=1.2, color="gray")

# P1
ax2.scatter([0.0], [P1_alt_km], s=80, color="black", edgecolor="white", linewidth=1.2, zorder=5)
ax2.text(0.6, P1_alt_km + 0.3, "P1", ha="left", va="bottom")

# Aircraft A in cross-section
ax2.scatter([dA_h/1000.0], [AC_A["h"]/1000.0], s=90, color=A_color, edgecolor="black", linewidth=1.2, zorder=6)
ax2.plot([0.0, dA_h/1000.0], [P1_alt_km, AC_A["h"]/1000.0], linewidth=2.0, color=A_color)
ax2.text(dA_h/1000.0 + 0.8, AC_A["h"]/1000.0,
         f"{AC_A['name']}\n$d$={dA_3d/1000:.1f} km ({'active' if A_inside else 'inactive'})",
         ha="left", va="center")

# Aircraft B: above mic but too high
ax2.scatter([dB_h/1000.0], [AC_B["h"]/1000.0], s=90, color=B_color, edgecolor="black", linewidth=1.2, zorder=6)
ax2.plot([0.0, dB_h/1000.0], [P1_alt_km, AC_B["h"]/1000.0], linewidth=2.0, color=B_color)
ax2.text(dB_h/1000.0 + 0.8, AC_B["h"]/1000.0,
         f"{AC_B['name']}\n$d$={dB_3d/1000:.1f} km ({'active' if B_inside else 'inactive'})",
         ha="left", va="center")

ax2.set_xlabel("Horizontal distance $d_h$ (km)")
ax2.set_ylabel("Altitude (km)")
ax2.grid(True, alpha=0.25)
ax2.set_aspect("equal", adjustable="box")

# Limits for clean framing
ax2.set_xlim(-(R/1000.0)*1.10, (R/1000.0)*1.35)
ax2.set_ylim(0, max(AC_B["h"]/1000.0, (R/1000.0) + P1_alt_km) * 1.08)

plt.tight_layout()
fig.savefig(OUT_PDF)           # vector PDF
fig.savefig(OUT_PNG, dpi=300)  # raster fallback
plt.show()