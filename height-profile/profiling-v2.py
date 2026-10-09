import os
import time
from collections import Counter

import laspy
import numpy as np
from scipy import ndimage
from laspy import LazBackend
from google.colab import drive

drive.mount("/content/drive", force_remount=True)

file_path = "/content/drive/MyDrive/Lucas, Pascale, Dione, and Ricky VIP Project/LazFiles/"
output_dir = os.path.join(file_path, "Output")

input_file = os.path.join(output_dir, "not_ground_raw.laz")
output_file = os.path.join(output_dir, "not_ground.laz")

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"{input_file} does not exist. Run 01_ClearGround_CSF.ipynb first."
    )

###

start_time = time.time()

las = laspy.read(input_file, laz_backend=LazBackend.Lazrs)

x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)
z = np.asarray(las.z, dtype=np.float64)

print(f"Loaded {len(las.points):,} non-ground points.")

###

BASE_CELL_SIZE = 0.30
MAX_GRID_CELLS = 35_000_000
MIN_COMPONENT_POINTS = 12

CLASS_OTHER = 1
CLASS_POLE = 20
CLASS_TRASHCAN = 21
CLASS_BOLLARD = 22
CLASS_COMPACT = 23

CLASS_NAMES = {
    1: "other non-ground",
    20: "pole / long circular object",
    21: "trashcan / short cylinder",
    22: "bollard",
    23: "compact street furniture",
}

def robust_geometry(xs, ys, zs, max_sample=25000):
    n = len(xs)
    if n > max_sample:
        step = int(np.ceil(n / max_sample))
        xs, ys, zs = xs[::step], ys[::step], zs[::step]

    z_lo, z_hi = np.percentile(zs, [2, 98])
    height = float(max(0.0, z_hi - z_lo))

    xy = np.column_stack((xs, ys))
    xy0 = xy - np.median(xy, axis=0)

    if len(xy0) >= 3:
        cov = np.cov(xy0, rowvar=False)
        _, eigvecs = np.linalg.eigh(cov)
        proj = xy0 @ eigvecs
        spans = np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)
        major = float(np.max(spans))
        minor = float(np.min(spans))
    else:
        major = minor = 0.0

    roundness = float(minor / max(major, 1e-6))
    vertical_ratio = float(height / max(major, 0.05))

    return height, major, minor, roundness, vertical_ratio


def max_vertical_column_span(ids, ix, iy, zz, ny):
    keys = ix[ids].astype(np.int64) * np.int64(ny) + iy[ids].astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    zmin = np.full(len(unique_keys), np.inf, dtype=np.float64)
    zmax = np.full(len(unique_keys), -np.inf, dtype=np.float64)

    np.minimum.at(zmin, inverse, zz[ids])
    np.maximum.at(zmax, inverse, zz[ids])

    return float(np.max(zmax - zmin)) if len(unique_keys) else 0.0

###

# One continuous XY connectivity grid over the full non-ground cloud.

xmin, xmax = float(x.min()), float(x.max())
ymin, ymax = float(y.min()), float(y.max())

cell_size = BASE_CELL_SIZE

def grid_shape(cell):
    nx = int(np.floor((xmax - xmin) / cell)) + 1
    ny = int(np.floor((ymax - ymin) / cell)) + 1
    return nx, ny

nx, ny = grid_shape(cell_size)

if nx * ny > MAX_GRID_CELLS:
    cell_size *= np.sqrt((nx * ny) / MAX_GRID_CELLS)
    nx, ny = grid_shape(cell_size)
    print(f"Adjusted connectivity cell size to {cell_size:.3f} m for RAM.")

ix = np.floor((x - xmin) / cell_size).astype(np.int32)
iy = np.floor((y - ymin) / cell_size).astype(np.int32)

ix = np.clip(ix, 0, nx - 1)
iy = np.clip(iy, 0, ny - 1)

occupied = np.zeros((nx, ny), dtype=bool)
occupied[ix, iy] = True

component_grid, n_components = ndimage.label(
    occupied,
    structure=np.ones((3, 3), dtype=np.uint8),
)

point_component = component_grid[ix, iy].astype(np.int32)

component_cell_count = np.bincount(
    component_grid[occupied],
    minlength=n_components + 1,
)

print(f"Continuous connected components: {n_components:,}")
print(f"Connectivity cell size: {cell_size:.3f} m")

###

classification = np.full(len(las.points), CLASS_OTHER, dtype=np.uint8)

order = np.argsort(point_component, kind="mergesort")
labels_sorted = point_component[order]
breaks = np.flatnonzero(np.diff(labels_sorted)) + 1
starts = np.r_[0, breaks]
ends = np.r_[breaks, len(order)]

summary = Counter()

for start, end in zip(starts, ends):
    component_id = int(labels_sorted[start])
    ids = order[start:end]
    npts = len(ids)

    if component_id == 0 or npts < MIN_COMPONENT_POINTS:
        continue

    xs = x[ids]
    ys = y[ids]
    zs = z[ids]

    height, major, minor, roundness, vertical_ratio = robust_geometry(xs, ys, zs)
    occupied_area = float(component_cell_count[component_id]) * cell_size**2

    vertical_core = 0.0
    if occupied_area <= 10.0 and npts <= 250_000:
        vertical_core = max_vertical_column_span(ids, ix, iy, z, ny)

    vertical_core_fraction = vertical_core / max(height, 0.05)

    # Pole / long circular vertical object
    narrow_round_pole = (
        1.5 <= height <= 18.0
        and 0.05 <= major <= 1.50
        and roundness >= 0.25
        and vertical_ratio >= 2.0
        and occupied_area <= 3.0
    )

    # Lamp/sign attachments can widen a pole cluster, so this route
    # looks for a narrow vertical core instead of only overall width.
    pole_with_attachment = (
        1.8 <= height <= 18.0
        and occupied_area <= 10.0
        and major <= 6.0
        and vertical_core >= 1.5
        and vertical_core_fraction >= 0.55
    )

    is_pole = narrow_round_pole or pole_with_attachment

    # Trashcan / short cylinder
    is_trashcan = (
        0.45 <= height <= 1.80
        and 0.25 <= major <= 1.50
        and 0.18 <= minor <= 1.30
        and roundness >= 0.45
        and occupied_area <= 2.5
        and 0.35 <= vertical_ratio <= 5.0
    )

    # Bollard
    is_bollard = (
        0.35 <= height <= 1.60
        and 0.08 <= major <= 0.65
        and roundness >= 0.30
        and occupied_area <= 0.8
        and vertical_ratio >= 0.8
    )

    # Generic compact street furniture
    is_compact = (
        0.30 <= height <= 3.0
        and major <= 2.2
        and minor <= 1.8
        and occupied_area <= 3.5
    )

    if is_pole:
        classification[ids] = CLASS_POLE
        summary["pole"] += 1
    elif is_trashcan:
        classification[ids] = CLASS_TRASHCAN
        summary["trashcan"] += 1
    elif is_bollard:
        classification[ids] = CLASS_BOLLARD
        summary["bollard"] += 1
    elif is_compact:
        classification[ids] = CLASS_COMPACT
        summary["compact"] += 1
    else:
        summary["other"] += 1

print("Component decisions:")
for name, count in summary.items():
    print(f"  {name:12s}: {count:,}")

###

# Save only the classified non-ground cloud.
# Point membership is identical to not_ground_raw.laz; only Classification changes.

out = laspy.LasData(las.header)
out.points = las.points.copy()
out.classification = classification
out.write(output_file)

print("\nDONE")
print("Saved:", output_file)

counts = Counter(classification.tolist())
for code in sorted(counts):
    print(f"class {code:2d} ({CLASS_NAMES.get(code, 'unknown')}): {counts[code]:,} points")

print(f"Runtime: {time.time() - start_time:.1f} s")
print("ground.laz from stage 1 was not changed.")
