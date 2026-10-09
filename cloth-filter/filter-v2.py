import os
import time
import laspy
import numpy as np
import CSF
from laspy import LazBackend
from google.colab import drive

drive.mount("/content/drive", force_remount=True)

file_path = "/content/drive/MyDrive/Lucas, Pascale, Dione, and Ricky VIP Project/LazFiles/"
input_file = os.path.join(file_path, "small.laz")
output_dir = os.path.join(file_path, "Output")
os.makedirs(output_dir, exist_ok=True)

ground_output = os.path.join(output_dir, "ground.laz")
not_ground_output = os.path.join(output_dir, "not_ground_raw.laz")

###

start_time = time.time()

inFile = laspy.read(input_file, laz_backend=LazBackend.Lazrs)

x = np.asarray(inFile.x, dtype=np.float64)
y = np.asarray(inFile.y, dtype=np.float64)
z = np.asarray(inFile.z, dtype=np.float64)

print(f"Loaded {len(inFile.points):,} points from small.laz")

###

# Working CSF settings from ModifiedClothFilter(11)

lo = np.percentile(z, 0.01)
hi = np.percentile(z, 99.99)

valid_mask = (z >= lo) & (z <= hi)
subset_idx = np.flatnonzero(valid_mask)

points_for_csf = np.column_stack((
    x[valid_mask],
    y[valid_mask],
    z[valid_mask],
))

csf_alg = CSF.CSF()
csf_alg.params.bSloopSmooth = False
csf_alg.params.cloth_resolution = 0.5
csf_alg.params.rigidness = 3
csf_alg.params.class_threshold = 0.5
csf_alg.params.iterations = 500

print(f"Running CSF on {len(points_for_csf):,} points...")

csf_alg.setPointCloud(points_for_csf.tolist())

ground_sub = CSF.VecInt()
object_sub = CSF.VecInt()
csf_alg.do_filtering(ground_sub, object_sub)

ground_indices = subset_idx[np.asarray(list(ground_sub), dtype=np.int64)]
not_ground_indices = subset_idx[np.asarray(list(object_sub), dtype=np.int64)]

trimmed_out_indices = np.flatnonzero(~valid_mask)
not_ground_indices = np.concatenate((not_ground_indices, trimmed_out_indices))

print(f"Ground points:     {len(ground_indices):,}")
print(f"Non-ground points: {len(not_ground_indices):,}")

###

# Save only the separation products.

ground_las = laspy.LasData(inFile.header)
ground_las.points = inFile.points[ground_indices]
ground_las.classification = np.full(len(ground_las.points), 2, dtype=np.uint8)
ground_las.write(ground_output)

not_ground_las = laspy.LasData(inFile.header)
not_ground_las.points = inFile.points[not_ground_indices]
not_ground_las.classification = np.full(len(not_ground_las.points), 1, dtype=np.uint8)
not_ground_las.write(not_ground_output)

print("\nDONE")
print("Saved:", ground_output)
print("Saved:", not_ground_output)
print(f"Runtime: {time.time() - start_time:.1f} s")
print("\nNow run 02_StreetFurniture_Classifier.ipynb")
