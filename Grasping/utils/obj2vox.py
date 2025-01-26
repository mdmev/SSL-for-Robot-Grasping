import trimesh
import numpy as np
import torch

# Load your 3D mesh
mesh = trimesh.load("/home/guillfa/TUM/ADL4R/studentGrasping/student_grasps_v1/03593526/b5f802103edd92d38c6cecb99943f941/1/mesh.obj")

# Target bounding box dimensions
target_size = 32
bounding_box = np.array([target_size, target_size, target_size])

# Resize the mesh to fit within the target bounding box
scale_factor = target_size / max(mesh.extents)
mesh.apply_scale(scale_factor)
mesh.show()

# Define voxel size based on the resized mesh
voxel_size = max(mesh.extents) / target_size
# Voxelize the resized mesh
voxelized_mesh = mesh.voxelized(voxel_size)

# Convert the voxelized mesh into a 32x32x32 grid for further processing
voxels = voxelized_mesh.matrix.astype(np.float32)

# Normalize voxel values to range [0, 1]
voxels = voxels / np.max(voxels)

# Add channel dimension for CNN compatibility
voxels = np.expand_dims(voxels, axis=0)  # Shape: (1, 32, 32, 32)

# Save the voxel grid to a .npy file
npy_file = "voxel_grid_32x32x32.npy"
np.save(npy_file, voxels)


loaded_voxels = np.load(npy_file)  # Load saved voxel grid
loaded_voxels = loaded_voxels[0]  # Remove the channel dimension (Shape: 7, 11, 33)

# Ensure voxel grid dimensions are consistent with loaded data
voxel_grid = trimesh.voxel.VoxelGrid(loaded_voxels)

# Convert the voxel grid back into a mesh
reconstructed_mesh = voxel_grid.as_boxes()

# Display the reconstructed mesh
reconstructed_mesh.show()
