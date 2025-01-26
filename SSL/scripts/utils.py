import torch
import matplotlib.pyplot as plt

def apply_random_mask(input_tensor, device, mask_ratio=0.3, block_size=3):
    mask = torch.ones_like(input_tensor)
    for i in range(mask.size(2) // block_size):
        for j in range(mask.size(3) // block_size):
            for k in range(mask.size(4) // block_size):
                if torch.rand(1).item() < mask_ratio:
                    mask[:, :, i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size,
                         k*block_size:(k+1)*block_size] = 0
    return input_tensor * mask.to(device)

def update_teacher_weights(student, teacher, step, ema_decay):
    with torch.no_grad():
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

def visualize_mask(mask1, mask2, frames_to_show=[0, 5, 10, 15, 20, 25]):
    # Iterate over the specified depth slices for the first item in the batch
    for d in frames_to_show:
        fix, axes = plt.subplots(1, 3, figsize=(15, 5))
        # First mask
        axes[0].imshow(mask1[d, :, :], cmap='gray')
        axes[0].set_title(f'Mask1 - Depth {d}')
        axes[0].axis('off')  # Hide axis for better visualization

        # Third mask
        axes[2].imshow(mask2[d, :, :], cmap='gray')
        axes[2].set_title(f'Mask2 - Depth {d}')
        axes[2].axis('off')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

def create_random_mask(s, mask_percentage):
    mask = torch.ones(s, dtype=torch.float32)
    num_to_mask = int(s[0]*s[1]*s[2]*mask_percentage)
    # Flatten, randomly choose elements to set to 0
    flat_idx = torch.randperm(s[0]*s[1]*s[2])[:num_to_mask]
    mask.view(-1)[flat_idx] = 0
    return mask

def create_block_mask(s, mask_percentage, block_size=8):
    mask = torch.ones(s, dtype=torch.float32)
    num_to_mask = int(s[0] * s[1] * s[2] * mask_percentage)
    masked_count = 0

    while masked_count < num_to_mask:
        # Randomly select a starting point
        i = torch.randint(0, s[0] - block_size + 1, (1,)).item()
        j = torch.randint(0, s[1] - block_size + 1, (1,)).item()
        k = torch.randint(0, s[2] - block_size + 1, (1,)).item()

        # Mask a block of size block_size x block_size x block_size
        mask[i:i+block_size, j:j+block_size, k:k+block_size] = 0
        masked_count += block_size**3

    return mask

def apply_mask_3d(input_tensor, mask):
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    output_mask = torch.nn.functional.interpolate(
        mask,
        size=input_tensor.shape[-1],
        mode='nearest'
    ).squeeze(0).squeeze(0)
    return input_tensor * output_mask

def apply_mask_decoder(input_tensor, mask):
    mask = mask.unsqueeze(1)
    output_mask = torch.nn.functional.interpolate(
        mask,
        size=input_tensor.shape[-1],
        mode='nearest'
    ).squeeze(1)
    input_tensor = input_tensor.squeeze(1)
    return input_tensor * output_mask

def reconstruct_and_save_mesh(sdf_file: str, output_obj_file: str, level: float = 0.015):
    import numpy as np
    import trimesh
    import skimage.measure

    # Load the SDF data
    sdf = np.load(sdf_file)
    print(f"SDF loaded with shape: {sdf.shape}")

    # Reconstruct the mesh using marching cubes
    vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save the mesh to an OBJ file
    mesh.export(output_obj_file)
    print(f"Reconstructed mesh saved to {output_obj_file}")

