import torchio as tio
import matplotlib.pyplot as plt

# ---- Load example MRI ----
subject = tio.datasets.Colin27()

# Get full 3D volume
image_3d = subject.t1.data.squeeze().numpy()

# Pick center slice (axial)
slice_idx = image_3d.shape[2] // 2
image = image_3d[:, :, slice_idx]

# ---- Apply custom transform ----
transform = tio.RandomBiasFieldDenoise(noise_reduction_factor=0.3)
denoised_subject = transform(subject)

# Extract denoised slice
denoised_3d = denoised_subject.t1.data.squeeze().numpy()
denoised_image = denoised_3d[:, :, slice_idx]

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(denoised_image, cmap='gray')
axes[1].set_title("After Denoise")
axes[1].axis("off")

plt.tight_layout()
plt.show()
