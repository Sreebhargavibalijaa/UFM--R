# utils.py — Enhanced Heatmap Overlay with Original Image and Color Scale

def plot_patch_overlay_on_image(contrib_map, rows, cols, img_tensor, filename="overlay.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as T
    import os

    # Convert contribution tensor to array
    contrib_array = contrib_map.detach().cpu().numpy()[0]

    if contrib_array.size != rows * cols:
        raise ValueError(f"Cannot reshape contribution map of size {contrib_array.size} into ({rows}, {cols})")

    # Reshape into patch map
    patch_map = contrib_array.reshape((rows, cols))

    # Convert original image
    img = T.ToPILImage()(img_tensor.squeeze(0)).convert("RGB")
    img_np = np.array(img.resize((cols * 10, rows * 10)))  # Match patch size visually

    # Plot heatmap on top of image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, alpha=0.9)
    heatmap = ax.imshow(patch_map, cmap="hot", alpha=0.5, interpolation="nearest",
                        extent=(0, img_np.shape[1], img_np.shape[0], 0))
    plt.colorbar(heatmap, ax=ax, shrink=0.8, label="Contribution Intensity")
    ax.axis("off")

    # Save overlay
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    return filename
