# src/data_utils/augmentations.py
import torch


def _ensure_time_last(x):
    """
    Ensures tensor is in [B, T, F] format.
    If input is [B, F, T], converts to [B, T, F].
    Returns converted tensor and flag to revert.
    """
    if x.dim() != 3:
        raise ValueError("Input must be 3D tensor")

    # Heuristic: if middle dim is smaller, likely [B, F, T]
    if x.shape[1] < x.shape[2]:
        return x.permute(0, 2, 1), True
    return x, False


def _restore_shape(x, was_transposed):
    if was_transposed:
        return x.permute(0, 2, 1)
    return x


def missing_data_injection(x, y, z, rate):
    """
    Randomly injects missing values (zeros) and updates missing mask.

    Args:
        x: [B, T, F] or [B, F, T]
        y: labels
        z: missing mask
        rate: fraction of elements to corrupt
    """
    if rate <= 0:
        return x, y, z

    x_proc, transposed = _ensure_time_last(x)

    total_elements = x_proc.numel()
    miss_size = int(rate * total_elements)

    if miss_size == 0:
        return x, y, z

    x_flat = x_proc.reshape(-1)

    indices = torch.randint(0, total_elements, (miss_size,), device=x.device)

    x_flat[indices] = 0

    # Update missing mask consistently
    z_proc = z.clone().reshape(-1)
    z_proc[indices] = 1

    x_out = x_flat.view_as(x_proc)
    z_out = z_proc.view_as(z)

    x_out = _restore_shape(x_out, transposed)

    return x_out, y, z_out


def point_ano(x, y, z, rate):
    """
    Injects point anomalies at the FINAL timestep.

    Args:
        x: [B, T, F] or [B, F, T]
    """
    if rate <= 0:
        return x, y, z

    x_proc, transposed = _ensure_time_last(x)

    B, T, F = x_proc.shape
    aug_size = int(rate * B)

    if aug_size <= 0:
        return x, y, z

    idx = torch.randint(0, B, (aug_size,), device=x.device)

    x_aug = x_proc[idx].clone()
    y_aug = y[idx].clone()
    z_aug = z[idx].clone()

    # Generate noise
    half = aug_size // 2
    noise_pos = torch.randint(1, 21, (half,), device=x.device).float() / 2
    noise_neg = torch.randint(-20, 0, (aug_size - half,), device=x.device).float() / 2
    ano_noise = torch.cat((noise_pos, noise_neg))

    # Apply to final timestep
    x_aug[:, -1, :] += ano_noise.view(-1, 1)

    # Label anomaly
    if y_aug.dim() == 2:
        y_aug[:, -1] = 1
    else:
        y_aug[:] = 1

    x_aug = _restore_shape(x_aug, transposed)

    return x_aug, y_aug, z_aug


def seg_ano(x, y, z, rate, method="swap"):
    """
    Segment anomaly via temporal swapping.

    Args:
        x: [B, T, F] or [B, F, T]
        method: currently supports 'swap'
    """
    if rate <= 0:
        return x, y, z

    x_proc, transposed = _ensure_time_last(x)

    B, T, F = x_proc.shape
    aug_size = int(rate * B)

    if aug_size <= 0:
        return x, y, z

    # Sample distinct pairs
    idx_1 = torch.randint(0, B, (aug_size,), device=x.device)
    idx_2 = torch.randint(0, B, (aug_size,), device=x.device)

    mask = idx_1 == idx_2
    while mask.any():
        idx_2[mask] = torch.randint(0, B, (mask.sum(),), device=x.device)
        mask = idx_1 == idx_2

    x_aug = x_proc[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()

    # Ensure valid segment start
    if T <= 2:
        return x, y, z

    time_start = torch.randint(
        low=max(1, T // 4),
        high=T - 1,
        size=(aug_size,),
        device=x.device,
    )

    for i in range(aug_size):
        if method == "swap":
            t0 = time_start[i]
            x_aug[i, t0:, :] = x_proc[idx_2[i], t0:, :]

            if y_aug.dim() == 2:
                y_aug[i, t0:] = 1
            else:
                y_aug[i] = 1

    x_aug = _restore_shape(x_aug, transposed)

    return x_aug, y_aug, z_aug
