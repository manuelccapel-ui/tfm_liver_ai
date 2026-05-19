# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# Visualización igual que el notebook original
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_center_slice_from_mask(mask):
    """
    Devuelve el corte axial correspondiente al centro medio de una máscara 3D.

    Esto replica la lógica del notebook original:

        coords = np.argwhere(pred_tumor > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)

    Si la máscara está vacía, devuelve None.
    """

    if mask is None:
        return None

    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return None

    cz, cy, cx = coords.mean(axis=0).astype(int)

    return int(cz)


def get_best_liver_slice(liver_mask):
    """
    Si no hay tumor, usamos el corte con mayor área hepática.
    """

    if liver_mask is None:
        return None

    areas = liver_mask.sum(axis=(1, 2))

    if areas.max() == 0:
        return liver_mask.shape[0] // 2

    return int(np.argmax(areas))


def save_overview_png(
    image,
    liver_mask,
    output_path,
    tumor_mask=None,
    title="INFORME IA - Diagnóstico Automático",
):
    """
    Visualización tipo notebook original:
        - TAC original
        - Hígado en verde
        - Tumor en rojo, si existe

    La diferencia importante respecto a la versión anterior es que,
    si hay tumor, el corte elegido es el centro medio del tumor,
    no el corte con mayor número de píxeles tumorales.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Elegir corte igual que en el notebook
    # --------------------------------------------------------
    if tumor_mask is not None and np.any(tumor_mask > 0):
        mejor_z = get_center_slice_from_mask(tumor_mask)
    else:
        mejor_z = get_best_liver_slice(liver_mask)

    if mejor_z is None:
        mejor_z = image.shape[0] // 2

    hay_tumor = tumor_mask is not None and np.any(tumor_mask > 0)

    n_cols = 3 if hay_tumor else 2

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # 1. TAC original
    # --------------------------------------------------------
    axes[0].imshow(image[mejor_z], cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("TAC Original")
    axes[0].axis("off")

        # --------------------------------------------------------
    # 2. Segmentación de hígado
    # --------------------------------------------------------
    axes[1].imshow(image[mejor_z], cmap="gray", vmin=-1, vmax=1)

    if liver_mask is not None and np.any(liver_mask[mejor_z] > 0):
        liver_overlay = np.ma.masked_where(
            liver_mask[mejor_z] == 0,
            liver_mask[mejor_z],
        )
        axes[1].imshow(liver_overlay, cmap="Greens", alpha=0.45)

    axes[1].set_title("Detección de Hígado (IA)")
    axes[1].axis("off")

    # --------------------------------------------------------
    # 3. Segmentación tumoral
    # --------------------------------------------------------
    if hay_tumor:
        axes[2].imshow(image[mejor_z], cmap="gray", vmin=-1, vmax=1)

        if np.any(tumor_mask[mejor_z] > 0):
            tumor_overlay = np.ma.masked_where(
                tumor_mask[mejor_z] == 0,
                tumor_mask[mejor_z],
            )
            axes[2].imshow(tumor_overlay, cmap="Reds", alpha=0.70)

        axes[2].set_title("Detección de Lesión")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path