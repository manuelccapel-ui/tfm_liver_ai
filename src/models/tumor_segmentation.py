# ============================================================
# MODELO DE SEGMENTACIÓN TUMORAL
# Arquitectura exacta usada en el notebook original
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from src.config import (
    DEVICE,
    PATH_MODELO_TUMOR,
    PATCH_SIZE_TUMOR,
    OVERLAP_TUMOR,
    THRESHOLD_TUMOR,
    MIN_TUMOR_COMPONENT_SIZE,
)


# ============================================================
# FUNCIONES AUXILIARES DE CARGA
# ============================================================

def limpiar_prefijo_module(state_dict):
    """
    Elimina el prefijo 'module.' si el modelo fue entrenado con DataParallel.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    nuevo = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            nuevo[k.replace("module.", "", 1)] = v
        else:
            nuevo[k] = v

    return nuevo


def extraer_state_dict(checkpoint):
    """
    Permite cargar tanto pesos guardados directamente como checkpoints.
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return limpiar_prefijo_module(checkpoint["model_state_dict"])
        if "state_dict" in checkpoint:
            return limpiar_prefijo_module(checkpoint["state_dict"])

    return limpiar_prefijo_module(checkpoint)


# ============================================================
# BLOQUE RESIDUAL 3D
# ============================================================

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )

        self.skip = nn.Identity()

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


# ============================================================
# RESIDUAL U-NET 3D TUMORAL
# ============================================================

class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_ch=32):
        super().__init__()

        self.enc1 = ResBlock3D(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResBlock3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResBlock3D(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ResBlock3D(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ResBlock3D(base_ch * 8, base_ch * 16, dropout=0.20)

        self.up4 = nn.ConvTranspose3d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = ResBlock3D(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self._match_and_concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._match_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_concat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    @staticmethod
    def _match_and_concat(x, skip):
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]

        if dz != 0 or dy != 0 or dx != 0:
            skip = skip[
                :,
                :,
                dz // 2 : dz // 2 + x.shape[2],
                dy // 2 : dy // 2 + x.shape[3],
                dx // 2 : dx // 2 + x.shape[4],
            ]

        return torch.cat([x, skip], dim=1)


# ============================================================
# CARGA DEL MODELO
# ============================================================

def load_tumor_model(device=DEVICE):
    """
    Carga el modelo de segmentación tumoral.
    """

    checkpoint = torch.load(PATH_MODELO_TUMOR, map_location=device)

    if isinstance(checkpoint, dict):
        base_ch = checkpoint.get("base_ch", 32)
    else:
        base_ch = 32

    model = ResidualUNet3D(
        in_channels=2,
        out_channels=1,
        base_ch=base_ch,
    ).to(device)

    state_dict = extraer_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# FUNCIONES AUXILIARES DE INFERENCIA TUMORAL
# ============================================================

def remove_small_components_3d(mask, min_size=15):
    """
    Elimina componentes conectadas pequeñas en una máscara 3D.
    """

    mask = (mask > 0).astype(np.uint8)

    labeled, n = ndimage.label(mask)

    if n == 0:
        return mask.astype(np.float32)

    sizes = np.bincount(labeled.ravel())

    remove_labels = np.where(sizes < min_size)[0]
    remove_labels = remove_labels[remove_labels != 0]

    out = mask.copy()

    for lab in remove_labels:
        out[labeled == lab] = 0

    return out.astype(np.float32)


def generate_sliding_starts(full_shape, patch_shape, overlap=0.5):
    """
    Genera las posiciones iniciales de cada parche para sliding window.
    """

    starts_per_dim = []

    for f, p in zip(full_shape, patch_shape):
        if f <= p:
            starts_per_dim.append([0])
            continue

        step = max(1, int(p * (1 - overlap)))

        starts = list(range(0, f - p + 1, step))

        if starts[-1] != f - p:
            starts.append(f - p)

        starts_per_dim.append(starts)

    return starts_per_dim


@torch.no_grad()
def predict_tumor_sliding(
    model,
    input_2ch,
    patch_shape=PATCH_SIZE_TUMOR,
    overlap=OVERLAP_TUMOR,
    device=DEVICE,
):
    """
    Segmentación tumoral con ventana deslizante.

    input_2ch debe tener shape:
        (2, Z, Y, X)

    Devuelve:
        prob_final: probabilidad tumoral con shape (Z, Y, X)
    """

    model.eval()

    if not isinstance(input_2ch, np.ndarray):
        raise TypeError("input_2ch debe ser un array numpy.")

    if input_2ch.ndim != 4:
        raise ValueError(f"input_2ch debe tener shape (2,Z,Y,X), pero tiene {input_2ch.shape}")

    if input_2ch.shape[0] != 2:
        raise ValueError(f"input_2ch debe tener 2 canales, pero tiene {input_2ch.shape[0]}")

    _, z, y, x = input_2ch.shape

    prob_sum = np.zeros((z, y, x), dtype=np.float32)
    count = np.zeros((z, y, x), dtype=np.float32)

    z_starts, y_starts, x_starts = generate_sliding_starts(
        full_shape=(z, y, x),
        patch_shape=patch_shape,
        overlap=overlap,
    )

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1 = z0 + patch_shape[0]
                y1 = y0 + patch_shape[1]
                x1 = x0 + patch_shape[2]

                patch = input_2ch[:, z0:z1, y0:y1, x0:x1].astype(np.float32)

                xt = torch.from_numpy(patch[None]).float().to(device)

                prob = torch.sigmoid(model(xt)).cpu().numpy()[0, 0]

                prob_sum[z0:z1, y0:y1, x0:x1] += prob
                count[z0:z1, y0:y1, x0:x1] += 1.0

    prob_final = prob_sum / np.maximum(count, 1e-8)

    return prob_final


@torch.no_grad()
def predict_tumor_mask(
    model,
    image,
    liver_mask,
    threshold=THRESHOLD_TUMOR,
    min_size=MIN_TUMOR_COMPONENT_SIZE,
    device=DEVICE,
):
    """
    Predice la máscara tumoral usando:
        canal 1: TAC
        canal 2: máscara hepática predicha

    Devuelve:
        tumor_mask: máscara binaria posprocesada
        tumor_prob: probabilidad tumoral
    """

    if image.shape != liver_mask.shape:
        raise ValueError(
            f"image y liver_mask deben tener la misma forma. "
            f"image={image.shape}, liver_mask={liver_mask.shape}"
        )

    image = image.astype(np.float32)
    liver_mask = liver_mask.astype(np.float32)

    input_2ch = np.stack([image, liver_mask], axis=0)

    tumor_prob = predict_tumor_sliding(
        model=model,
        input_2ch=input_2ch,
        patch_shape=PATCH_SIZE_TUMOR,
        overlap=OVERLAP_TUMOR,
        device=device,
    )

    # --------------------------------------------------------
    # 1. Umbralización
    # --------------------------------------------------------
    tumor_mask_raw = (tumor_prob >= threshold).astype(np.uint8)

    # --------------------------------------------------------
    # 2. Restricción anatómica al hígado
    # Esto es lo que tenías en tu notebook original:
    # pred_tumor_raw = pred_tumor_raw * pred_liver
    # --------------------------------------------------------
    tumor_mask_raw = tumor_mask_raw * (liver_mask > 0).astype(np.uint8)

    # --------------------------------------------------------
    # 3. Eliminación de ruido pequeño
    # --------------------------------------------------------
    tumor_mask = remove_small_components_3d(
        tumor_mask_raw,
        min_size=min_size,
    ).astype(np.uint8)

    return tumor_mask, tumor_prob