# ============================================================
# MODELO DE CLASIFICACIÓN HCC / CRLM
# Arquitectura exacta usada en el notebook original
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import torchvision.models.video as models_video

from src.config import DEVICE, PATH_MODELO_HCC_CRLM, ROI_HCC_CRLM


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
# CLASIFICADOR 3D HCC / CRLM
# ============================================================

class TumorClassifier3D_ROI(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models_video.r3d_18(weights=None)

        original_conv = self.backbone.stem[0]

        self.backbone.stem[0] = nn.Conv3d(
            in_channels=3,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        num_ftrs = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 1),
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# CARGA DEL MODELO
# ============================================================

def load_hcc_crlm_model(device=DEVICE):
    """
    Carga el modelo de clasificación HCC/CRLM.
    """

    model = TumorClassifier3D_ROI().to(device)

    checkpoint = torch.load(PATH_MODELO_HCC_CRLM, map_location=device)
    state_dict = extraer_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# EXTRACCIÓN DE ROI 3D
# ============================================================

def extract_roi(volume, cz, cy, cx, crop_z=32, crop_xy=96, is_mask=False):
    """
    Extrae una ROI 3D centrada en (cz, cy, cx).

    Si se sale de los límites, rellena con padding:
        - TAC: valor -1.0
        - máscaras: valor 0
    """

    z, y, x = volume.shape

    z0 = max(0, cz - crop_z // 2)
    z1 = min(z, cz + crop_z // 2)

    y0 = max(0, cy - crop_xy // 2)
    y1 = min(y, cy + crop_xy // 2)

    x0 = max(0, cx - crop_xy // 2)
    x1 = min(x, cx + crop_xy // 2)

    crop = volume[z0:z1, y0:y1, x0:x1]

    pad_value = 0 if is_mask else -1.0

    output = np.full(
        (crop_z, crop_xy, crop_xy),
        pad_value,
        dtype=np.float32,
    )

    oz0 = (crop_z - (z1 - z0)) // 2
    oy0 = (crop_xy - (y1 - y0)) // 2
    ox0 = (crop_xy - (x1 - x0)) // 2

    output[
        oz0 : oz0 + (z1 - z0),
        oy0 : oy0 + (y1 - y0),
        ox0 : ox0 + (x1 - x0),
    ] = crop

    return output


def get_tumor_center(tumor_mask):
    """
    Calcula el centro de masas aproximado de la máscara tumoral.

    Si la máscara está vacía, devuelve None.
    """

    coords = np.argwhere(tumor_mask > 0)

    if coords.shape[0] == 0:
        return None

    cz, cy, cx = coords.mean(axis=0).astype(int)

    return int(cz), int(cy), int(cx)


# ============================================================
# INFERENCIA HCC / CRLM
# ============================================================

@torch.no_grad()
def predict_hcc_crlm(
    model,
    image,
    liver_mask,
    tumor_mask,
    device=DEVICE,
):
    """
    Clasifica el tipo tumoral usando una ROI centrada en la máscara tumoral.

    Entrada:
        image: TAC, shape (Z,Y,X)
        liver_mask: máscara hepática, shape (Z,Y,X)
        tumor_mask: máscara tumoral, shape (Z,Y,X)

    Devuelve:
        prediction: "HCC" o "CRLM"
        prob_crlm: probabilidad de CRLM, suponiendo que la clase positiva del entrenamiento era CRLM
    """

    if image.shape != liver_mask.shape or image.shape != tumor_mask.shape:
        raise ValueError(
            "image, liver_mask y tumor_mask deben tener la misma forma. "
            f"image={image.shape}, liver_mask={liver_mask.shape}, tumor_mask={tumor_mask.shape}"
        )

    center = get_tumor_center(tumor_mask)

    if center is None:
        raise ValueError("La máscara tumoral está vacía. No se puede clasificar HCC/CRLM.")

    cz, cy, cx = center

    crop_z, crop_xy, _ = ROI_HCC_CRLM

    image_roi = extract_roi(
        image.astype(np.float32),
        cz,
        cy,
        cx,
        crop_z=crop_z,
        crop_xy=crop_xy,
        is_mask=False,
    )

    liver_roi = extract_roi(
        liver_mask.astype(np.float32),
        cz,
        cy,
        cx,
        crop_z=crop_z,
        crop_xy=crop_xy,
        is_mask=True,
    )

    tumor_roi = extract_roi(
        tumor_mask.astype(np.float32),
        cz,
        cy,
        cx,
        crop_z=crop_z,
        crop_xy=crop_xy,
        is_mask=True,
    )

    input_3ch = np.stack(
        [image_roi, liver_roi, tumor_roi],
        axis=0,
    )  # (3, Z, Y, X)

    x = torch.from_numpy(input_3ch).unsqueeze(0).float().to(device)

    logits = model(x)
    prob_crlm = torch.sigmoid(logits).item()

    if prob_crlm >= 0.5:
        prediction = "CRLM"
    else:
        prediction = "HCC"

    return prediction, prob_crlm