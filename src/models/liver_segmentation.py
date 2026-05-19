# ============================================================
# MODELO DE SEGMENTACIÓN HEPÁTICA
# Arquitectura exacta usada en el notebook original
# ============================================================

import torch
import torch.nn as nn
import numpy as np

from src.config import DEVICE, PATH_MODELO_LIVER


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
# BLOQUE RESIDUAL 3D PARA HÍGADO
# ============================================================

class ResBlock3D_Liver(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(
            in_ch,
            out_ch,
            1,
            bias=False,
        )

        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


# ============================================================
# U-NET 3D DE SEGMENTACIÓN HEPÁTICA
# ============================================================

class UNet_Liver(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=16):
        super().__init__()

        self.enc1 = ResBlock3D_Liver(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResBlock3D_Liver(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResBlock3D_Liver(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ResBlock3D_Liver(base_ch * 4, base_ch * 8)

        self.up2 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec2 = ResBlock3D_Liver(base_ch * 8, base_ch * 4)

        self.up1 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec1 = ResBlock3D_Liver(base_ch * 4, base_ch * 2)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2),
            ResBlock3D_Liver(base_ch, base_ch),
            nn.Conv3d(base_ch, out_channels, 1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d2 = self.dec2(torch.cat([self.up2(b), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))

        return self.out_conv(d1)


# ============================================================
# CARGA DEL MODELO
# ============================================================

def load_liver_model(device=DEVICE):
    """
    Carga el modelo entrenado de segmentación hepática.
    """

    model = UNet_Liver(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(PATH_MODELO_LIVER, map_location=device)
    state_dict = extraer_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# INFERENCIA DE MÁSCARA HEPÁTICA
# ============================================================

@torch.no_grad()
def predict_liver_mask(model, image, threshold=0.5, device=DEVICE):
    """
    Predice la máscara hepática a partir de un TAC 3D.

    Parámetros:
        model: modelo UNet_Liver cargado
        image: array numpy con forma (Z, Y, X)
        threshold: umbral de binarización

    Devuelve:
        liver_mask: array uint8 con forma (Z, Y, X)
        liver_prob: array float32 con probabilidades
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image debe ser un array numpy.")

    if image.ndim != 3:
        raise ValueError(f"image debe tener forma (Z,Y,X), pero tiene {image.shape}")

    x = torch.from_numpy(image.astype(np.float32))
    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
    x = x.to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)

    liver_prob = probs.squeeze().cpu().numpy().astype(np.float32)
    liver_mask = (liver_prob >= threshold).astype(np.uint8)

    return liver_mask, liver_prob