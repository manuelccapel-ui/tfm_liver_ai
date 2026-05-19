# ============================================================
# MODELO DE CLASIFICACIÓN SANO / PATOLÓGICO
# Arquitectura exacta usada en el notebook original
# ============================================================

import torch
import torch.nn as nn
import numpy as np

from src.config import DEVICE, PATH_MODELO_SANO


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
# GRADIENT REVERSAL LAYER
# ============================================================

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x, lambda_grl):
        return GradientReversalFn.apply(x, lambda_grl)


# ============================================================
# BLOQUE BÁSICO RESNET 3D
# ============================================================

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)

        return out


# ============================================================
# ENCODER RESNET 3D
# ============================================================

class ResNet3DEncoder(nn.Module):
    def __init__(self, block, layers):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv3d(
            2,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return torch.flatten(x, 1)


# ============================================================
# MODELO DANN RESNET34 3D
# ============================================================

class DANNResNet34_3D(nn.Module):
    def __init__(self, num_domains=4):
        super().__init__()

        self.encoder = ResNet3DEncoder(BasicBlock3D, [3, 4, 6, 3])
        self.grl = GradientReversal()

        self.label_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains),
        )

    def forward(self, x, lambda_grl=0.0):
        feats = self.encoder(x)

        label_logits = self.label_head(feats)
        domain_logits = self.domain_head(self.grl(feats, lambda_grl))

        return label_logits, domain_logits


# ============================================================
# CARGA DEL MODELO
# ============================================================

def load_healthy_pathological_model(device=DEVICE):
    """
    Carga el modelo de clasificación sano/patológico.
    """

    model = DANNResNet34_3D(num_domains=4).to(device)

    checkpoint = torch.load(PATH_MODELO_SANO, map_location=device)
    state_dict = extraer_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# INFERENCIA SANO / PATOLÓGICO
# ============================================================

@torch.no_grad()
def predict_healthy_pathological(model, image, liver_mask, threshold=0.5, device=DEVICE):
    """
    Clasifica un paciente como sano o patológico usando:
        canal 1: TAC
        canal 2: máscara de hígado predicha

    Devuelve:
        resultado: 'Sano' o 'Patológico'
        prob_pathological: probabilidad de clase patológica
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image debe ser un array numpy.")

    if not isinstance(liver_mask, np.ndarray):
        raise TypeError("liver_mask debe ser un array numpy.")

    if image.shape != liver_mask.shape:
        raise ValueError(
            f"image y liver_mask deben tener la misma forma. "
            f"image={image.shape}, liver_mask={liver_mask.shape}"
        )

    image = image.astype(np.float32)
    liver_mask = liver_mask.astype(np.float32)

    input_2ch = np.stack([image, liver_mask], axis=0)  # (2, Z, Y, X)

    x = torch.from_numpy(input_2ch).unsqueeze(0).float().to(device)  # (1, 2, Z, Y, X)

    label_logits, _ = model(x, lambda_grl=0.0)

    prob_pathological = torch.sigmoid(label_logits).item()

    if prob_pathological >= threshold:
        prediction = "Patológico"
    else:
        prediction = "Sano"

    return prediction, prob_pathological