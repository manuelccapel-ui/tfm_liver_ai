# ============================================================
# CONFIGURACIÓN DE EJEMPLO - TFM LIVER AI
# ============================================================
#
# INSTRUCCIONES:
# 1. Copia este archivo y renómbralo como:
#
#       src/config.py
#
# 2. Modifica las rutas para que apunten a tus datos y modelos locales.
#
# Este archivo NO contiene rutas personales ni datos sensibles.
# ============================================================

from pathlib import Path
import torch


# ============================================================
# RUTAS PRINCIPALES
# ============================================================

# Carpeta raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Carpeta donde se guardarán los resultados
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Carpeta base donde están los datos y modelos.
# Cambiar esta ruta en cada ordenador.
BASE_DATA_DIR = Path(r"PATH_TO_YOUR_DATA_FOLDER")


# ============================================================
# RUTAS DE DATOS
# ============================================================

# Carpeta con los pacientes preprocesados en formato .npz
NPZ_DIR = BASE_DATA_DIR / "dataset_liver_colab_definitivo" / "volumes_npz"


# ============================================================
# RUTAS DE MODELOS ENTRENADOS
# ============================================================

PATH_MODELO_LIVER = (
    BASE_DATA_DIR
    / "modelo_seg_higado_google"
    / "best_resunet_liver_definitivo.pth"
)

PATH_MODELO_SANO = (
    BASE_DATA_DIR
    / "modelo_clasificador_sano_patologico_google"
    / "best_dann_2channel_v3.pth"
)

PATH_MODELO_TUMOR = (
    BASE_DATA_DIR
    / "modelo_seg_tumor_google"
    / "best_resunet3d.pth"
)

PATH_MODELO_HCC_CRLM = (
    BASE_DATA_DIR
    / "modelo_clasificacion_patologicos_google"
    / "best_hcc_crlm_3channel_FINAL.pth"
)


# ============================================================
# DISPOSITIVO
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PARÁMETROS DE INFERENCIA
# ============================================================

# Clasificación sano/patológico
THRESHOLD_SANO_PATOLOGICO = 0.5

# Segmentación tumoral
THRESHOLD_TUMOR = 0.5
MIN_TUMOR_COMPONENT_SIZE = 15

# Sliding window del modelo tumoral
PATCH_SIZE_TUMOR = (64, 128, 128)
OVERLAP_TUMOR = 0.5

# ROI del clasificador HCC/CRLM
ROI_HCC_CRLM = (32, 96, 96)


# ============================================================
# FUNCIÓN DE COMPROBACIÓN
# ============================================================

def print_config():
    """
    Muestra por pantalla las rutas principales y comprueba si existen.
    """

    print("==================================================")
    print("CONFIGURACIÓN DEL PROYECTO")
    print("==================================================")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"BASE_DATA_DIR: {BASE_DATA_DIR}")
    print(f"NPZ_DIR: {NPZ_DIR}")
    print(f"OUTPUTS_DIR: {OUTPUTS_DIR}")
    print(f"DEVICE: {DEVICE}")
    print()

    paths = {
        "NPZ_DIR": NPZ_DIR,
        "PATH_MODELO_LIVER": PATH_MODELO_LIVER,
        "PATH_MODELO_SANO": PATH_MODELO_SANO,
        "PATH_MODELO_TUMOR": PATH_MODELO_TUMOR,
        "PATH_MODELO_HCC_CRLM": PATH_MODELO_HCC_CRLM,
    }

    for name, path in paths.items():
        status = "OK" if Path(path).exists() else "NO ENCONTRADO"
        print(f"{name}: {status}")
        print(f"  {path}")

    print("==================================================")