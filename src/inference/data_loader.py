# ============================================================
# CARGA DE PACIENTES .NPZ PARA INFERENCIA
# ============================================================

from pathlib import Path
import numpy as np


def load_patient_npz(path_npz):
    """
    Carga un paciente en formato .npz para inferencia.

    En la pipeline real solo usamos:
        - image: volumen TAC preprocesado

    No cargamos liver_mask porque la máscara hepática debe ser
    predicha por el modelo de segmentación de hígado.
    """

    path_npz = Path(path_npz)

    if not path_npz.exists():
        raise FileNotFoundError(f"No existe el archivo: {path_npz}")

    data = np.load(path_npz, allow_pickle=True)

    if "image" not in data.files:
        raise KeyError(
            f"El archivo {path_npz.name} no contiene la clave obligatoria 'image'. "
            f"Claves disponibles: {data.files}"
        )

    image = data["image"].astype(np.float32)

    patient = {
        "patient_id": path_npz.stem,
        "path": path_npz,
        "image": image,
        "keys": list(data.files),
    }

    return patient


def summarize_patient(patient):
    """
    Muestra por pantalla un resumen del paciente cargado.
    """

    print("==================================================")
    print("PACIENTE CARGADO PARA INFERENCIA")
    print("==================================================")
    print(f"ID paciente: {patient['patient_id']}")
    print(f"Ruta: {patient['path']}")
    print(f"Claves originales del .npz: {patient['keys']}")
    print()

    image = patient["image"]

    print("TAC:")
    print(f"  shape: {image.shape}")
    print(f"  dtype: {image.dtype}")
    print(f"  min: {image.min():.4f}")
    print(f"  max: {image.max():.4f}")
    print(f"  mean: {image.mean():.4f}")

    print("==================================================")


def get_first_npz(npz_dir):
    """
    Devuelve el primer archivo .npz encontrado en una carpeta.
    """

    npz_dir = Path(npz_dir)
    files = sorted(npz_dir.glob("*.npz"))

    if len(files) == 0:
        raise FileNotFoundError(f"No hay archivos .npz en: {npz_dir}")

    return files[0]