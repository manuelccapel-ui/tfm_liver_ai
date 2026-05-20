# ============================================================
# PREPROCESADO DE VOLÚMENES MÉDICOS PARA TFM LIVER AI
# ============================================================
#
# Este módulo replica el preprocesado usado para generar los .npz
# definitivos del proyecto:
#
#   - Resampling a spacing (1.5, 1.5, 1.5)
#   - Conversión a array (Z, Y, X)
#   - Ventana HU [-150, 250]
#   - Normalización a [-1, 1]
#   - Center crop/pad a (128, 224, 224)
#   - Guardado como .npz con clave "image"
#
# ============================================================

from pathlib import Path
import numpy as np
import SimpleITK as sitk

from src.preprocessing.medical_io import read_medical_image


# ============================================================
# PARÁMETROS DEL PREPROCESADO ORIGINAL
# ============================================================

TARGET_SPACING = (1.5, 1.5, 1.5)
FULL_SHAPE = (128, 224, 224)

HU_MIN = -150
HU_MAX = 250


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def resample_sitk_image(
    image,
    out_spacing=TARGET_SPACING,
    is_mask=False,
):
    """
    Reescala una imagen SimpleITK a un spacing fijo.

    Para imágenes se usa interpolación lineal.
    Para máscaras se usa vecino más cercano.

    Parámetros:
        image: imagen SimpleITK
        out_spacing: spacing objetivo en formato (x, y, z)
        is_mask: si True, usa interpolación nearest neighbor

    Devuelve:
        imagen SimpleITK reescalada
    """

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    out_size = [
        int(round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / out_spacing[2])),
    ]

    out_size = [max(1, int(s)) for s in out_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def center_crop_or_pad_3d(arr, target_shape=FULL_SHAPE):
    """
    Recorta o rellena un array 3D al tamaño objetivo.

    Esta función replica la lógica usada en el preprocesado original.

    Parámetros:
        arr: array 3D con forma (Z, Y, X)
        target_shape: forma final deseada (Z, Y, X)

    Devuelve:
        array 3D con forma target_shape
    """

    tz, ty, tx = target_shape
    z, y, x = arr.shape

    out = np.zeros((tz, ty, tx), dtype=arr.dtype)

    z0 = max(0, (z - tz) // 2)
    y0 = max(0, (y - ty) // 2)
    x0 = max(0, (x - tx) // 2)

    z1 = min(z, z0 + tz)
    y1 = min(y, y0 + ty)
    x1 = min(x, x0 + tx)

    crop = arr[z0:z1, y0:y1, x0:x1]

    oz = max(0, (tz - crop.shape[0]) // 2)
    oy = max(0, (ty - crop.shape[1]) // 2)
    ox = max(0, (tx - crop.shape[2]) // 2)

    out[
        oz:oz + crop.shape[0],
        oy:oy + crop.shape[1],
        ox:ox + crop.shape[2],
    ] = crop

    return out


def normalize_hu_to_minus1_1(img_arr, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Aplica ventana HU y normaliza el volumen a [-1, 1].

    Pasos:
        1. Clip a [hu_min, hu_max]
        2. Escalado a [0, 1]
        3. Escalado a [-1, 1]
    """

    img_arr = np.clip(img_arr, hu_min, hu_max)

    img_arr = (img_arr - hu_min) / (hu_max - hu_min)
    img_arr = img_arr * 2.0 - 1.0

    return img_arr.astype(np.float32)


def ensure_3d_array(arr):
    """
    Asegura que el array sea 3D.

    SimpleITK puede devolver:
        - 2D si se lee un único corte
        - 3D si se lee un volumen normal
        - 4D si el archivo tiene canales/fases

    En esta primera versión:
        - si es 2D, se añade eje Z
        - si es 3D, se deja igual
        - si es 4D, se toma el primer volumen/canal
    """

    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    elif arr.ndim == 3:
        pass

    elif arr.ndim == 4:
        # Caso típico: (T, Z, Y, X) o similar.
        # Tomamos el primer volumen para mantener una entrada 3D.
        arr = arr[0]

    else:
        raise ValueError(
            f"Se esperaba un array 2D, 3D o 4D, pero se recibió shape={arr.shape}"
        )

    return arr


# ============================================================
# FUNCIÓN PRINCIPAL DE PREPROCESADO
# ============================================================

def preprocess_sitk_image_to_array(image_sitk):
    """
    Preprocesa una imagen SimpleITK y devuelve un array listo para inferencia.

    Devuelve:
        img_arr: array con shape (128, 224, 224), dtype float32,
                 intensidades normalizadas a [-1, 1]
    """

    # --------------------------------------------------------
    # 1. Resampling a spacing fijo
    # --------------------------------------------------------
    image_rs = resample_sitk_image(
        image=image_sitk,
        out_spacing=TARGET_SPACING,
        is_mask=False,
    )

    # --------------------------------------------------------
    # 2. SimpleITK -> NumPy
    # SimpleITK devuelve array en formato (Z, Y, X)
    # --------------------------------------------------------
    img_arr = sitk.GetArrayFromImage(image_rs).astype(np.float32)
    img_arr = ensure_3d_array(img_arr)

    # --------------------------------------------------------
    # 3. Ventana HU y normalización a [-1, 1]
    # --------------------------------------------------------
    img_arr = normalize_hu_to_minus1_1(
        img_arr,
        hu_min=HU_MIN,
        hu_max=HU_MAX,
    )

    # --------------------------------------------------------
    # 4. Center crop / pad a shape fijo
    # --------------------------------------------------------
    img_arr = center_crop_or_pad_3d(
        img_arr,
        target_shape=FULL_SHAPE,
    )

    return img_arr.astype(np.float32)


def preprocess_medical_image_to_npz(
    input_path,
    output_path,
    patient_id=None,
):
    """
    Lee una imagen médica, aplica el preprocesado del proyecto y guarda un .npz.

    Parámetros:
        input_path: ruta a .nii, .nii.gz, .dcm o carpeta DICOM
        output_path: ruta de salida .npz
        patient_id: identificador opcional del paciente

    Guarda:
        .npz con claves:
            - image
            - patient_id

    Devuelve:
        diccionario con información del preprocesado
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Leer imagen médica
    # --------------------------------------------------------
    image_sitk = read_medical_image(input_path)

    # --------------------------------------------------------
    # 2. Preprocesar
    # --------------------------------------------------------
    img_arr = preprocess_sitk_image_to_array(image_sitk)

    # --------------------------------------------------------
    # 3. ID de paciente
    # --------------------------------------------------------
    if patient_id is None:
        if input_path.is_dir():
            patient_id = input_path.name
        else:
            patient_id = input_path.name.replace(".nii.gz", "").replace(".nii", "")

    # --------------------------------------------------------
    # 4. Guardar .npz temporal compatible con la pipeline
    # --------------------------------------------------------
    np.savez_compressed(
        output_path,
        image=img_arr.astype(np.float16),
        patient_id=str(patient_id),
    )

    info = {
        "patient_id": str(patient_id),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "shape": tuple(img_arr.shape),
        "dtype": str(img_arr.dtype),
        "min": float(img_arr.min()),
        "max": float(img_arr.max()),
        "mean": float(img_arr.mean()),
        "target_spacing": TARGET_SPACING,
        "target_shape": FULL_SHAPE,
        "hu_min": HU_MIN,
        "hu_max": HU_MAX,
    }

    return info


# ============================================================
# PRUEBA DESDE TERMINAL
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocesa una imagen médica y la convierte a .npz."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta a archivo NIfTI, DICOM o carpeta DICOM.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta de salida .npz.",
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="ID opcional del paciente.",
    )

    args = parser.parse_args()

    result = preprocess_medical_image_to_npz(
        input_path=args.input,
        output_path=args.output,
        patient_id=args.patient_id,
    )

    print("==================================================")
    print("PREPROCESADO COMPLETADO")
    print("==================================================")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("==================================================")