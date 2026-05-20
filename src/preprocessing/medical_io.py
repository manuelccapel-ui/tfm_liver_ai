# ============================================================
# LECTURA DE IMÁGENES MÉDICAS
# NIfTI / DICOM
# ============================================================

from pathlib import Path
import SimpleITK as sitk


def is_nifti_file(path):
    """
    Comprueba si un archivo es NIfTI.
    Acepta:
        - .nii
        - .nii.gz
    """

    path = Path(path)
    name = path.name.lower()

    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_dicom_file(path):
    """
    Comprueba de forma sencilla si un archivo podría ser DICOM.
    """

    path = Path(path)
    suffix = path.suffix.lower()

    return suffix in [".dcm", ".dicom"]


def read_nifti_image(path):
    """
    Lee un archivo NIfTI usando SimpleITK.

    Devuelve:
        image_sitk: imagen SimpleITK
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo NIfTI: {path}")

    if not is_nifti_file(path):
        raise ValueError(f"El archivo no parece NIfTI: {path}")

    image_sitk = sitk.ReadImage(str(path))

    return image_sitk


def read_dicom_series(folder):
    """
    Lee una serie DICOM desde una carpeta.

    La carpeta debe contener cortes pertenecientes a una misma serie.
    Si hay varias series DICOM dentro de la carpeta, se selecciona
    automáticamente la primera serie detectada.

    Devuelve:
        image_sitk: volumen 3D SimpleITK
    """

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta DICOM: {folder}")

    if not folder.is_dir():
        raise ValueError(f"La ruta DICOM debe ser una carpeta: {folder}")

    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(str(folder))

    if not series_ids:
        raise ValueError(
            f"No se han encontrado series DICOM válidas en la carpeta: {folder}"
        )

    # En esta primera versión seleccionamos la primera serie.
    # Para uso clínico real habría que permitir seleccionar serie/fase.
    series_id = series_ids[0]

    dicom_names = reader.GetGDCMSeriesFileNames(str(folder), series_id)

    if len(dicom_names) == 0:
        raise ValueError(f"La serie DICOM detectada no contiene archivos: {folder}")

    reader.SetFileNames(dicom_names)
    image_sitk = reader.Execute()

    return image_sitk


def read_single_dicom_file(path):
    """
    Lee un único archivo DICOM.

    Nota:
        Esto puede devolver una imagen 2D si se proporciona un único corte.
        Para inferencia real se recomienda subir una carpeta con una serie
        DICOM completa.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo DICOM: {path}")

    image_sitk = sitk.ReadImage(str(path))

    return image_sitk


def read_medical_image(path):
    """
    Lee una imagen médica desde:
        - archivo NIfTI (.nii o .nii.gz)
        - carpeta con serie DICOM
        - archivo DICOM individual

    Devuelve:
        image_sitk: imagen SimpleITK
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta indicada: {path}")

    if path.is_dir():
        return read_dicom_series(path)

    if is_nifti_file(path):
        return read_nifti_image(path)

    if is_dicom_file(path):
        return read_single_dicom_file(path)

    # Último intento: dejar que SimpleITK intente leerlo.
    # Esto permite aceptar DICOM sin extensión, si SimpleITK lo reconoce.
    try:
        return sitk.ReadImage(str(path))
    except Exception as e:
        raise ValueError(
            f"No se ha podido leer la imagen médica: {path}. "
            f"Formatos esperados: .nii, .nii.gz, .dcm, .dicom o carpeta DICOM."
        ) from e