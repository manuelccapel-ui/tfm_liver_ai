# ============================================================
# INSPECCIÓN DE ARCHIVOS .NPZ
# ============================================================

from pathlib import Path
import numpy as np

from src.config import NPZ_DIR


def describe_array(name, arr):
    """
    Muestra información básica de un array numpy.
    """
    print(f"  - {name}")
    print(f"      shape: {arr.shape}")
    print(f"      dtype: {arr.dtype}")

    if np.issubdtype(arr.dtype, np.number):
        print(f"      min: {np.min(arr):.4f}")
        print(f"      max: {np.max(arr):.4f}")
        print(f"      mean: {np.mean(arr):.4f}")


def inspect_npz_file(path_npz: Path):
    """
    Abre un archivo .npz y muestra sus claves y dimensiones.
    """
    print("==================================================")
    print(f"INSPECCIONANDO ARCHIVO:")
    print(path_npz)
    print("==================================================")

    data = np.load(path_npz, allow_pickle=True)

    print("Claves encontradas:")
    for key in data.files:
        print(f" - {key}")

    print()
    print("Contenido:")
    for key in data.files:
        arr = data[key]
        describe_array(key, arr)

    print("==================================================")


def main():
    print("==================================================")
    print("BUSCANDO ARCHIVOS .NPZ")
    print("==================================================")
    print(f"Carpeta NPZ_DIR:")
    print(NPZ_DIR)
    print()

    npz_files = sorted(NPZ_DIR.glob("*.npz"))

    print(f"Número de archivos .npz encontrados: {len(npz_files)}")
    print()

    if len(npz_files) == 0:
        print("ERROR: No se ha encontrado ningún archivo .npz.")
        return

    print("Primeros archivos encontrados:")
    for i, path in enumerate(npz_files[:10]):
        print(f"{i + 1}. {path.name}")

    print()

    # Inspeccionamos el primer archivo
    inspect_npz_file(npz_files[0])


if __name__ == "__main__":
    main()