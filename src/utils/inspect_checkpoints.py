# ============================================================
# INSPECCIÓN DE CHECKPOINTS .PTH
# ============================================================

import torch

from src.config import (
    PATH_MODELO_LIVER,
    PATH_MODELO_SANO,
    PATH_MODELO_TUMOR,
    PATH_MODELO_HCC_CRLM,
)


def is_state_dict(obj):
    """
    Comprueba si un objeto parece un state_dict de PyTorch.
    """
    return isinstance(obj, dict) and all(
        torch.is_tensor(v) for v in obj.values()
    )


def print_state_dict_summary(state_dict, max_items=25):
    """
    Muestra las primeras capas de un state_dict.
    """
    keys = list(state_dict.keys())

    print(f"Número de tensores/capas: {len(keys)}")
    print("Primeras capas:")

    for key in keys[:max_items]:
        tensor = state_dict[key]
        print(f"  - {key}: {tuple(tensor.shape)}")

    if len(keys) > max_items:
        print(f"  ... y {len(keys) - max_items} capas más")


def inspect_checkpoint(path, name):
    """
    Inspecciona un archivo .pth sin cargarlo todavía en una arquitectura.
    """

    print("\n" + "=" * 80)
    print(f"INSPECCIONANDO: {name}")
    print("=" * 80)
    print(f"Ruta: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        print("ERROR al cargar el checkpoint:")
        print(e)
        return

    print()
    print(f"Tipo cargado: {type(checkpoint)}")

    # Caso 1: el archivo es directamente un state_dict
    if is_state_dict(checkpoint):
        print()
        print("Formato detectado: state_dict directo")
        print_state_dict_summary(checkpoint)
        return

    # Caso 2: es un diccionario con varias claves
    if isinstance(checkpoint, dict):
        print()
        print("Formato detectado: diccionario/checkpoint")
        print("Claves principales:")

        for key in checkpoint.keys():
            value = checkpoint[key]
            if torch.is_tensor(value):
                print(f"  - {key}: tensor {tuple(value.shape)}")
            else:
                print(f"  - {key}: {type(value)}")

        # Buscar posibles state_dicts dentro
        possible_keys = [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "weights",
        ]

        for key in possible_keys:
            if key in checkpoint and is_state_dict(checkpoint[key]):
                print()
                print(f"State_dict encontrado dentro de la clave: '{key}'")
                print_state_dict_summary(checkpoint[key])
                return

        print()
        print("No se ha encontrado una clave típica de state_dict.")
        return

    # Caso 3: parece un modelo guardado completo
    print()
    print("Formato no estándar o modelo completo guardado con torch.save(model).")
    print("Tipo del objeto:")
    print(type(checkpoint))


def main():
    inspect_checkpoint(PATH_MODELO_LIVER, "Segmentación hígado")
    inspect_checkpoint(PATH_MODELO_SANO, "Clasificación sano/patológico")
    inspect_checkpoint(PATH_MODELO_TUMOR, "Segmentación tumor")
    inspect_checkpoint(PATH_MODELO_HCC_CRLM, "Clasificación HCC/CRLM")


if __name__ == "__main__":
    main()