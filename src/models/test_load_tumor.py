# ============================================================
# PRUEBA DE CARGA DEL MODELO DE SEGMENTACIÓN TUMORAL
# ============================================================

from src.models.tumor_segmentation import load_tumor_model


def main():
    print("Cargando modelo de segmentación tumoral...")
    model = load_tumor_model()
    print("Modelo cargado correctamente.")
    print(type(model))


if __name__ == "__main__":
    main()