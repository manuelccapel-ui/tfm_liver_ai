# ============================================================
# PRUEBA DE CARGA DEL MODELO DE HÍGADO
# ============================================================

from src.models.liver_segmentation import load_liver_model


def main():
    print("Cargando modelo de segmentación hepática...")
    model = load_liver_model()
    print("Modelo cargado correctamente.")
    print(type(model))


if __name__ == "__main__":
    main()