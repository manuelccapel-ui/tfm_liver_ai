# ============================================================
# PRUEBA DE CARGA DEL MODELO SANO / PATOLÓGICO
# ============================================================

from src.models.healthy_pathological import load_healthy_pathological_model


def main():
    print("Cargando modelo sano/patológico...")
    model = load_healthy_pathological_model()
    print("Modelo cargado correctamente.")
    print(type(model))


if __name__ == "__main__":
    main()