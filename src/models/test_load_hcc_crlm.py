# ============================================================
# PRUEBA DE CARGA DEL MODELO HCC / CRLM
# ============================================================

from src.models.hcc_crlm_classifier import load_hcc_crlm_model


def main():
    print("Cargando modelo HCC/CRLM...")
    model = load_hcc_crlm_model()
    print("Modelo cargado correctamente.")
    print(type(model))


if __name__ == "__main__":
    main()