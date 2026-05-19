# ============================================================
# PRUEBA DE PIPELINE STAGE 3
# TAC -> HÍGADO -> SANO/PATOLÓGICO -> TUMOR -> HCC/CRLM
# ============================================================

from src.config import NPZ_DIR
from src.inference.data_loader import get_first_npz
from src.inference.pipeline import analizar_paciente_stage3


def main():
    path_npz = get_first_npz(NPZ_DIR)

    report = analizar_paciente_stage3(path_npz)

    print()
    print("RESUMEN FINAL")
    print("-------------")
    print(f"Paciente: {report['patient_id']}")

    status = report["healthy_pathological_classification"]
    tumor = report["tumor_segmentation"]
    hcc_crlm = report["hcc_crlm_classification"]

    print("Resultado sano/patológico:", status["prediction"])
    print("Probabilidad patológico:", status["probability_pathological"])

    print("Segmentación tumoral ejecutada:", tumor["executed"])

    if tumor["executed"]:
        print("Tumor detectado:", tumor["tumor_detected"])
        print("Voxeles tumor:", tumor["positive_voxels"])

    print("Clasificación HCC/CRLM ejecutada:", hcc_crlm["executed"])

    if hcc_crlm["executed"]:
        print("Predicción final:", hcc_crlm["prediction"])
        print("Probabilidad CRLM:", hcc_crlm["probability_crlm"])
        print("Probabilidad HCC:", hcc_crlm["probability_hcc"])


if __name__ == "__main__":
    main()