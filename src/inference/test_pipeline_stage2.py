# ============================================================
# PRUEBA DE PIPELINE STAGE 2
# TAC -> HÍGADO -> SANO/PATOLÓGICO -> TUMOR
# ============================================================

from src.config import NPZ_DIR
from src.inference.data_loader import get_first_npz
from src.inference.pipeline import analizar_paciente_stage2


def main():
    path_npz = get_first_npz(NPZ_DIR)

    report = analizar_paciente_stage2(path_npz)

    print()
    print("RESUMEN FINAL")
    print("-------------")
    print(f"Paciente: {report['patient_id']}")

    status = report["healthy_pathological_classification"]
    tumor = report["tumor_segmentation"]

    print("Resultado sano/patológico:", status["prediction"])
    print("Probabilidad patológico:", status["probability_pathological"])

    print("Segmentación tumoral ejecutada:", tumor["executed"])

    if tumor["executed"]:
        print("Tumor detectado:", tumor["tumor_detected"])
        print("Voxeles tumor:", tumor["positive_voxels"])


if __name__ == "__main__":
    main()