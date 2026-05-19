# ============================================================
# PRUEBA DE PIPELINE STAGE 1
# TAC -> HÍGADO -> SANO/PATOLÓGICO
# ============================================================

from src.config import NPZ_DIR
from src.inference.data_loader import get_first_npz
from src.inference.pipeline import analizar_paciente_stage1


def main():
    path_npz = get_first_npz(NPZ_DIR)

    report = analizar_paciente_stage1(path_npz)

    print()
    print("RESUMEN FINAL")
    print("-------------")
    print(f"Paciente: {report['patient_id']}")
    print(
        "Resultado sano/patológico:",
        report["healthy_pathological_classification"]["prediction"],
    )
    print(
        "Probabilidad patológico:",
        report["healthy_pathological_classification"]["probability_pathological"],
    )


if __name__ == "__main__":
    main()