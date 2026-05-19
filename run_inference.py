# ============================================================
# SCRIPT PRINCIPAL DE INFERENCIA
# TFM LIVER AI
#
# Uso:
#   python run_inference.py --input ruta_paciente.npz
#
# También permite:
#   python run_inference.py --first
# ============================================================

import argparse
from pathlib import Path

from src.config import NPZ_DIR, OUTPUTS_DIR
from src.inference.data_loader import get_first_npz
from src.inference.pipeline import (
    analizar_paciente_stage1,
    analizar_paciente_stage2,
    analizar_paciente_stage3,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de inferencia para detección y clasificación hepática."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--input",
        type=str,
        help="Ruta al archivo .npz del paciente.",
    )

    group.add_argument(
        "--first",
        action="store_true",
        help="Usar automáticamente el primer .npz encontrado en NPZ_DIR.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Carpeta de salida. Si no se indica, se usa outputs/patient_id.",
    )

    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help=(
            "Stage de la pipeline: "
            "1 = hígado + sano/patológico; "
            "2 = añade tumor; "
            "3 = añade HCC/CRLM."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------
    # 1. Seleccionar paciente
    # --------------------------------------------------------
    if args.first:
        path_npz = get_first_npz(NPZ_DIR)
    else:
        path_npz = Path(args.input)

    if not path_npz.exists():
        raise FileNotFoundError(f"No existe el archivo indicado: {path_npz}")

    # --------------------------------------------------------
    # 2. Carpeta de salida
    # --------------------------------------------------------
    output_dir = Path(args.output) if args.output is not None else None

    # --------------------------------------------------------
    # 3. Ejecutar stage seleccionado
    # --------------------------------------------------------
    if args.stage == 1:
        report = analizar_paciente_stage1(
            path_npz=path_npz,
            output_dir=output_dir,
        )

    elif args.stage == 2:
        report = analizar_paciente_stage2(
            path_npz=path_npz,
            output_dir=output_dir,
        )

    elif args.stage == 3:
        report = analizar_paciente_stage3(
            path_npz=path_npz,
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Stage no válido: {args.stage}")

    # --------------------------------------------------------
    # 4. Resumen final
    # --------------------------------------------------------
    print()
    print("==================================================")
    print("RESUMEN FINAL DE INFERENCIA")
    print("==================================================")
    print(f"Paciente: {report['patient_id']}")

    status = report.get("healthy_pathological_classification", {})
    print("Resultado sano/patológico:", status.get("prediction"))
    print("Probabilidad patológico:", status.get("probability_pathological"))

    tumor = report.get("tumor_segmentation", {})
    print("Segmentación tumoral ejecutada:", tumor.get("executed"))

    if tumor.get("executed", False):
        print("Tumor detectado:", tumor.get("tumor_detected"))
        print("Voxeles tumor:", tumor.get("positive_voxels"))

    hcc_crlm = report.get("hcc_crlm_classification", {})
    if hcc_crlm:
        print("Clasificación HCC/CRLM ejecutada:", hcc_crlm.get("executed"))

        if hcc_crlm.get("executed", False):
            print("Predicción final:", hcc_crlm.get("prediction"))
            print("Probabilidad CRLM:", hcc_crlm.get("probability_crlm"))
            print("Probabilidad HCC:", hcc_crlm.get("probability_hcc"))

    print("==================================================")


if __name__ == "__main__":
    main()