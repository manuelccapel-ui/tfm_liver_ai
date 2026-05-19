# ============================================================
# PIPELINE PRINCIPAL DE INFERENCIA
#
# Stage 1:
#   TAC -> Segmentación hígado -> Clasificación sano/patológico
#
# Stage 2:
#   TAC -> Segmentación hígado -> Clasificación sano/patológico
#       -> si es patológico, segmentación tumoral
# ============================================================

import json
from pathlib import Path
import numpy as np
from src.utils.visualization import save_overview_png
from src.config import (
    OUTPUTS_DIR,
    THRESHOLD_SANO_PATOLOGICO,
    THRESHOLD_TUMOR,
    MIN_TUMOR_COMPONENT_SIZE,
)

from src.models.hcc_crlm_classifier import (
    load_hcc_crlm_model,
    predict_hcc_crlm,
)

from src.inference.data_loader import load_patient_npz

from src.models.liver_segmentation import (
    load_liver_model,
    predict_liver_mask,
)

from src.models.healthy_pathological import (
    load_healthy_pathological_model,
    predict_healthy_pathological,
)

from src.models.tumor_segmentation import (
    load_tumor_model,
    predict_tumor_mask,
)


# ============================================================
# FUNCIÓN AUXILIAR PARA GUARDAR JSON
# ============================================================

def save_json(data, path_json):
    """
    Guarda un diccionario como archivo JSON.
    """

    path_json = Path(path_json)
    path_json.parent.mkdir(parents=True, exist_ok=True)

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ============================================================
# STAGE 1
# TAC -> HÍGADO -> SANO/PATOLÓGICO
# ============================================================

def analizar_paciente_stage1(path_npz, output_dir=None):
    """
    Primera versión de la pipeline:

        1. Carga TAC
        2. Segmenta hígado
        3. Clasifica sano/patológico
        4. Guarda resultados básicos

    Parámetros:
        path_npz: ruta al archivo .npz del paciente
        output_dir: carpeta donde guardar resultados

    Devuelve:
        report: diccionario con resultados
    """

    # --------------------------------------------------------
    # 1. Cargar paciente
    # --------------------------------------------------------
    patient = load_patient_npz(path_npz)

    patient_id = patient["patient_id"]
    image = patient["image"]

    if output_dir is None:
        output_dir = OUTPUTS_DIR / patient_id
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("ANÁLISIS DE PACIENTE - STAGE 1")
    print("==================================================")
    print(f"Paciente: {patient_id}")
    print(f"Archivo: {patient['path']}")
    print(f"Output: {output_dir}")
    print()

    # --------------------------------------------------------
    # 2. Cargar modelo de hígado
    # --------------------------------------------------------
    print("[1/4] Cargando modelo de segmentación hepática...")
    model_liver = load_liver_model()
    print("      Modelo de hígado cargado.")

    # --------------------------------------------------------
    # 3. Predecir máscara hepática
    # --------------------------------------------------------
    print("[2/4] Prediciendo máscara hepática...")
    liver_mask_pred, liver_prob = predict_liver_mask(
        model=model_liver,
        image=image,
        threshold=0.5,
    )

    liver_voxels = int(liver_mask_pred.sum())

    print("      Segmentación hepática terminada.")
    print(f"      Voxeles positivos hígado: {liver_voxels}")

    # --------------------------------------------------------
    # 4. Cargar modelo sano/patológico
    # --------------------------------------------------------
    print("[3/4] Cargando modelo sano/patológico...")
    model_status = load_healthy_pathological_model()
    print("      Modelo sano/patológico cargado.")

    # --------------------------------------------------------
    # 5. Clasificar sano/patológico
    # --------------------------------------------------------
    print("[4/4] Clasificando sano/patológico...")

    prediction_status, prob_pathological = predict_healthy_pathological(
        model=model_status,
        image=image,
        liver_mask=liver_mask_pred,
        threshold=THRESHOLD_SANO_PATOLOGICO,
    )

    print("      Clasificación terminada.")
    print(f"      Predicción: {prediction_status}")
    print(f"      Probabilidad patológico: {prob_pathological:.4f}")

    # --------------------------------------------------------
    # 6. Guardar resultados
    # --------------------------------------------------------
    path_liver_mask = output_dir / "liver_mask_pred.npy"
    path_liver_prob = output_dir / "liver_prob_pred.npy"
    path_report = output_dir / "report_stage1.json"

    np.save(path_liver_mask, liver_mask_pred)
    np.save(path_liver_prob, liver_prob)

    report = {
        "patient_id": patient_id,
        "input_path": str(patient["path"]),
        "stage": "liver_segmentation_and_healthy_pathological_classification",
        "liver_segmentation": {
            "status": "ok",
            "threshold": 0.5,
            "positive_voxels": liver_voxels,
            "mask_path": str(path_liver_mask),
            "probability_map_path": str(path_liver_prob),
        },
        "healthy_pathological_classification": {
            "prediction": prediction_status,
            "threshold": THRESHOLD_SANO_PATOLOGICO,
            "probability_pathological": float(prob_pathological),
        },
        "warning": (
            "Resultado generado por un prototipo experimental. "
            "No validado para uso clínico."
        ),
    }

    save_json(report, path_report)

    print()
    print("Resultados guardados:")
    print(f"  {path_liver_mask}")
    print(f"  {path_liver_prob}")
    print(f"  {path_report}")
    print("==================================================")

    return report


# ============================================================
# STAGE 2
# TAC -> HÍGADO -> SANO/PATOLÓGICO -> TUMOR
# ============================================================

def analizar_paciente_stage2(path_npz, output_dir=None):
    """
    Segunda versión de la pipeline:

        1. Carga TAC
        2. Segmenta hígado
        3. Clasifica sano/patológico
        4. Si es patológico, segmenta tumor
        5. Guarda resultados

    Todavía no clasifica HCC/CRLM.
    """

    # --------------------------------------------------------
    # 1. Cargar paciente
    # --------------------------------------------------------
    patient = load_patient_npz(path_npz)

    patient_id = patient["patient_id"]
    image = patient["image"]

    if output_dir is None:
        output_dir = OUTPUTS_DIR / patient_id
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("ANÁLISIS DE PACIENTE - STAGE 2")
    print("==================================================")
    print(f"Paciente: {patient_id}")
    print(f"Archivo: {patient['path']}")
    print(f"Output: {output_dir}")
    print()

    # --------------------------------------------------------
    # 2. Cargar modelo de hígado
    # --------------------------------------------------------
    print("[1/6] Cargando modelo de segmentación hepática...")
    model_liver = load_liver_model()
    print("      Modelo de hígado cargado.")

    # --------------------------------------------------------
    # 3. Predecir máscara hepática
    # --------------------------------------------------------
    print("[2/6] Prediciendo máscara hepática...")
    liver_mask_pred, liver_prob = predict_liver_mask(
        model=model_liver,
        image=image,
        threshold=0.5,
    )

    liver_voxels = int(liver_mask_pred.sum())

    print("      Segmentación hepática terminada.")
    print(f"      Voxeles positivos hígado: {liver_voxels}")

    # --------------------------------------------------------
    # 4. Cargar modelo sano/patológico
    # --------------------------------------------------------
    print("[3/6] Cargando modelo sano/patológico...")
    model_status = load_healthy_pathological_model()
    print("      Modelo sano/patológico cargado.")

    # --------------------------------------------------------
    # 5. Clasificar sano/patológico
    # --------------------------------------------------------
    print("[4/6] Clasificando sano/patológico...")

    prediction_status, prob_pathological = predict_healthy_pathological(
        model=model_status,
        image=image,
        liver_mask=liver_mask_pred,
        threshold=THRESHOLD_SANO_PATOLOGICO,
    )

    print("      Clasificación terminada.")
    print(f"      Predicción: {prediction_status}")
    print(f"      Probabilidad patológico: {prob_pathological:.4f}")

    # --------------------------------------------------------
    # 6. Guardar resultados de hígado
    # --------------------------------------------------------
    path_liver_mask = output_dir / "liver_mask_pred.npy"
    path_liver_prob = output_dir / "liver_prob_pred.npy"

    np.save(path_liver_mask, liver_mask_pred)
    np.save(path_liver_prob, liver_prob)

    tumor_segmentation_result = {
        "executed": False,
        "reason": "Paciente clasificado como sano. No se ejecuta segmentación tumoral.",
    }

    # --------------------------------------------------------
    # 7. Si es patológico, segmentar tumor
    # --------------------------------------------------------
    if prediction_status == "Patológico":
        print("[5/6] Cargando modelo de segmentación tumoral...")
        model_tumor = load_tumor_model()
        print("      Modelo tumoral cargado.")

        print("[6/6] Prediciendo máscara tumoral con sliding window...")
        print("      Esto puede tardar varios minutos si estás usando CPU.")

        tumor_mask_pred, tumor_prob = predict_tumor_mask(
            model=model_tumor,
            image=image,
            liver_mask=liver_mask_pred,
            threshold=THRESHOLD_TUMOR,
            min_size=MIN_TUMOR_COMPONENT_SIZE,
        )

        tumor_voxels = int(tumor_mask_pred.sum())

        path_tumor_mask = output_dir / "tumor_mask_pred.npy"
        path_tumor_prob = output_dir / "tumor_prob_pred.npy"

        np.save(path_tumor_mask, tumor_mask_pred)
        np.save(path_tumor_prob, tumor_prob)

        tumor_segmentation_result = {
            "executed": True,
            "status": "ok",
            "threshold": THRESHOLD_TUMOR,
            "min_component_size": MIN_TUMOR_COMPONENT_SIZE,
            "positive_voxels": tumor_voxels,
            "tumor_detected": bool(tumor_voxels > 0),
            "mask_path": str(path_tumor_mask),
            "probability_map_path": str(path_tumor_prob),
        }

        print("      Segmentación tumoral terminada.")
        print(f"      Voxeles positivos tumor: {tumor_voxels}")

    else:
        print("[5/6] Paciente clasificado como sano.")
        print("      No se ejecuta segmentación tumoral.")

    # --------------------------------------------------------
    # 8. Guardar reporte final Stage 2
    # --------------------------------------------------------
    path_report = output_dir / "report_stage2.json"

    report = {
        "patient_id": patient_id,
        "input_path": str(patient["path"]),
        "stage": "liver_segmentation_healthy_pathological_and_tumor_segmentation",
        "liver_segmentation": {
            "status": "ok",
            "threshold": 0.5,
            "positive_voxels": liver_voxels,
            "mask_path": str(path_liver_mask),
            "probability_map_path": str(path_liver_prob),
        },
        "healthy_pathological_classification": {
            "prediction": prediction_status,
            "threshold": THRESHOLD_SANO_PATOLOGICO,
            "probability_pathological": float(prob_pathological),
        },
        "tumor_segmentation": tumor_segmentation_result,
        "warning": (
            "Resultado generado por un prototipo experimental. "
            "No validado para uso clínico."
        ),
    }

    save_json(report, path_report)

    print()
    print("Resultados guardados:")
    print(f"  {path_liver_mask}")
    print(f"  {path_liver_prob}")
    print(f"  {path_report}")

    if tumor_segmentation_result.get("executed", False):
        print(f"  {tumor_segmentation_result['mask_path']}")
        print(f"  {tumor_segmentation_result['probability_map_path']}")

    print("==================================================")

    return report

# ============================================================
# STAGE 3
# TAC -> HÍGADO -> SANO/PATOLÓGICO -> TUMOR -> HCC/CRLM
# ============================================================

def analizar_paciente_stage3(path_npz, output_dir=None):
    """
    Tercera versión de la pipeline:

        1. Ejecuta Stage 2:
            TAC -> hígado -> sano/patológico -> tumor
        2. Si hay tumor detectado, clasifica HCC/CRLM
        3. Genera report_stage3.json
        4. Genera report.json final
        5. Genera overview.png
    """

    # --------------------------------------------------------
    # 1. Ejecutar Stage 2 completo
    # --------------------------------------------------------
    report = analizar_paciente_stage2(
        path_npz=path_npz,
        output_dir=output_dir,
    )

    patient_id = report["patient_id"]

    if output_dir is None:
        output_dir = OUTPUTS_DIR / patient_id
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("==================================================")
    print("ANÁLISIS DE PACIENTE - STAGE 3")
    print("==================================================")
    print("Añadiendo clasificación HCC/CRLM...")
    print()

    # --------------------------------------------------------
    # 2. Cargar datos comunes
    # --------------------------------------------------------
    patient = load_patient_npz(path_npz)
    image = patient["image"]

    liver_mask = np.load(report["liver_segmentation"]["mask_path"])

    tumor_info = report["tumor_segmentation"]
    tumor_mask = None

    if tumor_info.get("executed", False):
        tumor_mask_path = tumor_info.get("mask_path")
        if tumor_mask_path is not None:
            tumor_mask = np.load(tumor_mask_path)

    hcc_crlm_result = {
        "executed": False,
        "reason": "No se ejecuta clasificación HCC/CRLM porque no hay tumor detectado.",
    }

    # --------------------------------------------------------
    # 3. Solo clasificar si se detectó tumor
    # --------------------------------------------------------
    if tumor_info.get("executed", False) and tumor_info.get("tumor_detected", False):

        print("[Stage 3] TAC, hígado y tumor cargados.")

        print("[Stage 3] Cargando modelo HCC/CRLM...")
        model_hcc_crlm = load_hcc_crlm_model()
        print("          Modelo HCC/CRLM cargado.")

        print("[Stage 3] Clasificando tipo tumoral...")

        prediction_tumor_type, prob_crlm = predict_hcc_crlm(
            model=model_hcc_crlm,
            image=image,
            liver_mask=liver_mask,
            tumor_mask=tumor_mask,
        )

        prob_hcc = 1.0 - prob_crlm

        hcc_crlm_result = {
            "executed": True,
            "status": "ok",
            "prediction": prediction_tumor_type,
            "probability_crlm": float(prob_crlm),
            "probability_hcc": float(prob_hcc),
            "threshold": 0.5,
            "interpretation": (
                "La probabilidad corresponde a la clase CRLM. "
                "Si probability_crlm >= 0.5, la predicción es CRLM; "
                "en caso contrario, HCC."
            ),
        }

        print("          Clasificación HCC/CRLM terminada.")
        print(f"          Predicción: {prediction_tumor_type}")
        print(f"          Probabilidad CRLM: {prob_crlm:.4f}")
        print(f"          Probabilidad HCC: {prob_hcc:.4f}")

    else:
        print("[Stage 3] No se clasifica HCC/CRLM porque no hay tumor detectado.")

    # --------------------------------------------------------
    # 4. Actualizar report
    # --------------------------------------------------------
    report["stage"] = "full_pipeline_liver_status_tumor_and_hcc_crlm"
    report["hcc_crlm_classification"] = hcc_crlm_result

    # --------------------------------------------------------
    # 5. Guardar overview.png
    # --------------------------------------------------------
    path_overview = output_dir / "overview.png"

    save_overview_png(
        image=image,
        liver_mask=liver_mask,
        tumor_mask=tumor_mask,
        output_path=path_overview,
        title=f"Paciente: {patient_id}",
    )

    # --------------------------------------------------------
    # 6. Guardar report final y report Stage 3
    # --------------------------------------------------------
    path_report_final = output_dir / "report.json"
    path_report_stage3 = output_dir / "report_stage3.json"

    report["final_outputs"] = {
        "report_path": str(path_report_final),
        "stage3_report_path": str(path_report_stage3),
        "overview_path": str(path_overview),
    }

    save_json(report, path_report_final)
    save_json(report, path_report_stage3)

    print()
    print("Reportes finales guardados en:")
    print(f"  {path_report_stage3}")
    print(f"  {path_report_final}")
    print(f"  {path_overview}")
    print("==================================================")

    return report