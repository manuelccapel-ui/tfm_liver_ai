# ============================================================
# PRUEBA DE INFERENCIA DEL MODELO DE HÍGADO
# ============================================================

from pathlib import Path
import numpy as np

from src.config import NPZ_DIR, OUTPUTS_DIR
from src.inference.data_loader import get_first_npz, load_patient_npz
from src.models.liver_segmentation import load_liver_model, predict_liver_mask


def dice_score(mask_pred, mask_true, eps=1e-8):
    """
    Calcula Dice entre dos máscaras binarias.
    Solo se usa aquí como comprobación, no como parte de la inferencia real.
    """
    mask_pred = (mask_pred > 0).astype(np.uint8)
    mask_true = (mask_true > 0).astype(np.uint8)

    inter = np.sum(mask_pred * mask_true)
    total = np.sum(mask_pred) + np.sum(mask_true)

    return (2.0 * inter + eps) / (total + eps)


def main():
    print("==================================================")
    print("PRUEBA DE INFERENCIA: SEGMENTACIÓN HEPÁTICA")
    print("==================================================")

    # --------------------------------------------------------
    # 1. Seleccionar primer paciente .npz
    # --------------------------------------------------------
    path_npz = get_first_npz(NPZ_DIR)
    print(f"Paciente seleccionado:")
    print(path_npz)
    print()

    # --------------------------------------------------------
    # 2. Cargar solo el TAC
    # --------------------------------------------------------
    patient = load_patient_npz(path_npz)
    image = patient["image"]

    print("TAC cargado:")
    print(f"  shape: {image.shape}")
    print(f"  min: {image.min():.4f}")
    print(f"  max: {image.max():.4f}")
    print()

    # --------------------------------------------------------
    # 3. Cargar modelo
    # --------------------------------------------------------
    print("Cargando modelo de hígado...")
    model = load_liver_model()
    print("Modelo cargado.")
    print()

    # --------------------------------------------------------
    # 4. Predecir máscara
    # --------------------------------------------------------
    print("Prediciendo máscara hepática...")
    liver_mask_pred, liver_prob = predict_liver_mask(
        model=model,
        image=image,
        threshold=0.5,
    )
    print("Predicción terminada.")
    print()

    print("Máscara predicha:")
    print(f"  shape: {liver_mask_pred.shape}")
    print(f"  voxeles positivos: {int(liver_mask_pred.sum())}")
    print(f"  prob min: {liver_prob.min():.4f}")
    print(f"  prob max: {liver_prob.max():.4f}")
    print(f"  prob mean: {liver_prob.mean():.4f}")
    print()

    # --------------------------------------------------------
    # 5. Guardar resultados
    # --------------------------------------------------------
    output_dir = OUTPUTS_DIR / patient["patient_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    path_mask = output_dir / "liver_mask_pred.npy"
    path_prob = output_dir / "liver_prob_pred.npy"

    np.save(path_mask, liver_mask_pred)
    np.save(path_prob, liver_prob)

    print("Resultados guardados en:")
    print(f"  {path_mask}")
    print(f"  {path_prob}")
    print()

    # --------------------------------------------------------
    # 6. Comprobación opcional contra liver_mask real
    # --------------------------------------------------------
    # Ojo: esto no forma parte de la inferencia real.
    # Solo lo usamos porque tus .npz tienen liver_mask y nos sirve
    # para comprobar que el modelo está funcionando bien.
    data_original = np.load(path_npz, allow_pickle=True)

    if "liver_mask" in data_original.files:
        liver_mask_true = data_original["liver_mask"].astype(np.uint8)
        dice = dice_score(liver_mask_pred, liver_mask_true)

        print("Comprobación contra máscara real del .npz:")
        print(f"  Dice hígado: {dice:.4f}")
        print()

    print("==================================================")
    print("PRUEBA FINALIZADA")
    print("==================================================")


if __name__ == "__main__":
    main()