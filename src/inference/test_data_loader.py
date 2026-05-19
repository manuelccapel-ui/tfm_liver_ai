# ============================================================
# PRUEBA DEL CARGADOR DE PACIENTES
# ============================================================

from src.config import NPZ_DIR
from src.inference.data_loader import get_first_npz, load_patient_npz, summarize_patient


def main():
    path_npz = get_first_npz(NPZ_DIR)
    patient = load_patient_npz(path_npz)
    summarize_patient(patient)


if __name__ == "__main__":
    main()