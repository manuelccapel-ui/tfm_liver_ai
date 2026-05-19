# ============================================================
# APP STREAMLIT - TFM LIVER AI
# ============================================================

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st


# ============================================================
# CONFIGURAR RUTA DEL PROYECTO
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from src.config import NPZ_DIR
from src.inference.pipeline import analizar_paciente_stage3
from src.inference.data_loader import load_patient_npz


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="TFM Liver AI",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# FUNCIONES AUXILIARES DE VISUALIZACIÓN
# ============================================================

def best_liver_slice(liver_mask):
    """
    Devuelve el corte axial donde más hígado aparece.
    """
    if liver_mask is None or liver_mask.sum() == 0:
        return 0

    return int(np.argmax(liver_mask.sum(axis=(1, 2))))


def best_tumor_slice(tumor_mask):
    """
    Devuelve el corte axial donde más tumor aparece.
    """
    if tumor_mask is None or tumor_mask.sum() == 0:
        return None

    return int(np.argmax(tumor_mask.sum(axis=(1, 2))))


def center_tumor_slice(tumor_mask):
    """
    Devuelve el corte axial correspondiente al centro medio del tumor.

    Replica la lógica del notebook original:
        coords = np.argwhere(pred_tumor > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)
    """
    if tumor_mask is None or tumor_mask.sum() == 0:
        return None

    coords = np.argwhere(tumor_mask > 0)
    cz, cy, cx = coords.mean(axis=0).astype(int)

    return int(cz)


def make_rgba_overlay(mask_slice, color, alpha):
    """
    Convierte una máscara binaria 2D en una capa RGBA coloreada.

    Parámetros:
        mask_slice: máscara 2D.
        color: tupla RGB con valores entre 0 y 1.
        alpha: opacidad de la máscara.

    Ejemplo:
        verde = (0.0, 0.95, 0.20)
        rojo = (1.0, 0.0, 0.0)
    """

    mask = mask_slice > 0

    rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(mask, alpha, 0.0)

    return rgba


def plot_slice_panels(
    image,
    liver_mask=None,
    tumor_mask=None,
    z=0,
    show_liver=True,
    show_tumor=True,
):
    """
    Dibuja tres paneles:
        1. TAC original.
        2. TAC + hígado.
        3. TAC + hígado + tumor.

    Colores:
        - Verde: hígado predicho.
        - Rojo: tumor predicho.
    """

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    green = (0.0, 0.95, 0.20)
    red = (1.0, 0.0, 0.0)

    # --------------------------------------------------------
    # 1. TAC original
    # --------------------------------------------------------
    axes[0].imshow(image[z], cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title(f"TAC original\nSlice z = {z}")
    axes[0].axis("off")

    # --------------------------------------------------------
    # 2. TAC + hígado
    # --------------------------------------------------------
    axes[1].imshow(image[z], cmap="gray", vmin=-1, vmax=1)

    if show_liver and liver_mask is not None and np.any(liver_mask[z] > 0):
        liver_rgba = make_rgba_overlay(
            liver_mask[z],
            color=green,
            alpha=0.50,
        )
        axes[1].imshow(liver_rgba)

    axes[1].set_title("Hígado predicho")
    axes[1].axis("off")

    # --------------------------------------------------------
    # 3. TAC + hígado + tumor
    # --------------------------------------------------------
    axes[2].imshow(image[z], cmap="gray", vmin=-1, vmax=1)

    if show_liver and liver_mask is not None and np.any(liver_mask[z] > 0):
        liver_rgba = make_rgba_overlay(
            liver_mask[z],
            color=green,
            alpha=0.35,
        )
        axes[2].imshow(liver_rgba)

    if show_tumor and tumor_mask is not None and np.any(tumor_mask[z] > 0):
        tumor_rgba = make_rgba_overlay(
            tumor_mask[z],
            color=red,
            alpha=0.80,
        )
        axes[2].imshow(tumor_rgba)

    axes[2].set_title("Hígado + lesión")
    axes[2].axis("off")

    # --------------------------------------------------------
    # Leyenda dentro de la figura
    # --------------------------------------------------------
    legend_handles = [
        mpatches.Patch(color=green, label="Hígado"),
        mpatches.Patch(color=red, label="Tumor"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    return fig


def load_arrays_for_viewer(report, selected_path):
    """
    Carga TAC, máscara hepática y máscara tumoral a partir del report.
    """

    patient = load_patient_npz(selected_path)
    image = patient["image"]

    liver_mask_path = Path(report["liver_segmentation"]["mask_path"])
    liver_mask = np.load(liver_mask_path)

    tumor_mask = None
    tumor_info = report.get("tumor_segmentation", {})

    if tumor_info.get("executed", False):
        tumor_mask_path = tumor_info.get("mask_path")

        if tumor_mask_path is not None and Path(tumor_mask_path).exists():
            tumor_mask = np.load(tumor_mask_path)

    return image, liver_mask, tumor_mask


# ============================================================
# FUNCIONES DE RENDERIZADO
# ============================================================

def render_main_metrics(report):
    """
    Muestra las tres métricas principales.
    """

    status = report["healthy_pathological_classification"]
    tumor = report["tumor_segmentation"]
    hcc_crlm = report["hcc_crlm_classification"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sano / Patológico",
            status["prediction"],
        )
        st.caption(f"P(pathológico) = {status['probability_pathological']:.4f}")

    with col2:
        if tumor["executed"]:
            tumor_label = "Sí" if tumor["tumor_detected"] else "No"

            st.metric(
                "Tumor detectado",
                tumor_label,
            )
            st.caption(f"Vóxeles tumorales = {tumor['positive_voxels']}")
        else:
            st.metric("Tumor detectado", "No ejecutado")

    with col3:
        if hcc_crlm["executed"]:
            st.metric(
                "Clasificación final",
                hcc_crlm["prediction"],
            )
            st.caption(f"P(CRLM) = {hcc_crlm['probability_crlm']:.4f}")
        else:
            st.metric("Clasificación final", "No ejecutada")


def render_interactive_viewer(report, selected_path):
    """
    Visor interactivo por slices.
    """

    image, liver_mask, tumor_mask = load_arrays_for_viewer(
        report=report,
        selected_path=selected_path,
    )

    st.header("Visor interactivo por slices")

    z_liver = best_liver_slice(liver_mask)
    z_tumor_best = best_tumor_slice(tumor_mask)
    z_tumor_center = center_tumor_slice(tumor_mask)

    # --------------------------------------------------------
    # Métricas de slices
    # --------------------------------------------------------
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Slice con más hígado", z_liver)

    with col_b:
        if z_tumor_best is not None:
            st.metric("Slice con más tumor", z_tumor_best)
        else:
            st.metric("Slice con más tumor", "No disponible")

    with col_c:
        if z_tumor_center is not None:
            st.metric("Centro medio del tumor", z_tumor_center)
        else:
            st.metric("Centro medio del tumor", "No disponible")

    st.markdown(
        """
        **Leyenda de colores:**  
        🟢 **Verde:** hígado predicho por el modelo.  
        🔴 **Rojo:** lesión tumoral predicha por el modelo.
        """
    )

    # --------------------------------------------------------
    # Controles
    # --------------------------------------------------------
    st.subheader("Controles de visualización")

    slice_mode = st.selectbox(
        "Corte inicial sugerido",
        [
            "Centro medio del tumor",
            "Corte con más tumor",
            "Corte con más hígado",
            "Corte central del TAC",
        ],
    )

    if slice_mode == "Centro medio del tumor" and z_tumor_center is not None:
        default_z = z_tumor_center
    elif slice_mode == "Corte con más tumor" and z_tumor_best is not None:
        default_z = z_tumor_best
    elif slice_mode == "Corte con más hígado":
        default_z = z_liver
    else:
        default_z = image.shape[0] // 2

    z = st.slider(
        "Selecciona slice axial",
        min_value=0,
        max_value=image.shape[0] - 1,
        value=int(default_z),
        step=1,
        key=f"slice_slider_{report['patient_id']}_{slice_mode}",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_liver = st.checkbox("Mostrar hígado", value=True)

    with col2:
        show_tumor = st.checkbox("Mostrar tumor", value=True)

    with col3:
        liver_voxels_slice = int(liver_mask[z].sum()) if liver_mask is not None else 0
        tumor_voxels_slice = int(tumor_mask[z].sum()) if tumor_mask is not None else 0

        st.write(f"**Vóxeles hígado en slice:** {liver_voxels_slice}")
        st.write(f"**Vóxeles tumor en slice:** {tumor_voxels_slice}")

    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------
    fig = plot_slice_panels(
        image=image,
        liver_mask=liver_mask,
        tumor_mask=tumor_mask,
        z=z,
        show_liver=show_liver,
        show_tumor=show_tumor,
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_structured_report(report):
    """
    Muestra el informe estructurado en formato Markdown.
    """

    status = report["healthy_pathological_classification"]
    liver = report["liver_segmentation"]
    tumor = report["tumor_segmentation"]
    hcc_crlm = report["hcc_crlm_classification"]

    st.markdown(f"""
## Informe estructurado

### Datos del paciente

| Campo | Valor |
|---|---|
| ID paciente | `{report["patient_id"]}` |
| Archivo de entrada | `{report["input_path"]}` |
| Fase de pipeline | `{report["stage"]}` |

---

### Segmentación hepática

| Campo | Valor |
|---|---|
| Estado | `{liver["status"]}` |
| Umbral | `{liver["threshold"]}` |
| Vóxeles positivos | `{liver["positive_voxels"]}` |
| Máscara | `{liver["mask_path"]}` |

---

### Clasificación sano/patológico

| Campo | Valor |
|---|---|
| Predicción | **{status["prediction"]}** |
| Probabilidad patológico | `{status["probability_pathological"]:.4f}` |
| Umbral | `{status["threshold"]}` |
""")

    st.markdown("---")
    st.markdown("### Segmentación tumoral")

    if tumor["executed"]:
        st.markdown(f"""
| Campo | Valor |
|---|---|
| Ejecutada | `{tumor["executed"]}` |
| Tumor detectado | `{tumor["tumor_detected"]}` |
| Vóxeles tumorales | `{tumor["positive_voxels"]}` |
| Umbral tumoral | `{tumor["threshold"]}` |
| Tamaño mínimo de componente | `{tumor["min_component_size"]}` |
| Máscara | `{tumor["mask_path"]}` |
""")
    else:
        st.markdown(f"""
| Campo | Valor |
|---|---|
| Ejecutada | `{tumor["executed"]}` |
| Motivo | `{tumor.get("reason", "No especificado")}` |
""")

    st.markdown("---")
    st.markdown("### Clasificación HCC / CRLM")

    if hcc_crlm["executed"]:
        st.markdown(f"""
| Campo | Valor |
|---|---|
| Ejecutada | `{hcc_crlm["executed"]}` |
| Predicción final | **{hcc_crlm["prediction"]}** |
| Probabilidad CRLM | `{hcc_crlm["probability_crlm"]:.4f}` |
| Probabilidad HCC | `{hcc_crlm["probability_hcc"]:.4f}` |
| Umbral | `{hcc_crlm["threshold"]}` |
""")
    else:
        st.markdown(f"""
| Campo | Valor |
|---|---|
| Ejecutada | `{hcc_crlm["executed"]}` |
| Motivo | `{hcc_crlm.get("reason", "No especificado")}` |
""")

    st.markdown("---")
    st.warning(
        "Resultado generado por un prototipo experimental. "
        "No validado para uso clínico."
    )


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "report" not in st.session_state:
    st.session_state.report = None

if "selected_path" not in st.session_state:
    st.session_state.selected_path = None


# ============================================================
# TÍTULO
# ============================================================

st.title("TFM Liver AI")
st.subheader("Pipeline automática para análisis hepático en TAC")

st.markdown(
    """
    Esta aplicación ejecuta la pipeline completa:

    **TAC → segmentación hepática → clasificación sano/patológico → segmentación tumoral → clasificación HCC/CRLM**

    > Prototipo experimental desarrollado con fines académicos. No validado para uso clínico.
    """
)


# ============================================================
# SELECCIÓN DE PACIENTE
# ============================================================

st.sidebar.header("Selección de paciente")

npz_files = sorted(NPZ_DIR.glob("*.npz"))

if len(npz_files) == 0:
    st.error(f"No se han encontrado archivos .npz en: {NPZ_DIR}")
    st.stop()

patient_names = [p.name for p in npz_files]

selected_name = st.sidebar.selectbox(
    "Selecciona un paciente .npz",
    patient_names,
)

selected_path = NPZ_DIR / selected_name

st.sidebar.markdown("**Archivo seleccionado:**")
st.sidebar.code(str(selected_path))


# Si el usuario cambia de paciente, limpiamos el resultado anterior
if st.session_state.selected_path is not None:
    if st.session_state.selected_path != str(selected_path):
        st.session_state.report = None

st.session_state.selected_path = str(selected_path)


# ============================================================
# BOTÓN DE ANÁLISIS
# ============================================================

run_button = st.sidebar.button("Ejecutar análisis completo", type="primary")


# ============================================================
# EJECUCIÓN DE LA PIPELINE
# ============================================================

if run_button:
    with st.spinner("Ejecutando la pipeline completa. Esto puede tardar unos minutos..."):
        report = analizar_paciente_stage3(selected_path)

    st.session_state.report = report
    st.success("Análisis completado correctamente.")


# ============================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================

if st.session_state.report is None:
    st.info(
        "Selecciona un paciente en la barra lateral y pulsa "
        "**Ejecutar análisis completo**."
    )
    st.stop()


report = st.session_state.report


# ------------------------------------------------------------
# RESULTADOS PRINCIPALES
# ------------------------------------------------------------

st.header("Resultados principales")
render_main_metrics(report)


# ------------------------------------------------------------
# VISOR INTERACTIVO
# ------------------------------------------------------------

render_interactive_viewer(
    report=report,
    selected_path=selected_path,
)


# ------------------------------------------------------------
# INFORME ESTRUCTURADO
# ------------------------------------------------------------

render_structured_report(report)


# ------------------------------------------------------------
# JSON COMPLETO Y DESCARGA
# ------------------------------------------------------------

st.header("Archivos generados")

report_path = Path(report["final_outputs"]["report_path"])

if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    st.download_button(
        label="Descargar report.json",
        data=report_text,
        file_name="report.json",
        mime="application/json",
    )

with st.expander("Ver JSON completo"):
    st.json(report)