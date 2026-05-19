\# TFM Liver AI



Prototipo académico para el análisis automático de lesiones hepáticas en imágenes TAC mediante modelos de \*deep learning\*.



El sistema implementa una pipeline completa de inferencia:



```text

TAC

→ segmentación hepática

→ clasificación sano/patológico

→ segmentación tumoral

→ clasificación HCC/CRLM

```



Además, incluye una aplicación interactiva desarrollada con \*\*Streamlit\*\* para ejecutar el análisis y visualizar los resultados por cortes axiales.



\---



\## Aviso importante



Este proyecto tiene finalidad exclusivamente académica y experimental.



No está validado para uso clínico y no debe utilizarse para tomar decisiones médicas reales.



\---



\## Estructura del proyecto



```text

tfm\_liver\_ai/

│

├── app/

│   └── streamlit\_app.py

│

├── src/

│   ├── config.example.py

│   │

│   ├── inference/

│   │   ├── data\_loader.py

│   │   ├── pipeline.py

│   │   ├── test\_pipeline\_stage1.py

│   │   ├── test\_pipeline\_stage2.py

│   │   └── test\_pipeline\_stage3.py

│   │

│   ├── models/

│   │   ├── liver\_segmentation.py

│   │   ├── healthy\_pathological.py

│   │   ├── tumor\_segmentation.py

│   │   └── hcc\_crlm\_classifier.py

│   │

│   └── utils/

│       └── visualization.py

│

├── run\_inference.py

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## Modelos incluidos en la pipeline



El sistema integra cuatro modelos principales:



1\. \*\*Segmentación hepática\*\*



&#x20;  Predice la máscara del hígado a partir del volumen TAC.



2\. \*\*Clasificación sano/patológico\*\*



&#x20;  Clasifica el caso como sano o patológico utilizando como entrada el TAC y la máscara hepática predicha.



3\. \*\*Segmentación tumoral\*\*



&#x20;  Si el caso se clasifica como patológico, se ejecuta un modelo de segmentación tumoral usando como entrada el TAC y la máscara hepática.



4\. \*\*Clasificación HCC/CRLM\*\*



&#x20;  Si se detecta tumor, se extrae una ROI centrada en la lesión y se clasifica el caso como HCC o CRLM.



\---



\## Datos y modelos



Por motivos de tamaño, privacidad y reproducibilidad, este repositorio no incluye:



```text

\- pacientes .npz

\- imágenes DICOM

\- archivos NIfTI

\- pesos de modelos .pth

\- salidas generadas en outputs/

```



Para ejecutar el proyecto es necesario disponer localmente de:



```text

\- pacientes preprocesados en formato .npz

\- pesos entrenados de los cuatro modelos

```



Las rutas deben configurarse en el archivo:



```text

src/config.py

```



Para ello, se debe copiar primero el archivo de ejemplo:



```bash

copy src\\config.example.py src\\config.py

```



y editar las rutas correspondientes en `src/config.py`.



\---



\## Instalación



Se recomienda utilizar un entorno de Anaconda con PyTorch instalado.



Ejemplo:



```bash

conda activate torchfix

pip install -r requirements.txt

```



Si se desea usar GPU, es necesario instalar una versión de PyTorch compatible con la versión de CUDA disponible en el equipo.



\---



\## Ejecución por terminal



Para ejecutar la pipeline completa sobre el primer paciente encontrado:



```bash

python run\_inference.py --first --stage 3

```



Para ejecutar la pipeline sobre un paciente concreto:



```bash

python run\_inference.py --input "ruta/al/paciente.npz" --stage 3

```



Los stages disponibles son:



```text

stage 1: segmentación hepática + clasificación sano/patológico

stage 2: añade segmentación tumoral

stage 3: añade clasificación HCC/CRLM

```



\---



\## Aplicación Streamlit



Para lanzar la interfaz interactiva:



```bash

streamlit run app\\streamlit\_app.py

```



La aplicación permite:



```text

\- seleccionar un paciente .npz

\- ejecutar la pipeline completa

\- visualizar los resultados principales

\- recorrer el TAC mediante un visor por slices

\- activar o desactivar las máscaras de hígado y tumor

\- consultar un informe estructurado

\- descargar el archivo report.json

```



En el visor interactivo:



```text

Verde: hígado predicho

Rojo: lesión tumoral predicha

```



\---



\## Archivos generados



Por cada paciente analizado se crea una carpeta dentro de:



```text

outputs/

```



Ejemplo:



```text

outputs/

└── crlm\_\_CRLM-CT-1001/

&#x20;   ├── liver\_mask\_pred.npy

&#x20;   ├── liver\_prob\_pred.npy

&#x20;   ├── tumor\_mask\_pred.npy

&#x20;   ├── tumor\_prob\_pred.npy

&#x20;   ├── report\_stage2.json

&#x20;   ├── report\_stage3.json

&#x20;   ├── report.json

&#x20;   └── overview.png

```



El archivo principal es:



```text

report.json

```



donde se guarda el resultado estructurado de la inferencia.



\---



\## Salida del sistema



El informe final generado por la pipeline contiene información como:



```text

\- ID del paciente

\- ruta del archivo de entrada

\- resultado de la segmentación hepática

\- probabilidad de caso patológico

\- resultado sano/patológico

\- resultado de segmentación tumoral

\- número de vóxeles tumorales detectados

\- clasificación final HCC/CRLM

\- probabilidad de CRLM

\- probabilidad de HCC

\- rutas de los archivos generados

```



\---



\## Interfaz visual



La aplicación Streamlit incluye un visor interactivo por cortes axiales. Este visor permite revisar el TAC completo y comprobar visualmente las máscaras generadas por los modelos.



La visualización se organiza en tres paneles:



```text

1\. TAC original

2\. TAC con máscara hepática

3\. TAC con máscara hepática y máscara tumoral

```



Además, el usuario puede seleccionar distintos cortes de referencia:



```text

\- centro medio del tumor

\- corte con más tumor

\- corte con más hígado

\- corte central del TAC

```



Esto permite inspeccionar de forma más precisa el comportamiento del modelo en el volumen 3D completo.



\---



\## Estado del proyecto



Este repositorio contiene una versión modular y profesionalizada del prototipo desarrollado para el TFM.



La versión inicial se desarrolló en notebooks. Posteriormente, el código se reorganizó en módulos independientes, se creó una pipeline ejecutable por terminal y se añadió una interfaz interactiva con Streamlit.



\---



\## Licencia y uso



Este proyecto se comparte con fines académicos y de investigación.



No se incluyen datos médicos, modelos entrenados ni resultados de pacientes.



El uso del código debe respetar las condiciones académicas y éticas asociadas al tratamiento de imágenes médicas.

