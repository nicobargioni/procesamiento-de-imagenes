============================================================
============================================================

🧠 Monitor de Atención Visual en Tiempo Real

Este proyecto implementa un sistema de monitoreo de atención en tiempo real utilizando visión por computadora con MediaPipe, visualización con Streamlit y procesamiento con OpenCV.

============================================================
============================================================

🎯 Objetivo

Detectar si un usuario está prestando atención frente a la cámara, evaluando la posición del rostro y la orientación de la mirada, con validación adicional mediante segmentación semántica para asegurar que hay una persona real en escena.

============================================================
============================================================

🔍 Funcionalidad

Detección facial en vivo con MediaPipe FaceMesh

Segmentación de personas con MediaPipe SelfieSegmentation, para validar que haya una persona real frente a cámara

Evaluación del índice de atención con criterios de posición de nariz y ojos

Penalizaciones si la cabeza está baja o la mirada desviada

Visualización del análisis en tiempo real con OpenCV + gráfico de atención con matplotlib

Al detener el monitoreo, generación de un gráfico final de resumen.

============================================================
============================================================

🗂️ Estructura del proyecto

monitor_atencion/
├── main.py                 # Interfaz principal con Streamlit
├── detector.py             # Lógica de atención y landmarks faciales
├── segmentacion.py         # Validación de presencia humana por segmentación
├── graficos.py             # Visualización del índice de atención
├── requirements.txt        # Lista de dependencias
└── README.md               # Documentación del proyecto

============================================================
============================================================

▶️ Instrucciones de uso

Instalar dependencias:

pip install -r requirements.txt

Ejecutar el sistema:

streamlit run main.py

Si no funciona, ejecutar python -m streamlit run main.py

============================================================
============================================================

⚙️ Dependencias (requirements.txt)

streamlit==1.30.0
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.26.4
matplotlib==3.8.3

============================================================
============================================================

📚 Créditos

Desarrollado por: Nicolás Bargioni y un mínimo de Github Copilot y Chatgpt

Proyecto académico basado en los contenidos la materia Procesamiento de Imágenes

============================================================
============================================================