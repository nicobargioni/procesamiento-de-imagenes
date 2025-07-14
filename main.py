# === Importación de librerías ===
import streamlit as st                           # Interfaz web interactiva
import cv2                                       # Procesamiento de imágenes con OpenCV
import numpy as np                               # Cálculos numéricos
import time                                      # Medición de tiempo
from collections import deque                    # Estructura para guardar historial de atención
import mediapipe as mp                           # Librería de visión por computadora en tiempo real

# === Importación de módulos personalizados ===
from detector import evaluar_atencion, dibujar_landmarks
from segmentacion import detectar_presencia_persona, aplicar_mascara_segmentacion
from graficos import graficar_atencion

# ---------- Estado inicial ----------
# Inicialización de variables persistentes entre ciclos (usando session_state)
if "running" not in st.session_state:
    st.session_state.running = False              # Indica si está en modo monitoreo
    st.session_state.cap = None                   # Cámara de video
    st.session_state.face_mesh = None             # Modelo de detección facial
    st.session_state.segmentador = None           # Modelo de segmentación semántica
    st.session_state.ventana_atencion = deque(maxlen=100)   # Buffer con últimos niveles de atención
    st.session_state.x_vals = deque(maxlen=100)             # Buffer con los números de frame
    st.session_state.total_frames = 0             # Conteo de todos los frames procesados
    st.session_state.atencion_frames = 0          # Conteo de frames con atención detectada
    st.session_state.start_time = None            # Tiempo de inicio del monitoreo
    st.session_state.last_report_time = 0         # Último timestamp en que se actualizó el sidebar
    st.session_state.attention_log = []           # Historial completo del índice de atención


# ---------- UI ----------
st.title("🎯 Monitor de Atención Visual en Tiempo Real")
st.markdown("""
Este programa utiliza visión por computadora para analizar tu nivel de atención durante una videollamada.
Evalúa si tu rostro está centrado y si tu mirada se mantiene hacia el frente.  
Ideal para contextos educativos, de trabajo remoto o validación de presencia.

👁️‍🗨️ A través de la webcam, el sistema detecta si desviás la mirada, girás la cabeza o bajás la vista, y muestra un indicador visual de atención junto a un gráfico en tiempo real.
### Instrucciones:
1. Asegurate de que tu cámara esté encendida y funcionando.
2. Ajustá los umbrales de atención en la barra lateral según tu preferencia (lo ideal es dejarlo en 0.4 para giro a la izquierda, 0.6 para giro a la derecha y 0.25 para cabeza baja).
3. Presioná "Iniciar monitoreo" para comenzar a evaluar tu atención.
4. Observá el indicador de atención y el gráfico en tiempo real.
5. Detenelo cuando quieras y revisá el resumen de tu atención.            
""")
st.subheader("👉 La premisa es la siguiente 👈")
st.markdown("Para demostrar tu atención, procurá estar justo en medio de donde te muestra la cámara 😉")

########### SIDEBAR ###########
with st.sidebar:
    st.subheader("🤓 Umbrales de Atención")
    
    # Ajuste de umbrales para evaluar qué se considera “atención”
    with st.expander("🎛 Ajustes de Umbrales", expanded=False):
        st.markdown("""
        Ajustá la sensibilidad del sistema de atención:
        - **Giro izquierda/derecha**: margen de movimiento horizontal permitido.
        - **Cabeza baja**: inclinación vertical antes de penalizar.
        """)
        umbral_giro_izquierda = st.slider("Giro hacia izquierda", 0.0, 1.0, 0.4, step=0.01)
        umbral_giro_derecha   = st.slider("Giro hacia derecha",  0.0, 1.0, 0.6, step=0.01)
        umbral_ojos_y_baja    = st.slider("Cabeza baja",          0.0, 1.0, 0.25, step=0.01)

    st.markdown("---")
    st.subheader("⚙️ Configuración")

    # Opciones de visualización
    # Opción 1: Mostrar landmarks faciales
    mostrar_landmarks = st.checkbox("😀 Mostrar landmarks faciales", value=True)
    st.caption("Visualiza los puntos y líneas sobre tu rostro (FaceMesh).")

    # Opción 2: Activar segmentación semántica
    usar_segmentacion = st.checkbox("🖼 Activar segmentación semántica", value=True)
    st.caption("Valida que haya una persona real (no una imagen). Requiere cámara activa.")

    # Opción 3: Ver máscara de segmentación
    ver_mascara_segmentacion = st.checkbox("👽 Ver máscara de segmentación", value=True)
    st.caption("Superpone una máscara verde sobre la persona detectada en la imagen. La idea es separar figura de fondo")

    # Opción 4: Mostrar resumen de atención
    mostrar_tabla = st.checkbox("📊 Mostrar resumen de atención al finalizar", value=True)
    st.caption("Muestra un gráfico y el promedio de atención al detener el monitoreo.")

    # Cronómetro de tiempo activo
    if st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        minutes, seconds = divmod(elapsed, 60)
        st.markdown(f"⏱️ Tiempo de llamada: **{minutes:02d}:{seconds:02d}**")

# ---------- Botones ----------
# Botón para iniciar monitoreo
if not st.session_state.running:
    if st.button("▶️ Iniciar monitoreo"):
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.total_frames = 0
        st.session_state.atencion_frames = 0
        st.session_state.attention_log.clear()
        st.session_state.ventana_atencion.clear()
        st.session_state.x_vals.clear()
else:
    # Botón para detener monitoreo y liberar la cámara
    if st.button("🛑 Detener monitoreo"):
        st.session_state.running = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

# ---------- Procesamiento ----------
if st.session_state.running:

    if st.session_state.cap is None:
        with st.spinner("⌛ Cargando modelo y preparando la cámara: va a tomar un minuto"):
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        st.session_state.segmentador = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    col1, col2 = st.columns(2)
    video_placeholder = col1.empty()
    grafico_placeholder = col2.empty()

    while st.session_state.running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            break

        # Procesamiento básico de frame
        frame = cv2.flip(frame, 1)                           # Espejo horizontal (efecto selfie)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)         # Conversión a RGB para MediaPipe
        results = st.session_state.face_mesh.process(rgb)    # Detección de rostro
        segment = st.session_state.segmentador.process(rgb)  # Segmentación de persona
        h, w, _ = frame.shape

        # Contador de frames
        st.session_state.total_frames += 1
        score = 0
        hay_persona = True

        # Validación de persona real (evita fotos)
        if usar_segmentacion:
            hay_persona = detectar_presencia_persona(segment.segmentation_mask)

        # Si se detecta rostro y persona válida...
        if hay_persona and results.multi_face_landmarks:
            for rostro in results.multi_face_landmarks:
                if mostrar_landmarks:
                    frame = dibujar_landmarks(frame, rostro)
                score, _ = evaluar_atencion(
                    rostro, w, h,
                    umbral_giro_izquierda, umbral_giro_derecha, umbral_ojos_y_baja
                )

            # Clasificación visual según atención
            if score >= 0.7:
                st.session_state.atencion_frames += 1
                texto = "ATENTO"
                color = (0, 255, 0)
            else:
                texto = "NO ATENTO"
                color = (0, 0, 255)
        else:
            texto = "Sin rostro o sin persona"
            color = (150, 150, 255)

        # Cálculo del índice de atención y actualización de buffers
        atencion_index = int((st.session_state.atencion_frames / st.session_state.total_frames) * 100)
        st.session_state.ventana_atencion.append(atencion_index)
        st.session_state.x_vals.append(st.session_state.total_frames)
        st.session_state.attention_log.append(atencion_index)

        # Anotaciones visuales en el frame
        cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"Atención: {atencion_index}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Aplicación de máscara de segmentación si está activada
        if ver_mascara_segmentacion and usar_segmentacion:
            frame = aplicar_mascara_segmentacion(frame, segment.segmentation_mask)

        # Mostrar frame y gráfico
        video_placeholder.image(frame, channels="BGR")
        fig = graficar_atencion(st.session_state.ventana_atencion, st.session_state.x_vals)
        grafico_placeholder.pyplot(fig)

        # Actualización lateral del promedio cada 30 segundos
        elapsed = int(time.time() - st.session_state.start_time)
        if elapsed - st.session_state.last_report_time >= 30:
            ultimos_30 = st.session_state.attention_log[-30:]
            if ultimos_30:
                promedio = sum(ultimos_30) / len(ultimos_30)
                st.sidebar.info(f"🧠 Atención últimos 30s: **{promedio:.1f}%**")
            st.session_state.last_report_time = elapsed

        time.sleep(0.03)  # Espera corta para no saturar CPU


# ---------- Resumen final ----------
# Muestra gráfico final y promedio cuando se detiene el monitoreo
if not st.session_state.running and st.session_state.attention_log and mostrar_tabla:
    st.subheader("📋 Resumen de atención")
    promedio_total = sum(st.session_state.attention_log) / len(st.session_state.attention_log)
    st.markdown(f"🧠 Promedio total: **{promedio_total:.2f}%**")
    fig_final = graficar_atencion(st.session_state.ventana_atencion, st.session_state.x_vals)
    st.pyplot(fig_final)

# ---------- Footer fijo ----------
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9f9f9;
            color: #666;
            text-align: center;
            font-size: 0.85em;
            padding: 0.5em 0;
            border-top: 1px solid #ddd;
        }
    </style>
    <div class="footer">
        Desarrollado por <strong>Nicolás Bargioni</strong> | Año 2025 | ISSD: Inteligencia Artificial y Ciencia de Datos 🧠
    </div>
    """,
    unsafe_allow_html=True
)