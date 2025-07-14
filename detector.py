# Este módulo contiene funciones relacionadas con la detección de landmarks faciales y la evaluación de atención visual.
# Utiliza MediaPipe Face Mesh para obtener puntos clave del rostro.
# Incluye:
# - dibujar_landmarks: dibuja la malla del rostro en el frame.
# - evaluar_atencion: calcula un score basado en la posición de la nariz y los ojos, penalizando la mirada baja o desviada.

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def dibujar_landmarks(frame, landmarks):

    """
    Dibuja los puntos y conexiones de la malla facial sobre un frame de video.

    Parámetros:
    - frame: imagen de la cámara en formato BGR (frame de OpenCV)
    - landmarks: puntos clave del rostro obtenidos con MediaPipe FaceMesh

    Retorna:
    - El frame con los landmarks dibujados (solo las conexiones de la malla)
    """
        
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
    )
    return frame

def evaluar_atencion(landmarks, w, h,
                     UMBRAL_CENTRO_MIN, UMBRAL_CENTRO_MAX, UMBRAL_OJOS_Y_BAJA):
    
    """
    Evalúa si el rostro está centrado y la mirada está dirigida al frente, usando landmarks faciales.

    Parámetros:
    - landmarks: puntos clave del rostro (MediaPipe FaceMesh)
    - w, h: ancho y alto del frame, en píxeles
    - UMBRAL_CENTRO_MIN, UMBRAL_CENTRO_MAX: límites para definir si la nariz y ojos están centrados horizontalmente
    - UMBRAL_OJOS_Y_BAJA: umbral vertical para detectar si los ojos están bajos (mirada hacia abajo)

    Lógica:
    - Evalúa si la nariz está centrada en el eje X → suma 0.5 al score si es correcto
    - Evalúa si el centro de los ojos está centrado → suma 0.5 al score si es correcto
    - Penaliza si la nariz está demasiado baja respecto al rostro → score se anula
    - Penaliza si los ojos están bajos en el eje Y → resta 0.3 al score

    Retorna:
    - score: valor entre 0 y 1 (puede ser negativo si hay penalizaciones)
    - detalles: lista de advertencias o penalizaciones detectadas
    """
        
    score = 0
    detalles = []

    # Coordenadas claves
    nose = landmarks.landmark[1]
    right_eye = landmarks.landmark[33]
    left_eye = landmarks.landmark[263]
    eyes_center_x = (right_eye.x + left_eye.x) / 2
    eyes_center_y = (right_eye.y + left_eye.y) / 2

    # Condiciones de atención
    if UMBRAL_CENTRO_MIN < nose.x < UMBRAL_CENTRO_MAX:
        score += 0.5
    else:
        detalles.append("Nariz fuera de centro")

    if UMBRAL_CENTRO_MIN < eyes_center_x < UMBRAL_CENTRO_MAX:
        score += 0.5
    else:
        detalles.append("Ojos fuera de centro")

    # Penalización por cabeza baja
    forehead_y = landmarks.landmark[10].y
    chin_y = landmarks.landmark[152].y
    nose_rel_y = (nose.y - forehead_y) / (chin_y - forehead_y)

    # Recordatorio: el eje Y va de 0 (arriba) a 1 (abajo)
    if nose_rel_y > 0.7:
        score = 0
        detalles.append("Penalización: Nariz muy baja")
    elif eyes_center_y > UMBRAL_OJOS_Y_BAJA:
        score -= 0.3
        detalles.append("Advertencia: Mirada baja")

    return score, detalles
