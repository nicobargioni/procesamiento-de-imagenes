import numpy as np
import cv2

# Acá va la validación de presencia humana (MediaPipe SelfieSegmentation)
# Este módulo se encarga de validar si hay presencia humana real en la imagen capturada,
# utilizando la segmentación semántica de MediaPipe.
# Se evalúa el porcentaje del frame que corresponde a la clase "persona".
# Si es inferior a un umbral configurable, se asume que no hay nadie frente a la cámara.
# Básicamente la función "principal" es que no se engañe al sistema con una foto, por ejemplo

def detectar_presencia_persona(mask, umbral=0.1):
    """
    Determina si hay suficiente área etiquetada como "persona" en la máscara de segmentación.
    """
    porcentaje_visible = np.mean(mask > 0.6)
    return porcentaje_visible > umbral

def aplicar_mascara_segmentacion(frame, mask, alpha=0.4, umbral=0.1):
    """
    Superpone visualmente la máscara de segmentación sobre el frame original.
    """
    if len(mask.shape) == 2:
        mask_3c = np.stack([mask] * 3, axis=-1)
    else:
        mask_3c = mask

    color_mask = np.zeros_like(frame, dtype=np.uint8)
    color_mask[:] = (0, 255, 0)  # Verde

    # Donde mask > umbral → se mezcla el frame original con color_mask
    blended = np.where(
        mask_3c > umbral,
        (alpha * frame + (1 - alpha) * color_mask).astype(np.uint8),
        frame
    )
    return blended
