# Este módulo se encarga de generar un gráfico en tiempo real del índice de atención.
# Utiliza matplotlib para crear una figura que puede ser renderizada por Streamlit.
# La función espera listas de atención acumulada y frames recorridos.

import matplotlib.pyplot as plt

def graficar_atencion(ventana_atencion, x_vals):
    """
    Genera una figura de matplotlib con el historial del índice de atención.

    Parámetros:
    - ventana_atencion: deque o lista con los valores de atención (%)
    - x_vals: deque o lista con los valores de tiempo o número de frame

    Retorna:
    - fig: figura de matplotlib lista para mostrar en Streamlit
    """
    fig, ax = plt.subplots()
    ax.plot(x_vals, ventana_atencion, color='lime', linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_title("Índice de Atención en Tiempo Real")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Atención (%)")
    ax.grid(True)
    return fig
