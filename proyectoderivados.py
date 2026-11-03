import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def calcular_parametros(S0, K, r, sigma, T, n):
    """Calcula los parámetros."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)
    return {'dt': dt, 'u': u, 'd': d, 'q': q, 'df': df}


def modelo_crr(S0, K, r, sigma, T, n, option_type='call', option_style='american', return_trees=False):
    """Valora una opción financiera usando el modelo binomial."""
    params = calcular_parametros(S0, K, r, sigma, T, n)
    u, d, q, df = params['u'], params['d'], params['q'], params['df']

    # Árbol de precios del activo subyacente
    S_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            S_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Árbol del valor de la opción
    V_tree = np.zeros((n + 1, n + 1))

    # Payoff al vencimiento
    if option_type.lower() == 'call':
        V_tree[:, n] = np.maximum(S_tree[:, n] - K, 0)
    else:
        V_tree[:, n] = np.maximum(K - S_tree[:, n], 0)

    # Inducción hacia atrás
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont_value = df * (q * V_tree[j, i + 1] + (1 - q) * V_tree[j + 1, i + 1])

            if option_style.lower() == 'american':
                if option_type.lower() == 'call':
                    exercise_value = max(S_tree[j, i] - K, 0)
                else:
                    exercise_value = max(K - S_tree[j, i], 0)
                V_tree[j, i] = max(cont_value, exercise_value)
            else:
                V_tree[j, i] = cont_value

    if return_trees:
        return V_tree[0, 0], S_tree, V_tree
    return V_tree[0, 0]


# --- Función Principal ---

def main():
    st.set_page_config(
        page_title="Modelo Binomial para Valoración de Opciones Financieras",
        page_icon="imagen.jpg",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Simulación de Opciones Financieras en tiempo discreto")
    st.subheader(
        "Esta aplicación valora opciones de tipo Call/Put, europeas, americanas y exóticas de primera generación"
        "utilizando árboles binomiales."
    )

    # --- BARRA LATERAL ---
    st.sidebar.header("Configura tu opción")
    option_style = st.sidebar.selectbox(
        "Estilo de la opción", ("Americana", "Europea")
    ).lower()
    option_type = st.sidebar.selectbox(
        "Tipo de la opción", ("Put", "Call")
    ).lower()

    st.sidebar.header("Parámetros del activo y del mercado")
    S0 = st.sidebar.number_input(
        "Precio actual (S0)", min_value=1.0, value=180.0, step=1.0, format="%.2f"
    )
    K = st.sidebar.number_input(
        "Precio de ejercicio (Strike K)", min_value=1.0, value=180.0, step=1.0, format="%.2f"
    )
    r = st.sidebar.slider(
        "Tasa libre de riesgo anual", min_value=0.0, max_value=0.20,
        value=0.05, step=0.001, format="%.3f"
    )
    sigma = st.sidebar.slider(
        "Volatilidad anual", min_value=0.01, max_value=1.0,
        value=0.25, step=0.01, format="%.2f"
    )
    T = st.sidebar.slider(
        "Tiempo al vencimiento en años (T)", min_value=0.1, max_value=5.0,
        value=1.0, step=0.1, format="%.1f"
    )

    st.sidebar.header("Número de pasos del árbol")
    n = st.sidebar.slider(
        "Número de pasos (n)", min_value=1, max_value=20, value=3, step=1
    )

    # --- CÁLCULOS ---
    params = calcular_parametros(S0, K, r, sigma, T, n)
    precio_opcion, S_tree, V_tree = modelo_crr(
        S0, K, r, sigma, T, n, option_type, option_style, return_trees=True
    )

    # --- RESULTADOS ---
    st.subheader(f"Resultados de la valoración ({option_style.capitalize()} - {option_type.capitalize()})")
    col1, col2 = st.columns(2)
    col1.metric(
        f"Precio Teórico de la Opción",
        f"${precio_opcion:.4f} MXN"
    )
    col2.metric("Pasos del árbol (n)", n)

    st.divider()

    # --- PARÁMETROS CALCULADOS ---
    st.subheader("Resultado de los parámetros del modelo")
    params_df = pd.DataFrame({
        "Parámetro": [
            "Δt (paso de tiempo)",
            "u (factor de subida)",
            "d (factor de bajada)",
            "q (probabilidad de riesgo neutral)",
            "FD (factor de descuento)",
        ],
        "Valor": [
            f"{params['dt']:.6f}",
            f"{params['u']:.6f}",
            f"{params['d']:.6f}",
            f"{params['q']:.6f}",
            f"{params['df']:.6f}",
        ],
    })
    st.table(params_df)

    st.divider()

    # --- ÁRBOL BINOMIAL ---
    st.subheader("Árbol binomial")
    st.markdown(
        "Cada nodo muestra el precio del activo (`S`) y el valor de la opción (`V`). "
        
    )

    fig_tree = go.Figure()

    # Aristas del árbol
    for i in range(n):
        for j in range(i + 1):
            fig_tree.add_trace(go.Scatter(
                x=[i, i + 1], y=[S_tree[j, i], S_tree[j, i + 1]],
                mode='lines', line=dict(color='lightblue', width=2),
                hoverinfo='skip', showlegend=False,
            ))
            fig_tree.add_trace(go.Scatter(
                x=[i, i + 1], y=[S_tree[j, i], S_tree[j + 1, i + 1]],
                mode='lines', line=dict(color='lightblue', width=2),
                hoverinfo='skip', showlegend=False,
            ))

    # Nodos
    for i in range(n + 1):
        for j in range(i + 1):
            s_val, v_val = S_tree[j, i], V_tree[j, i]
            fig_tree.add_trace(go.Scatter(
                x=[i], y=[s_val], mode='markers',
                marker=dict(size=12, color='darkblue'),
                text=f"S: ${s_val:.2f}<br>V: ${v_val:.2f}",
                hoverinfo='text', showlegend=False,
            ))
            # Etiqueta permanente
            fig_tree.add_annotation(
                x=i, y=s_val + S0 * 0.05,
                text=f"S:${s_val:.1f}<br>V:${v_val:.1f}",
                showarrow=False,
                font=dict(size=9, color="black"),
                bgcolor="rgba(255, 255, 200, 0.7)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=2,
            )

    fig_tree.update_layout(
        title=dict(
            text=f"Árbol - Opción {option_type.capitalize()} {option_style.capitalize()}",
            font=dict(size=20),
        ),
        xaxis_title="Pasos de tiempo",
        yaxis_title="Precio del activo (MXN)",
        template='simple_white',
        height=700,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
    )
    st.plotly_chart(fig_tree, use_container_width=True)


if __name__ == "__main__":
    main()