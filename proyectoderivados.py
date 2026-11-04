import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools

# Colores sacados del logo, para usarlos en las gráficas y en el CSS
AZUL_MARINO = "#0D2B4E"
AZUL = "#1F6FB5"
AZUL_CLARO = "#4A9FE0"
ROJO = "#C0392B"


def formato_mxn(x):
    """Da formato de dinero: $1,234.56 o -$1,234.56."""
    if abs(x) < 0.005:
        x = 0.0
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def calcular_parametros(S0, K, r, sigma, T, n):
    """Calcula los parámetros."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)
    return {'dt': dt, 'u': u, 'd': d, 'q': q, 'df': df}


def modelo_crr(S0, K, r, sigma, T, n, option_type='call', option_style='americana', return_trees=False):
    """Valora una opción o un forward usando el modelo binomial."""
    params = calcular_parametros(S0, K, r, sigma, T, n)
    u, d, q, df = params['u'], params['d'], params['q'], params['df']

    # Árbol visual de precios
    S_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            S_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    es_asiatica = option_style.lower().startswith('asi')
    es_lookback = option_style.lower() == 'lookback'
    es_forward = option_style.lower() == 'forward'

    # FORWARD
    # Payoff lineal (puede ser negativo) y sin ejercicio anticipado
    if es_forward:
        V_tree = np.zeros((n + 1, n + 1))
        if option_type.lower() == 'larga':
            V_tree[:, n] = S_tree[:, n] - K
        else:
            V_tree[:, n] = K - S_tree[:, n]

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                V_tree[j, i] = df * (q * V_tree[j, i + 1] + (1 - q) * V_tree[j + 1, i + 1])

        if return_trees:
            return V_tree[0, 0], S_tree, V_tree, None
        return V_tree[0, 0]

    #  OPCIONES EXÓTICAS
    if es_asiatica or es_lookback:
        caminos = list(itertools.product([1, 0], repeat=n))
        payoff_total = 0.0
        datos_trayectorias = []

        for idx, camino in enumerate(caminos):
            S_actual = S0
            precios_camino = [S0]
            subidas = sum(camino)

            for paso in camino:
                if paso == 1: S_actual *= u
                else:         S_actual *= d
                precios_camino.append(S_actual)

            payoff = 0.0
            metrica = 0.0

            if es_asiatica:
                metrica = np.mean(precios_camino)
                if option_type.lower() == 'call': payoff = max(metrica - K, 0)
                else: payoff = max(K - metrica, 0)
                nombre_metrica = "Promedio"

            elif es_lookback:
                if option_type.lower() == 'call':
                    metrica = max(precios_camino)
                    payoff = max(metrica - K, 0)
                    nombre_metrica = "Máximo"
                else:
                    metrica = min(precios_camino)
                    payoff = max(K - metrica, 0)
                    nombre_metrica = "Mínimo"

            prob = (q ** subidas) * ((1 - q) ** (n - subidas))
            payoff_total += prob * payoff

            camino_str = "".join(["↑" if p == 1 else "↓" for p in camino])
            datos_trayectorias.append({
                "Camino": camino_str,
                "Precio Final ($S_T$)": round(precios_camino[-1], 2),
                f"{nombre_metrica} del Camino": round(metrica, 2),
                "Probabilidad": f"{prob*100:.2f}%",
                "Payoff (Vencimiento)": round(payoff, 2)
            })

        # Se descuenta todo el plazo T, no solo un paso
        precio_opcion = payoff_total * np.exp(-r * T)
        V_tree = np.zeros((n + 1, n + 1))  # Árbol falso para no romper la gráfica
        if return_trees:
            return precio_opcion, S_tree, V_tree, datos_trayectorias
        return precio_opcion

    # OPCIONES EUROPEAS Y AMERICANAS
    V_tree = np.zeros((n + 1, n + 1))
    if option_type.lower() == 'call':
        V_tree[:, n] = np.maximum(S_tree[:, n] - K, 0)
    else:
        V_tree[:, n] = np.maximum(K - S_tree[:, n], 0)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont_value = df * (q * V_tree[j, i + 1] + (1 - q) * V_tree[j + 1, i + 1])

            # Antes decía 'american' y el menú manda 'americana', por eso nunca entraba aquí
            if option_style.lower() == 'americana':
                if option_type.lower() == 'call':
                    exercise_value = max(S_tree[j, i] - K, 0)
                else:
                    exercise_value = max(K - S_tree[j, i], 0)
                V_tree[j, i] = max(cont_value, exercise_value)
            else:
                V_tree[j, i] = cont_value

    if return_trees:
        return V_tree[0, 0], S_tree, V_tree, None
    return V_tree[0, 0]


# --- Función para crear gráficos de árboles ---
def graficar_arbol(S_tree, V_tree, n, S0, titulo_y, titulo_grafico, modo="precios"):
    """
    modo="precios": Muestra solo S.
    modo="ambos": Muestra S y V.
    modo="derivado": Muestra solo V.
    """
    fig = go.Figure()
    # Aristas
    for i in range(n):
        for j in range(i + 1):
            for j_sig in (j, j + 1):
                fig.add_trace(go.Scatter(
                    x=[i, i + 1], y=[S_tree[j, i], S_tree[j_sig, i + 1]],
                    mode='lines', line=dict(color=AZUL_CLARO, width=1.5),
                    hoverinfo='skip', showlegend=False,
                ))

    # Nodos
    for i in range(n + 1):
        for j in range(i + 1):
            s_val = S_tree[j, i]
            v_val = V_tree[j, i]

            if modo == "precios":
                texto = f"S: {formato_mxn(s_val)}"
            elif modo == "derivado":
                texto = f"V: {formato_mxn(v_val)}"
            else:
                texto = f"S: {formato_mxn(s_val)}<br>V: {formato_mxn(v_val)}"

            # Valores negativos del derivado en rojo (pasa en el forward)
            color_texto = ROJO if (modo != "precios" and v_val < -0.005) else AZUL_MARINO

            fig.add_trace(go.Scatter(
                x=[i], y=[s_val], mode='markers',
                marker=dict(size=11, color=AZUL_MARINO, line=dict(color="white", width=2)),
                text=texto, hoverinfo='text', showlegend=False,
            ))
            fig.add_annotation(
                x=i, y=s_val + S0 * 0.05, text=texto, showarrow=False,
                font=dict(size=10, color=color_texto), bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=AZUL, borderwidth=1, borderpad=3,
            )
    fig.update_layout(
        title=dict(text=titulo_grafico, font=dict(size=17, color=AZUL_MARINO)),
        xaxis_title="Pasos de tiempo", yaxis_title=titulo_y,
        template='plotly_white', height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, showgrid=False),
        yaxis=dict(gridcolor="#E6EDF5"),
    )
    return fig


# --- Estilo de la página ---
def aplicar_estilos():
    """CSS para las tarjetas de resultados, títulos y barra lateral."""
    st.markdown(f"""
    <style>
    .block-container {{ padding-top: 2rem; }}
    h1, h2, h3 {{ color: {AZUL_MARINO}; }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #DCE6F2;
        border-left: 5px solid {AZUL};
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(13, 43, 78, 0.08);
    }}
    div[data-testid="stMetricValue"] {{ color: {AZUL_MARINO}; }}
    section[data-testid="stSidebar"] {{ border-right: 3px solid {AZUL}; }}
    .subtitulo {{ color: #5B6B7F; font-size: 1.05rem; margin-top: -0.8rem; }}
    </style>
    """, unsafe_allow_html=True)


def encabezado():
    col_logo, col_titulo = st.columns([1, 9], vertical_alignment="center")
    with col_logo:
        st.image("imagen.jpg", width=95)
    with col_titulo:
        st.title("Simulación de derivados en tiempo discreto")
        st.markdown(
            '<p class="subtitulo">Modelo binomial CRR para opciones (europeas, americanas, '
            'asiáticas y lookback) y contratos forward.</p>',
            unsafe_allow_html=True,
        )


# --- Función Principal ---
def main():
    st.set_page_config(
        page_title="Modelo Binomial para Valoración de Derivados",
        page_icon="imagen.jpg", layout="wide", initial_sidebar_state="expanded",
    )
    aplicar_estilos()
    encabezado()

    # --- BARRA LATERAL ---
    st.sidebar.header("Configura tu derivado")
    instrumento = st.sidebar.radio("Instrumento", ("Opción", "Forward"), horizontal=True)
    es_forward = instrumento == "Forward"

    if es_forward:
        option_style = "forward"
        option_type = st.sidebar.selectbox("Posición", ("Larga", "Corta")).lower()
    else:
        option_style = st.sidebar.selectbox("Estilo de la opción", ("Americana", "Europea", "Asiática", "Lookback")).lower()
        option_type = st.sidebar.selectbox("Tipo de la opción", ("Put", "Call")).lower()

    st.sidebar.header("Parámetros del activo y del mercado")
    S0 = st.sidebar.number_input("Precio actual (S0)", min_value=1.0, value=180.0, step=1.0, format="%.2f")
    r = st.sidebar.slider("Tasa libre de riesgo anual", 0.0, 0.20, 0.05, 0.001, format="%.3f")
    sigma = st.sidebar.slider("Volatilidad anual", 0.01, 1.0, 0.25, 0.01, format="%.2f")
    T = st.sidebar.slider("Tiempo al vencimiento (T)", 0.1, 5.0, 1.0, 0.1, format="%.1f")

    if es_forward:
        F0 = S0 * np.exp(r * T)
        usar_justo = st.sidebar.checkbox("Usar el precio forward justo (K = F₀)", value=True)
        if usar_justo:
            K = F0
            st.sidebar.caption(f"K = F₀ = {formato_mxn(F0)}")
        else:
            K = st.sidebar.number_input("Precio de entrega (K)", min_value=1.0, value=float(round(F0, 2)), step=1.0, format="%.2f")
    else:
        K = st.sidebar.number_input("Precio de ejercicio (Strike K)", min_value=1.0, value=180.0, step=1.0, format="%.2f")

    st.sidebar.header("Número de pasos del árbol")
    n = st.sidebar.slider("Número de pasos (n)", 1, 15, 3, 1,
        help="Para opciones exóticas se recomienda un número menor de pasos para que el programa no tarde demasiado."
    )

    # --- CÁLCULOS ---
    params = calcular_parametros(S0, K, r, sigma, T, n)
    precio_opcion, S_tree, V_tree, datos_trayectorias = modelo_crr(
        S0, K, r, sigma, T, n, option_type, option_style, return_trees=True
    )

    # --- RESULTADOS ---
    if es_forward:
        st.subheader(f"Resultados (Forward - posición {option_type})")
        signo = 1 if option_type == 'larga' else -1
        valor_formula = signo * (S0 - K * np.exp(-r * T))

        col1, col2, col3 = st.columns(3)
        col1.metric("Valor del forward (árbol)", f"{formato_mxn(precio_opcion)} MXN")
        col2.metric("Valor con fórmula cerrada", f"{formato_mxn(valor_formula)} MXN")
        col3.metric("Precio forward F₀", f"{formato_mxn(F0)} MXN")

        st.latex(r"F_0 = S_0\,e^{rT} \qquad V_0^{\text{larga}} = S_0 - K\,e^{-rT} \qquad V_0^{\text{corta}} = K\,e^{-rT} - S_0")
        st.info(
            "El forward es una obligación de comprar (larga) o vender (corta) el activo a K en la fecha T. "
            "Por eso el payoff puede ser negativo y no hay ejercicio anticipado. "
            "El árbol y la fórmula dan lo mismo, y si K = F₀ el contrato vale 0 al inicio."
        )
    else:
        st.subheader(f"Resultados ({option_style.capitalize()} - {option_type.capitalize()})")
        col1, col2 = st.columns(2)
        col1.metric("Precio teórico de la opción", f"{formato_mxn(precio_opcion)} MXN")
        col2.metric("Pasos del árbol (n)", n)

    st.divider()

    # --- PARÁMETROS CALCULADOS ---
    st.subheader("Parámetros del modelo")
    params_df = pd.DataFrame({
        "Parámetro": ["Δt", "u (subida)", "d (bajada)", "q (riesgo neutral)", "FD (descuento)"],
        "Valor": [f"{params['dt']:.6f}", f"{params['u']:.6f}", f"{params['d']:.6f}", f"{params['q']:.6f}", f"{params['df']:.6f}"],
    })
    st.dataframe(params_df, hide_index=True, width="stretch")
    st.divider()

    # --- GRÁFICOS Y TABLAS ---
    if option_style in ['asiática', 'asiatica', 'lookback']:
        st.subheader("Árbol binomial (precios del subyacente)")
        st.info("Para opciones exóticas el valor no depende de un solo nodo sino del camino. Revisa la tabla de abajo.")

        fig_tree = graficar_arbol(S_tree, V_tree, n, S0, "Precio del activo (MXN)", f"Árbol de precios - {option_style.capitalize()}", modo="precios")
        st.plotly_chart(fig_tree, width="stretch")

        st.divider()
        st.subheader("Desglose de trayectorias")
        st.dataframe(pd.DataFrame(datos_trayectorias), width="stretch", hide_index=True)
        st.caption("El precio teórico es el valor presente de la suma de (probabilidad × payoff).")

    else:
        # Europea, Americana o Forward: dos árboles lado a lado
        st.subheader("Árboles binomiales")
        col_izq, col_der = st.columns(2)

        with col_izq:
            fig_precios = graficar_arbol(S_tree, V_tree, n, S0, "Precio del activo (MXN)", "1. Árbol del subyacente (S)", modo="precios")
            st.plotly_chart(fig_precios, width="stretch")

        with col_der:
            titulo_v = "2. Árbol del forward (V)" if es_forward else "2. Árbol del derivado (V)"
            fig_valores = graficar_arbol(S_tree, V_tree, n, S0, "Valor (MXN)", titulo_v, modo="derivado")
            st.plotly_chart(fig_valores, width="stretch")


if __name__ == "__main__":
    main()
