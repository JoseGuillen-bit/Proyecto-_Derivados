import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools

# Paleta de colores. La sacamos del logo para que todo combine
AZUL_MARINO = "#0D2B4E"
AZUL = "#1F6FB5"
AZUL_CLARO = "#4A9FE0"
ROJO = "#832D0B"


def formato_mxn(x):
    """Pone los números como dinero, por ejemplo $1,234.56 o -$1,234.56."""
    # Si el número es casi cero lo dejamos en cero, para que no salga un "-$0.00" feo
    if abs(x) < 0.005:
        x = 0.0
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def calcular_parametros(S0, K, r, sigma, T, n):
    """Calcula los parámetros del árbol: Δt, u, d, q y el factor de descuento."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)
    return {'dt': dt, 'u': u, 'd': d, 'q': q, 'df': df}


def modelo_crr(S0, K, r, sigma, T, n, option_type='call', option_style='americana', return_trees=False):
    """Valúa una opción o un forward con el modelo binomial de Cox-Ross-Rubinstein."""
    params = calcular_parametros(S0, K, r, sigma, T, n)
    u, d, q, df = params['u'], params['d'], params['q'], params['df']

    # Armamos el árbol de precios del subyacente.
    # La columna i es el paso de tiempo y la fila j cuántas veces ha bajado el precio
    S_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            S_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    es_asiatica = option_style.lower().startswith('asi')
    es_lookback = option_style.lower() == 'lookback'
    es_forward = option_style.lower() == 'forward'

    # ---------- FORWARD ----------
    # Aquí no hay max(·, 0) porque el forward es obligatorio: si te va mal, pierdes.
    # Tampoco hay ejercicio anticipado, el contrato se liquida en T y ya
    if es_forward:
        V_tree = np.zeros((n + 1, n + 1))
        if option_type.lower() == 'larga':
            V_tree[:, n] = S_tree[:, n] - K
        else:
            V_tree[:, n] = K - S_tree[:, n]

        # Nos regresamos nodo por nodo: promedio con q y descontamos un paso
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                V_tree[j, i] = df * (q * V_tree[j, i + 1] + (1 - q) * V_tree[j + 1, i + 1])

        if return_trees:
            return V_tree[0, 0], S_tree, V_tree, None
        return V_tree[0, 0]

    # ---------- EXÓTICAS (asiática y lookback) ----------
    # Estas dependen de todo el camino que siguió el precio, no solo de dónde terminó.
    # Por eso recorremos todos los caminos posibles (son 2^n) uno por uno
    if es_asiatica or es_lookback:
        caminos = list(itertools.product([1, 0], repeat=n))
        payoff_total = 0.0
        datos_trayectorias = []

        for idx, camino in enumerate(caminos):
            S_actual = S0
            precios_camino = [S0]
            subidas = sum(camino)

            # 1 es subida y 0 es bajada
            for paso in camino:
                if paso == 1: S_actual *= u
                else:         S_actual *= d
                precios_camino.append(S_actual)

            payoff = 0.0
            metrica = 0.0

            if es_asiatica:
                # La asiática paga sobre el promedio de los precios del camino
                metrica = np.mean(precios_camino)
                if option_type.lower() == 'call': payoff = max(metrica - K, 0)
                else: payoff = max(K - metrica, 0)
                nombre_metrica = "Promedio"

            elif es_lookback:
                # La lookback "mira hacia atrás": el call usa el máximo y el put el mínimo
                if option_type.lower() == 'call':
                    metrica = max(precios_camino)
                    payoff = max(metrica - K, 0)
                    nombre_metrica = "Máximo"
                else:
                    metrica = min(precios_camino)
                    payoff = max(K - metrica, 0)
                    nombre_metrica = "Mínimo"

            # Probabilidad riesgo neutral de este camino en particular
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

        # El payoff esperado está en T, así que lo traemos a hoy con todo el plazo (e^(-rT)),
        # no con un solo paso
        precio_opcion = payoff_total * np.exp(-r * T)
        # Para estas no hay árbol de valores, pero mandamos uno en ceros para que la gráfica no truene
        V_tree = np.zeros((n + 1, n + 1))
        if return_trees:
            return precio_opcion, S_tree, V_tree, datos_trayectorias
        return precio_opcion

    # ---------- EUROPEAS Y AMERICANAS ----------
    # Empezamos por el final: en T la opción vale su payoff
    V_tree = np.zeros((n + 1, n + 1))
    if option_type.lower() == 'call':
        V_tree[:, n] = np.maximum(S_tree[:, n] - K, 0)
    else:
        V_tree[:, n] = np.maximum(K - S_tree[:, n], 0)

    # Y vamos hacia atrás hasta llegar a hoy
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont_value = df * (q * V_tree[j, i + 1] + (1 - q) * V_tree[j + 1, i + 1])

            # En la americana comparamos esperar contra ejercer ya, y nos quedamos con lo que más convenga.
            # Ojo: el menú manda 'americana' en español, por eso la comparación es con esa palabra
            if option_style.lower() == 'americana':
                if option_type.lower() == 'call':
                    exercise_value = max(S_tree[j, i] - K, 0)
                else:
                    exercise_value = max(K - S_tree[j, i], 0)
                V_tree[j, i] = max(cont_value, exercise_value)
            else:
                # La europea solo se puede ejercer al final, así que siempre toca esperar
                V_tree[j, i] = cont_value

    if return_trees:
        return V_tree[0, 0], S_tree, V_tree, None
    return V_tree[0, 0]


def _cobertura(S, Vu, Vd, u, d, df):
    # α es cuántas acciones necesitas y β cuánto dinero va al bono,
    # para que el portafolio valga lo mismo que el derivado suba o baje el precio
    alpha = (Vu - Vd) / (S * (u - d))
    beta = df * (u * Vd - d * Vu) / (u - d)
    return alpha, beta


def portafolio_cobertura(S0, K, r, sigma, T, n, option_type, option_style):
    """
    Arma una tabla con el portafolio réplica (α, β) en cada nodo del árbol.
    En las exóticas va por historia, porque ahí importa el camino completo.
    """
    params = calcular_parametros(S0, K, r, sigma, T, n)
    u, d, q, df = params['u'], params['d'], params['q'], params['df']
    estilo = option_style.lower()
    tipo = option_type.lower()
    filas = []

    # Exóticas: dos caminos que llegan al mismo precio pueden valer distinto,
    # así que cada historia (↑↓↑...) lleva su propio portafolio
    if estilo.startswith('asi') or estilo == 'lookback':
        def payoff_camino(camino):
            precios = [S0]
            for paso in camino:
                precios.append(precios[-1] * (u if paso == 1 else d))
            if estilo.startswith('asi'):
                m = np.mean(precios)
                return max(m - K, 0) if tipo == 'call' else max(K - m, 0)
            if tipo == 'call':
                return max(max(precios) - K, 0)
            return max(K - min(precios), 0)

        # Guardamos el valor de cada historia, empezando por las completas (las de n pasos)
        V = {c: payoff_camino(c) for c in itertools.product([1, 0], repeat=n)}

        # Vamos recortando las historias un paso a la vez hasta llegar al inicio
        for k in range(n - 1, -1, -1):
            for pref in itertools.product([1, 0], repeat=k):
                Vu, Vd = V[pref + (1,)], V[pref + (0,)]
                V[pref] = df * (q * Vu + (1 - q) * Vd)
                ups = sum(pref)
                S = S0 * (u ** ups) * (d ** (k - ups))
                alpha, beta = _cobertura(S, Vu, Vd, u, d, df)
                filas.append({
                    "Paso": k,
                    "Nodo": "".join("↑" if p == 1 else "↓" for p in pref) or "Inicio",
                    "S": S, "alpha": alpha, "beta": beta,
                    "Valor": alpha * S + beta,
                })

    # Europeas, americanas y forward: aquí el árbol sí se junta,
    # así que basta con calcular el portafolio en cada nodo
    else:
        _, S_tree, V_tree, _ = modelo_crr(
            S0, K, r, sigma, T, n, option_type, option_style, return_trees=True
        )
        for i in range(n):
            for j in range(i + 1):
                S = S_tree[j, i]
                Vu, Vd = V_tree[j, i + 1], V_tree[j + 1, i + 1]
                V_nodo = V_tree[j, i]
                alpha, beta = _cobertura(S, Vu, Vd, u, d, df)

                # Si en este nodo la americana vale más que esperar, es porque conviene ejercer.
                # En ese caso el portafolio es simplemente el del ejercicio inmediato
                ejerce = estilo == 'americana' and V_nodo > (alpha * S + beta) + 1e-9
                if ejerce:
                    alpha = -1.0 if tipo == 'put' else 1.0
                    beta = K if tipo == 'put' else -K

                filas.append({
                    "Paso": i,
                    "Nodo": f"{i - j}↑ {j}↓",
                    "S": S, "alpha": alpha, "beta": beta,
                    "Valor": alpha * S + beta,
                })

    return pd.DataFrame(filas).sort_values("Paso", kind="stable").reset_index(drop=True)


# ---------- Gráfica de los árboles ----------
def graficar_arbol(S_tree, V_tree, n, S0, titulo_y, titulo_grafico, modo="precios"):
    """
    Dibuja el árbol. Según el modo, en cada nodo se ve:
    "precios" -> solo S
    "derivado" -> solo V
    "ambos" -> S y V juntos
    """
    fig = go.Figure()

    # Primero las líneas que conectan cada nodo con sus dos hijos (el de subida y el de bajada)
    for i in range(n):
        for j in range(i + 1):
            for j_sig in (j, j + 1):
                fig.add_trace(go.Scatter(
                    x=[i, i + 1], y=[S_tree[j, i], S_tree[j_sig, i + 1]],
                    mode='lines', line=dict(color=AZUL_CLARO, width=1.5),
                    hoverinfo='skip', showlegend=False,
                ))

    # Luego los puntitos con su etiqueta
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

            # Si el derivado vale negativo (pasa con el forward) lo pintamos de rojo para que se note
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


# ---------- Estilo de la página ----------
def aplicar_estilos():
    """Un poco de CSS para que las tarjetas, los títulos y la barra lateral se vean mejor."""
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
    # Logo a la izquierda y título a la derecha
    col_logo, col_titulo = st.columns([1, 9], vertical_alignment="center")
    with col_logo:
        st.image("imagen.jpg", width=95)
    with col_titulo:
        st.title("Simulación de derivados en tiempo discreto")
        st.markdown(
            '<p class="subtitulo">Modelo binomial para opciones (europeas, americanas, '
            'asiáticas y lookback) y contratos forward.</p>',
            unsafe_allow_html=True,
        )


# ---------- Página principal ----------
def main():
    st.set_page_config(
        page_title="Modelo Binomial para Valoración de Derivados",
        page_icon="imagen.jpg", layout="wide", initial_sidebar_state="expanded",
    )
    aplicar_estilos()
    encabezado()

    # Barra lateral: aquí el usuario escoge qué quiere valuar
    st.sidebar.header("Configura tu derivado")
    instrumento = st.sidebar.radio("Instrumento", ("Opción", "Forward"), horizontal=True)
    es_forward = instrumento == "Forward"

    # Si es forward se elige la posición; si es opción, el estilo y si es call o put
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

    # El K va al final porque en el forward depende de S0, r y T (F0 = S0·e^(rT))
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

    # Con todo lo que eligió el usuario, hacemos las cuentas
    params = calcular_parametros(S0, K, r, sigma, T, n)
    precio_opcion, S_tree, V_tree, datos_trayectorias = modelo_crr(
        S0, K, r, sigma, T, n, option_type, option_style, return_trees=True
    )

    # Resultados principales
    if es_forward:
        st.subheader(f"Resultados (Forward - posición {option_type})")
        signo = 1 if option_type == 'larga' else -1
        valor_formula = signo * (S0 - K * np.exp(-r * T))

        # Ponemos lado a lado lo que da el árbol y lo que da la fórmula, para ver que coinciden
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

    # Parámetros del modelo, en tarjetas igual que los resultados de arriba
    st.subheader("Parámetros del modelo")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Δt", f"{params['dt']:.4f}")
    c2.metric("u (subida)", f"{params['u']:.4f}")
    c3.metric("d (bajada)", f"{params['d']:.4f}")
    c4.metric("q (riesgo neutral)", f"{params['q']:.4f}")
    c5.metric("FD (descuento)", f"{params['df']:.4f}")
    st.divider()

    # Gráficas y tablas
    if option_style in ['asiática', 'asiatica', 'lookback']:
        # En las exóticas el valor no vive en un nodo, así que mostramos el árbol de precios
        # y luego la tabla con cada camino
        st.subheader("Árbol binomial (precios del subyacente)")
        st.info("Para opciones exóticas el valor no depende de un solo nodo sino del camino. Revisa la tabla de abajo.")

        fig_tree = graficar_arbol(S_tree, V_tree, n, S0, "Precio del activo (MXN)", f"Árbol de precios - {option_style.capitalize()}", modo="precios")
        st.plotly_chart(fig_tree, width="stretch")

        st.divider()
        st.subheader("Desglose de trayectorias")
        st.dataframe(pd.DataFrame(datos_trayectorias), width="stretch", hide_index=True)
        st.caption("El precio teórico es el valor presente de la suma de (probabilidad × payoff).")

    else:
        # Europea, americana o forward: el árbol del subyacente y el del derivado, uno junto al otro
        st.subheader("Árboles binomiales")
        col_izq, col_der = st.columns(2)

        with col_izq:
            fig_precios = graficar_arbol(S_tree, V_tree, n, S0, "Precio del activo (MXN)", "1. Árbol del subyacente (S)", modo="precios")
            st.plotly_chart(fig_precios, width="stretch")

        with col_der:
            titulo_v = "2. Árbol del forward (V)" if es_forward else "2. Árbol del derivado (V)"
            fig_valores = graficar_arbol(S_tree, V_tree, n, S0, "Valor (MXN)", titulo_v, modo="derivado")
            st.plotly_chart(fig_valores, width="stretch")

    # Portafolio de cobertura: sale para cualquier derivado que se elija
    st.divider()
    st.subheader("Portafolio de cobertura (réplica)")
    st.latex(r"\alpha = \frac{V_u - V_d}{S\,(u-d)} \qquad \beta = e^{-r\Delta t}\,\frac{u\,V_d - d\,V_u}{u-d}")

    df_cob = portafolio_cobertura(S0, K, r, sigma, T, n, option_type, option_style)

    # Primero el portafolio con el que arrancas hoy (t = 0)
    ini = df_cob.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("α inicial (acciones)", f"{ini['alpha']:.4f}")
    c2.metric("β inicial (bono)", f"{formato_mxn(ini['beta'])} MXN")
    c3.metric("Valor del portafolio (αS₀ + β)", f"{formato_mxn(ini['Valor'])} MXN")

    # Y luego cómo va cambiando: el usuario escoge el paso y ve todos los nodos de ese paso
    paso_sel = st.selectbox(
        "Paso del árbol", list(range(n)),
        format_func=lambda p: "t = 0 (inicio)" if p == 0 else f"Paso {p}",
    )
    tabla = df_cob[df_cob["Paso"] == paso_sel].copy()
    tabla_vista = pd.DataFrame({
        "Nodo": tabla["Nodo"],
        "Precio del activo (S)": tabla["S"].map(formato_mxn),
        "α (acciones)": tabla["alpha"].map(lambda x: f"{x:.4f}"),
        "β (bono, MXN)": tabla["beta"].map(formato_mxn),
        "Valor del portafolio": tabla["Valor"].map(formato_mxn),
    })
    st.dataframe(tabla_vista, hide_index=True, width="stretch")
    st.caption(
        "α > 0: comprar acciones; α < 0: vender en corto. β > 0: invertir en el bono; β < 0: pedir prestado. "
        "Este portafolio replica al derivado, así que quien lo vende se cubre tomando la posición contraria."
    )


if __name__ == "__main__":
    main()
