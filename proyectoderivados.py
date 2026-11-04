import streamlit as st
import numpy as np
import pandas as pd
import cálculos
import interfaz

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
