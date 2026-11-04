import streamlit as st
import plotly.graph_objects as go

# Paleta de colores. La sacamos del logo para que todo combine
AZUL_MARINO = "#0D2B4E"
AZUL = "#1F6FB5"
AZUL_CLARO = "#4A9FE0"
ROJO = "#832D0B"


def formato_mxn(x):
    """Pone los números como dinero"""
    # Si el número es casi cero lo dejamos en cero, para que no salga un "-$0.00" feo
    if abs(x) < 0.005:
        x = 0.0
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


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