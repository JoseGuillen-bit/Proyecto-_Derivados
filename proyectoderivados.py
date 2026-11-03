import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools

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

    # Árbol visual de precios (para graficar)
    S_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            S_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # --- LÓGICA PARA OPCIONES EXÓTICAS (Caminos exactos) ---
    # Usamos startswith('asi') para evitar problemas con tildes (Asiática / asiatica)
    es_asiatica = option_style.lower().startswith('asi')
    es_lookback = option_style.lower() == 'lookback'

    if es_asiatica or es_lookback:
        caminos = list(itertools.product([1, 0], repeat=n))
        payoff_total = 0.0
        datos_trayectorias = [] 
        
        for idx, camino in enumerate(caminos):
            S_actual = S0
            precios_camino = [S0]
            subidas = sum(camino)
            
            # Reconstruimos el camino paso a paso
            for paso in camino:
                if paso == 1: # Sube
                    S_actual *= u
                else:         # Baja
                    S_actual *= d
                precios_camino.append(S_actual)
            
            # Inicializamos variables para evitar errores
            payoff = 0.0
            metrica = 0.0
            
            # Calculamos el payoff según el tipo de opción
            if es_asiatica:
                metrica = np.mean(precios_camino)
                if option_type.lower() == 'call':
                    payoff = max(metrica - K, 0)
                else:
                    payoff = max(K - metrica, 0)
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
            
            # Probabilidad de este camino específico
            prob = (q ** subidas) * ((1 - q) ** (n - subidas))
            
            # Sumamos al valor esperado
            payoff_total += prob * payoff
            
            # Guardamos los datos para la tabla
            camino_str = "".join(["↑" if p == 1 else "↓" for p in camino])
            datos_trayectorias.append({
                "Camino": camino_str,
                "Precio Final": round(precios_camino[-1], 2),
                f"{nombre_metrica} del Camino": round(metrica, 2),
                "Probabilidad": f"{prob*100:.2f}%",
                "Payoff (Vencimiento)": round(payoff, 2)
            })
            
        # Descontamos a valor presente
        precio_opcion = payoff_total * df
        
        # Creamos un V_tree falso solo para que el gráfico no falle
        V_tree = np.zeros((n + 1, n + 1))
        if return_trees:
            return precio_opcion, S_tree, V_tree, datos_trayectorias
        return precio_opcion

    # --- LÓGICA PARA OPCIONES EUROPEAS Y AMERICANAS (Árbol rápido) ---
    else:
        V_tree = np.zeros((n + 1, n + 1))
        if option_type.lower() == 'call':
            V_tree[:, n] = np.maximum(S_tree[:, n] - K, 0)
        else:
            V_tree[:, n] = np.maximum(K - S_tree[:, n], 0)

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
            return V_tree[0, 0], S_tree, V_tree, None
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
        "Esta aplicación valora opciones de tipo Call/Put, europeas, americanas y exóticas de primera generación "
        "utilizando árboles binomiales."
    )

    # --- BARRA LATERAL ---
    st.sidebar.header("Configura tu opción")
    
    option_style = st.sidebar.selectbox(
        "Estilo de la opción", ("Americana", "Europea", "Asiática", "Lookback")
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
        "Número de pasos (n)", min_value=1, max_value=15, value=3, step=1,
        help="Para opciones Asiáticas y Lookback, valores altos generan muchas filas en la tabla."
    )

    # --- CÁLCULOS ---
    params = calcular_parametros(S0, K, r, sigma, T, n)
    precio_opcion, S_tree, V_tree, datos_trayectorias = modelo_crr(
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
    st.subheader("Árbol binomial de precios")
    
    if option_style in ['asiática', 'asiatica', 'lookback']:
        st.info("**Nota:** Para estas opciones exóticas, el cálculo se realiza evaluando **todos los caminos posibles exactos**. Revisa la tabla de abajo para ver el desglose.")
    
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

    # Nodos (Solo mostramos los precios del activo S)
    for i in range(n + 1):
        for j in range(i + 1):
            s_val = S_tree[j, i]
            texto = f"S: ${s_val:.2f}"
                
            fig_tree.add_trace(go.Scatter(
                x=[i], y=[s_val], mode='markers',
                marker=dict(size=12, color='darkblue'),
                text=texto,
                hoverinfo='text', showlegend=False,
            ))
            fig_tree.add_annotation(
                x=i, y=s_val + S0 * 0.05,
                text=texto,
                showarrow=False,
                font=dict(size=9, color="black"),
                bgcolor="rgba(255, 255, 200, 0.7)",
                bordercolor="orange",
                borderwidth=1,
                borderpad=2,
            )

    fig_tree.update_layout(
        title=dict(
            text=f"Árbol de Precios - Opción {option_type.capitalize()} {option_style.capitalize()}",
            font=dict(size=20),
        ),
        xaxis_title="Pasos de tiempo",
        yaxis_title="Precio del activo (MXN)",
        template='simple_white',
        height=600,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- TABLA DE TRAYECTORIAS (Solo para exóticas) ---
    if datos_trayectorias is not None:
        st.divider()
        st.subheader("Desglose de Trayectorias")
        st.markdown("Esta tabla muestra el cálculo exacto de cada camino posible hasta el vencimiento.")
        
        df_trayectorias = pd.DataFrame(datos_trayectorias)
        
        # Mostramos la tabla con un diseño bonito
        st.dataframe(
            df_trayectorias, 
            use_container_width=True, 
            hide_index=True
        )
        
        st.caption("↑ = Subida (u) | ↓ = Bajada (d). El Precio Teórico final es el Valor Presente de la suma de (Probabilidad × Payoff) de todos los caminos.")

if __name__ == "__main__":
    main()