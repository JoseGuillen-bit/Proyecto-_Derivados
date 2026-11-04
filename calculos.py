import numpy as np
import pandas as pd
import itertools


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

