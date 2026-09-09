# mdp_corregido.py
# CC3104 - Aprendizaje por Refuerzo | Grupo 1 - Entregables 1.3 y 1.4
#
# Version corregida del MDP de gestion de inventario farmaceutico, lista
# para importar y compartir con los Grupos 3, 4 y 6 durante la sesion
# presencial. El desarrollo completo, la justificacion de cada cambio y la
# verificacion contra los contraejemplos de los entregables 1.1 y 1.2 estan
# en parcial_practico.ipynb; este archivo es solo el codigo, sin la prosa.
#
# Estado corregido:
#   (inventario, dias_vencimiento, demanda_nivel, unidades_en_transito, tendencia_demanda)
# Acciones: [0, 10, 20, 30, 40, 50]

import numpy as np

# ---------------------------------------------------------------------------
# ESPACIO DE ESTADOS
# ---------------------------------------------------------------------------

# dias_vencimiento y demanda_nivel: sin cambio respecto al MDP original.
DIAS_VENCIMIENTO = [1, 7, 14, 30, 60]
NIVELES_DEMANDA = ['bajo', 'medio', 'alto', 'crítico']
DEMAND_MAP = {'bajo': 5, 'medio': 15, 'alto': 25, 'crítico': 40}

# unidades_en_transito [NUEVO]: pedido realizado el dia anterior, aun en
# camino. Restaura la propiedad de Markov: con el mismo estado observado
# (20, 14, 'medio') y accion 0, el inventario siguiente del MDP original
# depende de si ayer se pidieron 30 unidades o no, algo que el estado no
# podia distinguir (entregable 1.1).
UNIDADES_TRANSITO = [0, 10, 20, 30, 40, 50]

# tendencia_demanda [NUEVO]: direccion del cambio de demanda en el corto
# plazo. La transicion original congela la demanda (new_demand =
# demand_level); con la tendencia en el estado, la demanda puede cambiar de
# forma probabilistica en vez de quedarse fija todo el episodio (entregable 1.2).
TENDENCIAS_DEMANDA = ['bajando', 'estable', 'subiendo']

# El comentario de cabecera del codigo original del examen declara 10
# niveles de inventario (multiplos de 10 entre 0 y 100), pero esa lista
# tiene en realidad 11 valores, y la propia funcion transition() original
# tampoco discretiza el inventario a esa grilla (con demanda 'bajo' el
# inventario cae en 55, 45, que no son multiplos de 10). Usamos el numero
# que declara el comentario (10) para que el factor de expansion de
# tabla_impacto() sea comparable con los 200 estados que ya usa el Grupo 7
# en su entregable 7.2. Si se usa el conteo literal de 11 niveles, el factor
# de expansion no cambia (verificado en parcial_practico.ipynb).
NIVELES_INVENTARIO_DECLARADOS = 10

# Probabilidad de que la demanda pase a cada nivel, condicionada a la
# tendencia actual y al nivel actual. Es un supuesto ilustrativo, no
# calibrado con datos reales de la farmacia, que hace que la demanda suba
# con mas probabilidad cuando la tendencia es 'subiendo', y baje con mas
# probabilidad cuando es 'bajando'.
PROB_DEMANDA = {
    'bajando': {
        'bajo':    {'bajo': 0.80, 'medio': 0.15, 'alto': 0.04, 'crítico': 0.01},
        'medio':   {'bajo': 0.40, 'medio': 0.45, 'alto': 0.12, 'crítico': 0.03},
        'alto':    {'bajo': 0.10, 'medio': 0.35, 'alto': 0.45, 'crítico': 0.10},
        'crítico': {'bajo': 0.05, 'medio': 0.20, 'alto': 0.35, 'crítico': 0.40},
    },
    'estable': {
        'bajo':    {'bajo': 0.70, 'medio': 0.20, 'alto': 0.08, 'crítico': 0.02},
        'medio':   {'bajo': 0.15, 'medio': 0.65, 'alto': 0.15, 'crítico': 0.05},
        'alto':    {'bajo': 0.05, 'medio': 0.20, 'alto': 0.60, 'crítico': 0.15},
        'crítico': {'bajo': 0.02, 'medio': 0.08, 'alto': 0.25, 'crítico': 0.65},
    },
    'subiendo': {
        'bajo':    {'bajo': 0.50, 'medio': 0.30, 'alto': 0.15, 'crítico': 0.05},
        'medio':   {'bajo': 0.05, 'medio': 0.45, 'alto': 0.35, 'crítico': 0.15},
        'alto':    {'bajo': 0.02, 'medio': 0.10, 'alto': 0.48, 'crítico': 0.40},
        'crítico': {'bajo': 0.01, 'medio': 0.04, 'alto': 0.20, 'crítico': 0.75},
    },
}

# Probabilidad de que la tendencia misma cambie (cadena de Markov sobre la
# tendencia). Es persistente, alta probabilidad de mantenerse, pero puede
# transicionar de forma gradual.
PROB_TENDENCIA = {
    'bajando':  {'bajando': 0.60, 'estable': 0.35, 'subiendo': 0.05},
    'estable':  {'bajando': 0.20, 'estable': 0.60, 'subiendo': 0.20},
    'subiendo': {'bajando': 0.05, 'estable': 0.35, 'subiendo': 0.60},
}


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------

def discretizar_inventario(inv):
    """Redondea el inventario al multiplo de 10 mas cercano, acotado en [0, 100]."""
    return int(min(100, max(0, round(inv / 10.0)))) * 10


def discretizar_dias(dias):
    """Mapea los dias restantes al valor mas cercano dentro de la grilla declarada."""
    return min(DIAS_VENCIMIENTO, key=lambda x: abs(x - dias))


# ---------------------------------------------------------------------------
# FUNCION DE TRANSICION CORREGIDA
# ---------------------------------------------------------------------------

def transition(state, action):
    """Transicion estocastica del MDP corregido.

    state = (inventario, dias_vencimiento, demanda_nivel, unidades_en_transito, tendencia)
    action: unidades a pedir, en {0, 10, 20, 30, 40, 50}

    Retorna (estado_siguiente, demanda_no_atendida). La demanda no atendida
    es la cantidad exacta de unidades que faltaron en el paso, algo que la
    transicion original descartaba al recortar el inventario con max(0, ...).
    Se expone aqui por si resulta util para quien diseñe la funcion de
    recompensa, no porque el enunciado lo exija de forma explicita.
    """
    inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia = state

    # El pedido de ayer llega hoy antes de servir la demanda. Este es el
    # cambio que restaura la propiedad de Markov (entregable 1.1).
    inventario_efectivo = min(100, inventario + en_transito)

    demanda_hoy = DEMAND_MAP[demanda_nivel]
    demanda_no_atendida = max(0, demanda_hoy - inventario_efectivo)
    new_inventory = discretizar_inventario(max(0, inventario_efectivo - demanda_hoy))
    new_days = discretizar_dias(max(1, dias_vencimiento - 1))

    # Muestreamos la demanda y la tendencia siguientes; a diferencia del
    # original (demanda congelada), aqui pueden cambiar (entregable 1.2).
    probs_d = PROB_DEMANDA[tendencia][demanda_nivel]
    new_demand = str(np.random.choice(list(probs_d.keys()), p=list(probs_d.values())))

    probs_t = PROB_TENDENCIA[tendencia]
    new_tendencia = str(np.random.choice(list(probs_t.keys()), p=list(probs_t.values())))

    # El pedido de hoy queda en transito para manana.
    new_transito = action

    next_state = (new_inventory, new_days, new_demand, new_transito, new_tendencia)
    return next_state, int(demanda_no_atendida)


def transition_determinista(state, action):
    """Version determinista de transition(): usa el valor mas probable de cada
    distribucion en vez de muestrear. Util para depurar y para comparar
    contra el original sin ruido aleatorio de por medio. Misma firma que
    transition(), compatible con el resto del sistema."""
    inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia = state

    inventario_efectivo = min(100, inventario + en_transito)
    demanda_hoy = DEMAND_MAP[demanda_nivel]
    demanda_no_atendida = max(0, demanda_hoy - inventario_efectivo)
    new_inventory = discretizar_inventario(max(0, inventario_efectivo - demanda_hoy))
    new_days = discretizar_dias(max(1, dias_vencimiento - 1))

    probs_d = PROB_DEMANDA[tendencia][demanda_nivel]
    new_demand = max(probs_d, key=probs_d.get)

    probs_t = PROB_TENDENCIA[tendencia]
    new_tendencia = max(probs_t, key=probs_t.get)

    next_state = (new_inventory, new_days, new_demand, action, new_tendencia)
    return next_state, int(demanda_no_atendida)


# ---------------------------------------------------------------------------
# COMPATIBILIDAD CON EL FORMATO ORIGINAL (para los Grupos 3, 4 y 6)
# ---------------------------------------------------------------------------

def estado_a_original(state):
    """Proyecta el estado corregido (5 variables) al formato original (3
    variables), para que el codigo de otros grupos siga funcionando sin
    reescribirse."""
    inventario, dias_vencimiento, demanda_nivel, _, _ = state
    return (inventario, dias_vencimiento, demanda_nivel)


def estado_desde_original(state, en_transito=0, tendencia='estable'):
    """Expande un estado original (3 variables) al formato corregido (5
    variables), con valores neutros por defecto cuando no hay informacion
    adicional disponible."""
    inventario, dias_vencimiento, demanda_nivel = state
    return (inventario, dias_vencimiento, demanda_nivel, en_transito, tendencia)


# ---------------------------------------------------------------------------
# ENTREGABLE 1.4: IMPACTO SOBRE EL ESPACIO DE ESTADOS
# ---------------------------------------------------------------------------

def tabla_impacto():
    """Imprime la comparacion del espacio de estados original vs corregido
    y devuelve los totales, para que se puedan reusar en otro analisis."""
    acciones = 6
    dias = len(DIAS_VENCIMIENTO)
    demandas = len(NIVELES_DEMANDA)
    transitos = len(UNIDADES_TRANSITO)
    tendencias = len(TENDENCIAS_DEMANDA)

    total_original = NIVELES_INVENTARIO_DECLARADOS * dias * demandas
    total_corregido = NIVELES_INVENTARIO_DECLARADOS * dias * demandas * transitos * tendencias

    print(f'{"Dimension":38} {"Original":>10} {"Corregido":>12}')
    print(f'{"Niveles de inventario":38} {NIVELES_INVENTARIO_DECLARADOS:>10} {NIVELES_INVENTARIO_DECLARADOS:>12}')
    print(f'{"Dias hasta vencimiento":38} {dias:>10} {dias:>12}')
    print(f'{"Niveles de demanda":38} {demandas:>10} {demandas:>12}')
    print(f'{"Unidades en transito (nuevo)":38} {"---":>10} {transitos:>12}')
    print(f'{"Tendencia de demanda (nuevo)":38} {"---":>10} {tendencias:>12}')
    print()
    print(f'{"Total de estados":38} {total_original:>10} {total_corregido:>12}')
    print(f'{"Pares (estado, accion)":38} {total_original*acciones:>10} {total_corregido*acciones:>12}')
    print(f'{"Entradas de la tabla Q":38} {total_original*acciones:>10} {total_corregido*acciones:>12}')
    print()
    print('Factor de expansion:', total_corregido / total_original)

    return {
        'total_original': total_original,
        'total_corregido': total_corregido,
        'factor_expansion': total_corregido / total_original,
    }


# ---------------------------------------------------------------------------
# TESTS DE VERIFICACION
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('--- Tests de compatibilidad y transicion ---')
    print()

    # El estado corregido tiene 5 componentes.
    s0 = (20, 14, 'medio', 0, 'estable')
    assert len(s0) == 5

    # Proyeccion al formato original y expansion de vuelta.
    assert estado_a_original(s0) == (20, 14, 'medio')
    assert estado_desde_original((20, 14, 'medio')) == (20, 14, 'medio', 0, 'estable')

    # Transicion determinista retorna 5 variables y stockout entero.
    s_next, stockout = transition_determinista(s0, action=10)
    assert len(s_next) == 5
    assert isinstance(stockout, int)

    # Un pedido en transito deja mas inventario que no tenerlo.
    sn_con, _ = transition_determinista((10, 14, 'medio', 30, 'estable'), 0)
    sn_sin, _ = transition_determinista((10, 14, 'medio', 0, 'estable'), 0)
    assert sn_con[0] > sn_sin[0]

    # La demanda puede cambiar en la transicion estocastica, y el estado
    # sigue siendo compatible con la categoria 'crítico' con tilde que usa
    # el codigo original del examen (S10 - Codigo Examen.py).
    np.random.seed(0)
    demandas_observadas = {
        transition((20, 14, 'crítico', 0, 'subiendo'), 0)[0][2]
        for _ in range(100)
    }
    assert len(demandas_observadas) > 1

    print('Tests de transicion y compatibilidad: OK')
    print()

    tabla_impacto()
