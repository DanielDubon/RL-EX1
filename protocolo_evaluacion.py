# Entregable 7.3 (codigo) - protocolo de evaluacion de 3 capas.
# Implementa el protocolo descrito en la celda 8 del notebook: gate de
# cobertura, prueba de estres en estados de baja frecuencia, y proyeccion
# de KPIs de negocio. Se corre sobre el entorno y las politicas del Grupo 5
# (exploracion_grupo5.py), que ya entrega Q y sa_pairs_visited listos.

import numpy as np
from exploracion_grupo5 import (
    PharmacyInventoryEnv, train_epsilon_greedy, train_optimistic_ucb,
)

# ---------------------------------------------------------------------------
# Capa 1: gate de cobertura de entrenamiento
# ---------------------------------------------------------------------------

def gate_cobertura(env, sa_pairs_visited,
                    min_pares_pct=0.90, min_estados_pct=0.95,
                    min_pares_pct_por_demanda=0.80):
    """Bloquea el despliegue si la cobertura de entrenamiento es insuficiente.

    Rechaza antes de mirar reward o KPIs: cobertura insuficiente invalida
    cualquier metrica calculada despues (ver Entregable 7.2).
    """
    total_sa = env.n_states * env.n_actions
    cobertura_pares = len(sa_pairs_visited) / total_sa

    estados_visitados = {s for s, _ in sa_pairs_visited}
    cobertura_estados = len(estados_visitados) / env.n_states

    cobertura_por_demanda = {}
    for dem in env.demand_levels:
        estados_dem = [s for s in env.all_states if s[2] == dem]
        pares_dem = len(estados_dem) * env.n_actions
        pares_dem_visitados = sum(
            1 for s in estados_dem for a in range(env.n_actions)
            if (s, a) in sa_pairs_visited
        )
        cobertura_por_demanda[dem] = pares_dem_visitados / pares_dem

    aprobado = (
        cobertura_pares >= min_pares_pct
        and cobertura_estados >= min_estados_pct
        and all(c >= min_pares_pct_por_demanda for c in cobertura_por_demanda.values())
    )

    return {
        'capa': 'cobertura',
        'aprobado': aprobado,
        'cobertura_pares_pct': round(cobertura_pares * 100, 1),
        'cobertura_estados_pct': round(cobertura_estados * 100, 1),
        'cobertura_por_demanda_pct': {k: round(v * 100, 1) for k, v in cobertura_por_demanda.items()},
    }


# ---------------------------------------------------------------------------
# Capa 2: prueba de estres en estados de baja frecuencia
# ---------------------------------------------------------------------------

def _es_estado_de_estres(state, umbral_inventario=30):
    inventory, _, demand = state
    return inventory <= umbral_inventario and demand in ('alto', 'crítico')


def _tasa_stockout_simulado(env, Q, estados):
    """Fraccion de estados donde la accion greedy deja inventario+pedido
    por debajo de la demanda diaria (mismo criterio que Entregable 7.2)."""
    if not estados:
        return 0.0
    n_stockout = 0
    for s in estados:
        inventory, _, demand = s
        demanda_val = env.demand_map[demand]
        accion_idx = int(np.argmax(Q[s]))
        accion_val = env.actions[accion_idx]
        if inventory + accion_val < demanda_val:
            n_stockout += 1
    return n_stockout / len(estados)


def prueba_estres(env, Q, umbral_multiplo=2.0):
    """Compara la tasa de stockout simulado en estados de estres (inventario
    bajo + demanda alta/critica) contra la tasa general. Rechaza si el
    subconjunto de estres falla mas de `umbral_multiplo` veces la tasa
    general."""
    estados_estres = [s for s in env.all_states if _es_estado_de_estres(s)]
    tasa_general = _tasa_stockout_simulado(env, Q, env.all_states)
    tasa_estres = _tasa_stockout_simulado(env, Q, estados_estres)

    limite = tasa_general * umbral_multiplo
    aprobado = tasa_estres <= max(limite, 1e-9) or (tasa_general == 0 and tasa_estres == 0)

    return {
        'capa': 'estres',
        'aprobado': aprobado,
        'n_estados_estres': len(estados_estres),
        'tasa_stockout_general_pct': round(tasa_general * 100, 1),
        'tasa_stockout_estres_pct': round(tasa_estres * 100, 1),
    }


# ---------------------------------------------------------------------------
# Capa 3: proyeccion de KPIs de negocio
# ---------------------------------------------------------------------------

def proyeccion_kpis(env, Q, n_episodios=200, dias_semana=7):
    """Traduce el desempeño simulado a las mismas unidades que reporta
    produccion: stockouts/semana, vencimientos/semana, inventario promedio.
    No incluye conversion a costo en dolares porque el codigo del examen no
    define un costo por unidad almacenada — esa calibracion queda como
    tarea explicita antes de fijar la condicion de aceptacion de costo."""
    stockouts = []
    vencimientos = []
    inventarios = []

    for _ in range(n_episodios):
        state = env.reset()
        for _ in range(dias_semana):
            inventory, days, demand = state
            demanda_val = env.demand_map[demand]
            accion_idx = int(np.argmax(Q[state]))
            accion_val = env.actions[accion_idx]

            stockouts.append(int(inventory + accion_val < demanda_val))
            next_state, _reward, _done, _ = env.step(accion_idx)
            vencimientos.append(int(next_state[1] <= 7 and next_state[0] > 0))
            inventarios.append(next_state[0])
            state = next_state

    return {
        'capa': 'kpis',
        'stockouts_por_semana_proyectado': round(np.mean(stockouts) * dias_semana, 2),
        'vencimientos_por_semana_proyectado': round(np.mean(vencimientos) * dias_semana, 2),
        'inventario_promedio': round(float(np.mean(inventarios)), 2),
        'nota': 'sin conversion a $/semana: falta costo por unidad almacenada en el codigo del examen',
    }


# ---------------------------------------------------------------------------
# Orquestador: aplica las 3 capas en orden, corta si Capa 1 rechaza
# ---------------------------------------------------------------------------

def protocolo_evaluacion(env, Q, sa_pairs_visited, nombre='politica'):
    print(f"\n=== Protocolo de evaluacion — {nombre} ===")

    capa1 = gate_cobertura(env, sa_pairs_visited)
    print(f"Capa 1 (cobertura): {'APROBADO' if capa1['aprobado'] else 'RECHAZADO'} "
          f"— pares {capa1['cobertura_pares_pct']}%, estados {capa1['cobertura_estados_pct']}%, "
          f"por demanda {capa1['cobertura_por_demanda_pct']}")

    if not capa1['aprobado']:
        print("  -> Despliegue bloqueado en Capa 1. No se evalua Capa 2 ni 3 "
              "(cobertura insuficiente invalida cualquier metrica posterior).")
        return {'nombre': nombre, 'aprobado_final': False, 'capas': [capa1]}

    capa2 = prueba_estres(env, Q)
    print(f"Capa 2 (estres): {'APROBADO' if capa2['aprobado'] else 'RECHAZADO'} "
          f"— stockout general {capa2['tasa_stockout_general_pct']}%, "
          f"stockout en estres {capa2['tasa_stockout_estres_pct']}% "
          f"({capa2['n_estados_estres']} estados de estres)")

    capa3 = proyeccion_kpis(env, Q)
    print(f"Capa 3 (KPIs): stockouts/semana proyectados = {capa3['stockouts_por_semana_proyectado']}, "
          f"vencimientos/semana = {capa3['vencimientos_por_semana_proyectado']}, "
          f"inventario promedio = {capa3['inventario_promedio']}")
    print(f"  Nota: {capa3['nota']}")

    aprobado_final = capa1['aprobado'] and capa2['aprobado']
    print(f"\n>>> Resultado final: {'APROBADO PARA DESPLIEGUE' if aprobado_final else 'RECHAZADO'}")

    return {'nombre': nombre, 'aprobado_final': aprobado_final, 'capas': [capa1, capa2, capa3]}


if __name__ == '__main__':
    env = PharmacyInventoryEnv()

    np.random.seed(42)
    Q_original, _vc, sap_original = train_epsilon_greedy(env)

    np.random.seed(42)
    Q_propuesta, _vc2, sap_propuesta = train_optimistic_ucb(env)

    resultado_original = protocolo_evaluacion(env, Q_original, sap_original, nombre='epsilon-greedy (original)')
    resultado_propuesta = protocolo_evaluacion(env, Q_propuesta, sap_propuesta, nombre='Optimista+UCB (Grupo 5)')

    print("\n\n=== RESUMEN ===")
    for r in (resultado_original, resultado_propuesta):
        print(f"{r['nombre']:30s} -> {'APROBADO' if r['aprobado_final'] else 'RECHAZADO'}")
