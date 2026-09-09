import numpy as np
from collections import defaultdict
import math

np.random.seed(42)

class PharmacyInventoryEnv:

    def __init__(self):

        self.inventory_levels = list(range(0, 100, 10))
        self.days_levels = [1, 7, 14, 30, 60]
        self.demand_levels = ['bajo', 'medio', 'alto', 'crítico']
        self.demand_map = {'bajo': 5, 'medio': 15, 'alto': 25, 'crítico': 40}


        self.actions = [0, 10, 20, 30, 40, 50]
        self.n_actions = len(self.actions)


        self.all_states = [
            (inv, days, dem)
            for inv in self.inventory_levels
            for days in self.days_levels
            for dem in self.demand_levels
        ]
        self.n_states = len(self.all_states)

        self.state = None
        self.steps = 0
        self.max_steps = 60

    def reset(self):
        inv = np.random.choice(self.inventory_levels)
        days = np.random.choice(self.days_levels)
        demand = np.random.choice(self.demand_levels)
        self.state = (inv, days, demand)
        self.steps = 0
        return self.state

    def _discretize_inventory(self, value):
        return min(self.inventory_levels, key=lambda x: abs(x - value))

    def _discretize_days(self, value):
        return min(self.days_levels, key=lambda x: abs(x - value))

    def step(self, action_idx):
        action = self.actions[action_idx]
        inv, days, dem = self.state
        daily_demand = self.demand_map[dem]

        new_inv = self._discretize_inventory(
            min(100, max(0, inv + action - daily_demand))
        )
        new_days = self._discretize_days(max(1, days - 1))
        new_dem = dem

        next_state = (new_inv, new_days, new_dem)

        reward = new_inv * 0.5
        reward += -10 if new_days <= 7 else 0
        reward += -2 if action > 0 else 0

        self.state = next_state
        self.steps += 1
        done = self.steps >= self.max_steps

        return next_state, reward, done, {}


def train_epsilon_greedy(env, episodes=1000, epsilon=0.05, alpha=0.9, gamma=0.99):
    Q = defaultdict(lambda: np.zeros(env.n_actions))


    visit_counts = defaultdict(lambda: np.zeros(env.n_actions, dtype=int))
    sa_pairs_visited = set()

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:

            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])


            visit_counts[state][action] += 1
            sa_pairs_visited.add((state, action))


            next_state, reward, done, _ = env.step(action)


            td_target = reward + gamma * np.max(Q[next_state])
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

    return Q, visit_counts, sa_pairs_visited

def train_optimistic_ucb(env, episodes=1000, c=2.0, alpha=0.1, gamma=0.99,
                         optimistic_value=100.0):

    Q = defaultdict(lambda: np.full(env.n_actions, optimistic_value))

    N_state = defaultdict(int)
    N_state_action = defaultdict(lambda: np.zeros(env.n_actions, dtype=int))

    visit_counts = defaultdict(lambda: np.zeros(env.n_actions, dtype=int))
    sa_pairs_visited = set()

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            N_state[state] += 1


            q_values = Q[state].copy()

            for a in range(env.n_actions):
                if N_state_action[state][a] == 0:



                    q_values[a] = optimistic_value + 1000
                else:

                    ucb_bonus = c * math.sqrt(
                        math.log(N_state[state]) / N_state_action[state][a]
                    )
                    q_values[a] = Q[state][a] + ucb_bonus


            action = np.argmax(q_values)


            N_state_action[state][action] += 1
            visit_counts[state][action] += 1
            sa_pairs_visited.add((state, action))


            next_state, reward, done, _ = env.step(action)


            td_target = reward + gamma * np.max(Q[next_state])
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state

    return Q, visit_counts, sa_pairs_visited


if __name__ == "__main__":
    env = PharmacyInventoryEnv()
    total_sa = env.n_states * env.n_actions

    print(f"Espacio: {env.n_states} estados x {env.n_actions} acciones = {total_sa} pares (s,a)")
    print()


    print("Entrenando epsilon-greedy (epsilon=0.05)")
    np.random.seed(42)
    Q_orig, vc_orig, sap_orig = train_epsilon_greedy(env)


    print("Entrenando Optimista + UCB")
    np.random.seed(42)
    Q_prop, vc_prop, sap_prop = train_optimistic_ucb(env)


    print()
    print("=" * 65)
    print(f"{'METRICA':<40} {'eps-greedy':>10} {'Opt+UCB':>10}")
    print("=" * 65)
    print(f"{'Pares (s,a) visitados':<40} {len(sap_orig):>10} {len(sap_prop):>10}")
    print(f"{'% cobertura':<40} {len(sap_orig)/total_sa*100:>9.1f}% {len(sap_prop)/total_sa*100:>9.1f}%")
    print(f"{'Pares NO visitados':<40} {total_sa - len(sap_orig):>10} {total_sa - len(sap_prop):>10}")


    print()
    print("Pares NO visitados por nivel de demanda:")
    for dem in env.demand_levels:
        not_orig = sum(1 for s in env.all_states if s[2] == dem
                       for a in range(6) if (s, a) not in sap_orig)
        not_prop = sum(1 for s in env.all_states if s[2] == dem
                       for a in range(6) if (s, a) not in sap_prop)
        print(f"  {dem:<10} eps-greedy: {not_orig:>4}    Opt+UCB: {not_prop:>4}")


    print()
    print("Pares NO visitados por accion:")
    for ai, a_val in enumerate(env.actions):
        not_orig = sum(1 for s in env.all_states if (s, ai) not in sap_orig)
        not_prop = sum(1 for s in env.all_states if (s, ai) not in sap_prop)
        print(f"  pedir {a_val:>2}   eps-greedy: {not_orig:>4}    Opt+UCB: {not_prop:>4}")


    print()
    print("=" * 65)
    print("DISTRIBUCION DE VISITAS POR PAR (s,a)")
    print("=" * 65)


    for nombre, vc, sap in [("eps-greedy", vc_orig, sap_orig),
                             ("Optimista+UCB", vc_prop, sap_prop)]:
        counts = []
        for s in env.all_states:
            for a in range(env.n_actions):
                counts.append(vc[s][a])
        counts = np.array(counts)
        visited = counts[counts > 0]

        print(f"\n  {nombre}:")
        print(f"    Pares con 0 visitas:      {np.sum(counts == 0):>6}")
        print(f"    Pares con 1-5 visitas:    {np.sum((counts >= 1) & (counts <= 5)):>6}")
        print(f"    Pares con 6-20 visitas:   {np.sum((counts >= 6) & (counts <= 20)):>6}")
        print(f"    Pares con 21-100 visitas: {np.sum((counts >= 21) & (counts <= 100)):>6}")
        print(f"    Pares con >100 visitas:   {np.sum(counts > 100):>6}")
        print(f"    ---")
        print(f"    Promedio (pares visitados):  {visited.mean():>8.1f}")
        print(f"    Desviacion estandar:         {visited.std():>8.1f}")
        print(f"    Minimo:                      {visited.min():>8}")
        print(f"    Maximo:                      {visited.max():>8}")





    print()
    print("=" * 65)
    print("DISTRIBUCION DE VISITAS POR ESTADO")
    print("=" * 65)

    for nombre, vc in [("eps-greedy", vc_orig), ("Optimista+UCB", vc_prop)]:
        state_visits = []
        for s in env.all_states:
            total = int(vc[s].sum())
            state_visits.append(total)
        state_visits = np.array(state_visits)
        visited = state_visits[state_visits > 0]

        print(f"\n  {nombre}:")
        print(f"    Estados con 0 visitas:      {np.sum(state_visits == 0):>6}")
        print(f"    Estados con 1-50 visitas:   {np.sum((state_visits >= 1) & (state_visits <= 50)):>6}")
        print(f"    Estados con 51-200 visitas: {np.sum((state_visits >= 51) & (state_visits <= 200)):>6}")
        print(f"    Estados con 201-500:        {np.sum((state_visits >= 201) & (state_visits <= 500)):>6}")
        print(f"    Estados con >500 visitas:   {np.sum(state_visits > 500):>6}")
        print(f"    ---")
        print(f"    Promedio:  {visited.mean():>8.1f}")
        print(f"    Desv. std: {visited.std():>8.1f}")


    print("\nEstados donde la politica causa stockout directo:")
    print(f"{'Estado':^35} {'Accion':>10} {'Demanda':>10} {'Acciones probadas':>20}")
    print("-" * 77)

    stockout_states_orig = 0
    poor_exploration_stockout = 0

    for s in env.all_states:
        inv, days, dem = s
        demand_val = env.demand_map[dem]
        best_a_idx = np.argmax(Q_orig[s])
        best_a_val = env.actions[best_a_idx]
        n_tried = int(sum(1 for a in range(6) if vc_orig[s][a] > 0))


        if inv + best_a_val < demand_val:
            stockout_states_orig += 1
            if n_tried <= 3:
                poor_exploration_stockout += 1
                if stockout_states_orig <= 8:
                    print(f"  ({inv:>2}, {days:>2}, {dem:<8})"
                          f"  pedir {best_a_val:>2}"
                          f"  necesita {demand_val:>2}"
                          f"      {n_tried}/6 acciones")

    print(f"\n  Total estados con stockout directo: {stockout_states_orig}")
    print(f"  De estos, con pobre exploracion (<=3 acciones probadas): {poor_exploration_stockout}")
