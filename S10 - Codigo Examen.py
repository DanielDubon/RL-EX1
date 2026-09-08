# ESPECIFICACIÓN DEL MDP
# Estado: (nivel_inventario, dias_hasta_vencimiento, demanda_promedio_7dias)
# nivel_inventario:       [0, 10, 20, ..., 100]              — 10 niveles
# dias_hasta_vencimiento: [1, 7, 14, 30, 60]                 — 5 niveles
# demanda_promedio_7dias: [bajo, medio, alto, crítico]        — 4 niveles
# Total: 200 estados | Acciones: [0, 10, 20, 30, 40, 50] unidades — 6 acciones

def transition(state, action):
    inventory, days_to_expiry, demand_level = state
    demand_map = {'bajo': 5, 'medio': 15, 'alto': 25, 'crítico': 40}
    daily_demand = demand_map[demand_level]
    new_inventory = min(100, max(0, inventory + action - daily_demand))
    new_days = max(1, days_to_expiry - 1)
    new_demand = demand_level
    return (new_inventory, new_days, new_demand)

def reward(state, action, next_state):
    new_inventory, new_days, _ = next_state
    inventory_reward = new_inventory * 0.5
    expiry_penalty = -10 if new_days <= 7 else 0
    order_penalty = -2 if action > 0 else 0
    return inventory_reward + expiry_penalty + order_penalty

def train(env, episodes=1000):
    Q = defaultdict(lambda: np.zeros(6))
    alpha = 0.9
    gamma = 0.99
    epsilon = 0.05
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(6)
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])
            state = next_state
    return Q

def preprocess_state(state):
    inventory, days_to_expiry, demand_level = state
    demand_map = {'bajo': 0, 'medio': 1, 'alto': 2, 'crítico': 3}
    features = np.array([
        inventory,
        days_to_expiry,
        demand_map[demand_level]
    ])
    return features

class LinearApproximator:
    def __init__(self, n_features=3, n_actions=6):
        self.weights = np.zeros((n_actions, n_features))
    def predict(self, features, action):
        return np.dot(self.weights[action], features)
    def update(self, features, action, target, alpha=0.01):
        prediction = self.predict(features, action)
        error = target - prediction
        self.weights[action] += alpha * error * features

def evaluate_policy(Q, env, episodes=100):
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'min_reward': np.min(total_rewards)
    }

# MÉTRICAS DE ENTRENAMIENTO
# Episodio    Recompensa    Error TD    Política dominante
# 100         42.3          8.42        pedir_50
# 500         48.1          7.12        pedir_50
# 1000        51.7          5.21        pedir_50
# Varianza entre episodios: 0.8
# Política greedy: pedir 50 en 94% de estados

# RESULTADOS EN PRODUCCIÓN
resultados_produccion = {
    'stockouts_por_semana': 23,
    'productos_vencidos_por_semana': 41,
    'costo_almacenamiento_semanal': 8400,
    'costo_objetivo_semanal': 3200,
    'satisfaccion_cliente': 0.61
}

# RESULTADOS EN SIMULACIÓN
resultados_simulacion = {
    'mean_reward': 51.2,
    'std_reward': 0.9,
    'min_reward': 48.3
}