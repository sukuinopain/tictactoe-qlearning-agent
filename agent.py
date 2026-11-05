import random
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9999, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if not available_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)

        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions):
        current_q = self.get_q_value(state, action)

        if next_available_actions:
            max_future_q = max([self.get_q_value(next_state, a) for a in next_available_actions])
        else:
            max_future_q = 0.0

        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q

        self.q_table[(state, action)] = current_q + self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_policy(self, filename="policy.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_policy(self, filename="policy.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)