import random
import matplotlib.pyplot as plt
from env import TicTacToe
from agent import QLearningAgent

def train_self_play(episodes=150000):
    env = TicTacToe()
    agent_x = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99996, min_epsilon=0.05)
    agent_o = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99996, min_epsilon=0.05)

    win_history = []
    x_wins = 0
    o_wins = 0
    draws = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        prev_state_x, prev_action_x = None, None
        prev_state_o, prev_action_o = None, None

        mode = random.random()

        while not done:
            available_actions = env.get_available_actions()
            current_turn = env.current_player

            if current_turn == 1:
                if mode < 0.15:
                    action = random.choice(available_actions)
                else:
                    action = agent_x.choose_action(state, available_actions)

                next_state, reward, done, winner = env.step(action)

                if prev_state_x is not None:
                    agent_x.update(prev_state_x, prev_action_x, 0.0, state, available_actions)

                prev_state_x = state
                prev_action_x = action

                if done:
                    if winner == 1:
                        agent_x.update(prev_state_x, prev_action_x, 1.0, next_state, [])
                        if prev_state_o is not None:
                            agent_o.update(prev_state_o, prev_action_o, -1.0, next_state, [])
                        x_wins += 1
                    elif winner == 0:
                        agent_x.update(prev_state_x, prev_action_x, 0.1, next_state, [])
                        if prev_state_o is not None:
                            agent_o.update(prev_state_o, prev_action_o, 0.1, next_state, [])
                        draws += 1

                state = next_state

            else:
                if mode >= 0.15 and mode < 0.30:
                    action = random.choice(available_actions)
                else:
                    action = agent_o.choose_action(state, available_actions)

                next_state, reward, done, winner = env.step(action)

                if prev_state_o is not None:
                    agent_o.update(prev_state_o, prev_action_o, 0.0, state, available_actions)

                prev_state_o = state
                prev_action_o = action

                if done:
                    if winner == -1:
                        agent_o.update(prev_state_o, prev_action_o, 1.0, next_state, [])
                        if prev_state_x is not None:
                            agent_x.update(prev_state_x, prev_action_x, -1.0, next_state, [])
                        o_wins += 1
                    elif winner == 0:
                        agent_o.update(prev_state_o, prev_action_o, 0.1, next_state, [])
                        if prev_state_x is not None:
                            agent_x.update(prev_state_x, prev_action_x, 0.1, next_state, [])
                        draws += 1

                state = next_state

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        if episode % 5000 == 0:
            total_games = x_wins + o_wins + draws
            win_history.append((x_wins / total_games, o_wins / total_games, draws / total_games))
            print(f"Episode {episode}/{episodes} | X Win: {x_wins/total_games:.2f} | O Win: {o_wins/total_games:.2f} | Draw: {draws/total_games:.2f} | Epsilon: {agent_x.epsilon:.3f}")
            x_wins, o_wins, draws = 0, 0, 0

    agent_x.save_policy("policy_x.pkl")
    agent_o.save_policy("policy_o.pkl")
    print("Training finished and policies saved.")

    return win_history

def plot_training_results(win_history):
    x_rates = [h[0] for h in win_history]
    o_rates = [h[1] for h in win_history]
    draw_rates = [h[2] for h in win_history]
    intervals = [i * 5000 for i in range(1, len(win_history) + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(intervals, x_rates, label="X Win Rate", color="blue")
    plt.plot(intervals, o_rates, label="O Win Rate", color="red")
    plt.plot(intervals, draw_rates, label="Draw Rate", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Rate")
    plt.title("Self-Play Training Dynamics")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    history = train_self_play(episodes=150000)
    plot_training_results(history)