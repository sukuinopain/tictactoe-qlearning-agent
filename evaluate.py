import os
import random
from env import TicTacToe
from agent import QLearningAgent

def evaluate_vs_random(games=1000):
    if not os.path.exists("policy_x.pkl") or not os.path.exists("policy_o.pkl"):
        print("Model policies not found! Please run train.py first.")
        return

    agent_x = QLearningAgent(epsilon=0.0)
    agent_x.load_policy("policy_x.pkl")

    agent_o = QLearningAgent(epsilon=0.0)
    agent_o.load_policy("policy_o.pkl")

    env = TicTacToe()

    print(f"\n--- Evaluating Agent (as X - First Player) vs Random Opponent ({games} games) ---")
    x_wins, random_o_wins, draws_1 = 0, 0, 0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            if env.current_player == 1:
                action = agent_x.choose_action(state, available_actions)
            else:
                action = random.choice(available_actions)

            state, reward, done, winner = env.step(action)

        if winner == 1:
            x_wins += 1
        elif winner == -1:
            random_o_wins += 1
        else:
            draws_1 += 1

    print(f"Agent (X) Win Rate:   {x_wins / games * 100:.2f}% ({x_wins}/{games})")
    print(f"Random Opponent Win:  {random_o_wins / games * 100:.2f}% ({random_o_wins}/{games})")
    print(f"Draw Rate:            {draws_1 / games * 100:.2f}% ({draws_1}/{games})")

    print(f"\n--- Evaluating Agent (as O - Second Player) vs Random Opponent ({games} games) ---")
    o_wins, random_x_wins, draws_2 = 0, 0, 0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            if env.current_player == 1:
                action = random.choice(available_actions)
            else:
                action = agent_o.choose_action(state, available_actions)

            state, reward, done, winner = env.step(action)

        if winner == -1:
            o_wins += 1
        elif winner == 1:
            random_x_wins += 1
        else:
            draws_2 += 1

    print(f"Agent (O) Win Rate:   {o_wins / games * 100:.2f}% ({o_wins}/{games})")
    print(f"Random Opponent Win:  {random_x_wins / games * 100:.2f}% ({random_x_wins}/{games})")
    print(f"Draw Rate:            {draws_2 / games * 100:.2f}% ({draws_2}/{games})")

if __name__ == "__main__":
    evaluate_vs_random(games=1000)