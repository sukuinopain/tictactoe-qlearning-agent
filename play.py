import os
from env import TicTacToe
from agent import QLearningAgent

def play_vs_agent():
    if not os.path.exists("policy_x.pkl") or not os.path.exists("policy_o.pkl"):
        print("Model policies not found! Please run train.py first.")
        return

    print("Choose your symbol:")
    print("1: Play as X (Go first)")
    print("2: Play as O (Go second)")
    user_choice = input("Enter 1 or 2: ").strip()

    if user_choice == "1":
        human_player = 1
        agent_player = -1
        agent = QLearningAgent(epsilon=0.0)
        agent.load_policy("policy_o.pkl")
    else:
        human_player = -1
        agent_player = 1
        agent = QLearningAgent(epsilon=0.0)
        agent.load_policy("policy_x.pkl")

    env = TicTacToe()
    state = env.reset()
    done = False

    print("\nGame start! Board positions are mapped from 0 to 8:")
    print(" 0 | 1 | 2 ")
    print(" 3 | 4 | 5 ")
    print(" 6 | 7 | 8 \n")

    while not done:
        env.render()
        available_actions = env.get_available_actions()

        if env.current_player == human_player:
            valid_move = False
            while not valid_move:
                try:
                    move = int(input(f"Your turn ({'X' if human_player == 1 else 'O'}). Enter cell (0-8): "))
                    if move in available_actions:
                        valid_move = True
                        state, reward, done, winner = env.step(move)
                    else:
                        print("Invalid move! Cell is either taken or out of range. Choose from:", available_actions)
                except ValueError:
                    print("Please enter a valid number between 0 and 8.")
        else:
            print("Agent is thinking...")
            action = agent.choose_action(state, available_actions)
            state, reward, done, winner = env.step(action)
            print(f"Agent chose cell: {action}")

    print("\nFinal Board:")
    env.render()

    if winner == human_player:
        print("Congratulations! You won!")
    elif winner == agent_player:
        print("Agent won! Better luck next time.")
    else:
        print("It's a draw! Perfect play from both sides.")

if __name__ == "__main__":
    play_vs_agent()