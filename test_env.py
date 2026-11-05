from env import TicTacToe

game = TicTacToe()
game.reset()

print("Initial Board:")
game.render()

game.step(4)
game.step(0)
game.step(1)

print("Board after moves:")
game.render()
print("Available actions:", game.get_available_actions())