from DQN import DQN
import torch
import pyspiel as sp
import random

game = sp.load_game("bargaining")
state = game.new_initial_state()
state.apply_action(random.randint(0, 9))

# Initialize the DQN model
model = DQN(in_states=309, h1_nodes=512, h2_nodes=256, h3_nodes=128, out_actions=10)
model.load_state_dict(torch.load("bargaining_dqn.pt", map_location=torch.device('cpu')))
model.eval()

# Pass the state tensor to the model
q_values = model(torch.FloatTensor(state.information_state_tensor()))
print("Q-values:", q_values)
