import streamlit as st
import pyspiel as sp
import torch
import random  # Ensure this is imported
from DQN import DQN, Heuristics, BargainingDQN

# Load pre-trained DQN model
@st.cache(allow_output_mutation=True)
def load_dqn_model(file_path):
    game = sp.load_game("bargaining")  # Ensure consistent game config
    num_states = game.information_state_tensor_size()  # Should be 309
    num_actions = game.num_distinct_actions()  # Should be 10 in your current setup
    
    # Initialize the DQN model with the current game setup
    model = DQN(in_states=num_states, h1_nodes=512, h2_nodes=256, h3_nodes=128, out_actions=num_actions)
    
    # Adjust output layer to match the saved model's output size (e.g., 121)
    model.adjust_output_layer(new_out_actions=121)
    
    # Load the weights from the saved model
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')), strict=False)
    
    # Re-adjust the output layer to match the current game's action space (e.g., 10)
    model.adjust_output_layer(new_out_actions=num_actions)
    
    # Set the model to evaluation mode
    model.eval()
    return model

# Initialize the app
st.set_page_config(layout="wide")
st.title("Bargaining Game")
st.write("Negotiate with AI agents (DQN or Heuristic) in the bargaining environment.")

# Load the game
game = sp.load_game("bargaining")
print("Information state size:", game.information_state_tensor_size())

# Initialize session state variables
if "state" not in st.session_state:
    st.session_state["state"] = game.new_initial_state()
    st.session_state["state"].apply_action(random.randint(0, 9))  # Initialize chance node
if "human_player" not in st.session_state:
    st.session_state["human_player"] = random.choice([0, 1])  # Randomly assign human player
if "agent_type" not in st.session_state:
    st.session_state["agent_type"] = None
if "game_over" not in st.session_state:
    st.session_state["game_over"] = False

# User selects the agent type
agent_choice = st.selectbox("Choose the agent to negotiate with:", ["DQN Model", "Heuristic Agent"])
if st.button("Start Game"):
    st.session_state["agent_type"] = agent_choice
    st.session_state["state"] = game.new_initial_state()
    st.session_state["state"].apply_action(random.randint(0, 9))  # Reset chance node
    st.session_state["game_over"] = False
    st.experimental_rerun()

# Load the DQN model if selected
if st.session_state["agent_type"] == "DQN Model":
    policy_dqn = load_dqn_model("bargaining_dqn.pt")
    bargaining_dqn = BargainingDQN()

# Display the current state
st.subheader("Game State")
st.text(st.session_state["state"])

# Show the player's role
st.write(f"You are Player {st.session_state['human_player']}")

# Define human action
def human_action(state):
    legal_actions = state.legal_actions()
    action_strings = {a: state.action_to_string(a) for a in legal_actions}
    action = st.selectbox("Choose your action:", legal_actions, format_func=lambda a: action_strings[a])
    if st.button("Submit Action"):
        return action
    st.stop()  # Wait for user interaction

# Define agent action
def agent_action(state):
    if st.session_state["agent_type"] == "DQN Model":
        action = bargaining_dqn.state_to_action(state, game, model=1)
    elif st.session_state["agent_type"] == "Heuristic Agent":
        heuristics_agent = Heuristics()
        action = heuristics_agent.valueAgent(state)
    else:
        action = None  # This should never happen
    print(f"Agent selected action: {action}")
    return action


# Play the game
if not st.session_state["state"].is_terminal() and not st.session_state["game_over"]:
    current_player = st.session_state["state"].current_player()
    if current_player == st.session_state["human_player"]:
        # Human's turn
        st.subheader("Your Turn")
        action = human_action(st.session_state["state"])
        if action is not None:
            st.session_state["state"].apply_action(action)
            st.experimental_rerun()
    else:
        # Agent's turn
        st.subheader("Agent's Turn")
        action = agent_action(st.session_state["state"])
        if action is not None:
            st.write(f"Agent chose: {st.session_state['state'].action_to_string(action)}")
            st.session_state["state"].apply_action(action)
            st.experimental_rerun()
        else:
            st.error("Agent failed to select a valid action.")
else:
    # Game over
    st.subheader("Game Over")
    st.text(st.session_state["state"])
    rewards = st.session_state["state"].rewards()
    st.write(f"Player 0 reward: {rewards[0]}")
    st.write(f"Player 1 reward: {rewards[1]}")
    st.session_state["game_over"] = True
