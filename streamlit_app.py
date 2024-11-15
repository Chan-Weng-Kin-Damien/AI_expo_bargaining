import streamlit as st
import pyspiel as sp
import torch
from DQN import DQN, Heuristics
from bargaining import new_game, play
import random


@st.cache(allow_output_mutation=True)
def load_dqn_model(file_path):
    game = sp.load_game("bargaining(max_turns=30,discount=0.98)")
    num_states = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    model = DQN(in_states=num_states, h1_nodes=512, h2_nodes=256, h3_nodes=128, out_actions=num_actions)
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load DQN models
policy_model_1 = load_dqn_model("bargaining_dqn.pt")
policy_model_2 = load_dqn_model("bargaining_dqn2.pt") 


# Initialize Streamlit app
st.set_page_config(layout="wide")
st.title("Bargaining Simulation")
st.write("Negotiate with AI agents using reinforcement learning or heuristics.")

# Agent selection
agent_choice = st.selectbox(
    "Select the agent to negotiate with:",
    ["Reinforcement Learning (DQN Model 1)", "Reinforcement Learning (DQN Model 2)", "Heuristic Agent"]
)

# Initialize session state for game variables
if "game" not in st.session_state:
    st.session_state["game"] = sp.load_game("bargaining(max_turns=30,discount=0.98)")
if "state" not in st.session_state:
    st.session_state["state"] = st.session_state["game"].new_initial_state()
    st.session_state["state"].apply_action(random.randint(0, 9))  # Initialize the chance node action
if "human_player" not in st.session_state:
    st.session_state["human_player"] = random.choice([0, 1])  # Randomly assign human as Player 0 or 1
if "agent_model" not in st.session_state:
    st.session_state["agent_model"] = None
if "game_over" not in st.session_state:
    st.session_state["game_over"] = False

# Reset game option
if st.button("Start New Game"):
    st.session_state["state"] = st.session_state["game"].new_initial_state()
    st.session_state["state"].apply_action(random.randint(0, 9))  # Reset chance node
    st.session_state["human_player"] = random.choice([0, 1])
    st.session_state["game_over"] = False
    st.experimental_rerun()

# Assign the selected agent model
if agent_choice == "Reinforcement Learning (DQN Model 1)":
    st.session_state["agent_model"] = policy_model_1
elif agent_choice == "Reinforcement Learning (DQN Model 2)":
    st.session_state["agent_model"] = policy_model_2
else:
    st.session_state["agent_model"] = Heuristics()

# Display game state
st.subheader("Game State")
st.text(st.session_state["state"])

# Show player assignment
st.write(f"You are Player {st.session_state['human_player']}")

# Define the human action function
def human_action(state):
    legal_actions = state.legal_actions()
    action_strings = {a: state.action_to_string(a) for a in legal_actions}
    action = st.selectbox("Choose your action:", legal_actions, format_func=lambda a: action_strings[a])
    if st.button("Submit Action"):
        return action
    st.stop()  # Wait for user interaction

# Define the agent action function
def agent_action(state):
    if isinstance(st.session_state["agent_model"], BargainingDQN):
        # Use the BargainingDQN model to compute the action
        action = st.session_state["agent_model"].state_to_action(
            state, st.session_state["game"], 1
        )  # Pass the game and agent index (1 for agent)
        return action
    elif isinstance(st.session_state["agent_model"], Heuristics):
        # Use the Heuristics model to compute the action
        action = st.session_state["agent_model"].valueAgent(state)
        return action
    else:
        raise ValueError("Invalid agent model type.")


# Define the player actions
player_actions = {
    st.session_state["human_player"]: human_action,
    1 - st.session_state["human_player"]: agent_action
}

# Run the game logic
if not st.session_state["state"].is_terminal() and not st.session_state["game_over"]:
    current_player = st.session_state["state"].current_player()
    if current_player == st.session_state["human_player"]:
        # Human turn
        action = human_action(st.session_state["state"])
        if action is not None:
            st.session_state["state"].apply_action(action)
            st.experimental_rerun()
    else:
        # Agent turn
        action = agent_action(st.session_state["state"])
        st.write(f"Agent chose: {st.session_state['state'].action_to_string(action)}")
        st.session_state["state"].apply_action(action)
        st.experimental_rerun()
else:
    # Game over
    st.subheader("Game Over")
    st.text(st.session_state["state"])
    rewards = st.session_state["state"].rewards()
    st.write(f"Player 0 reward: {rewards[0]}")
    st.write(f"Player 1 reward: {rewards[1]}")
    st.session_state["game_over"] = True
