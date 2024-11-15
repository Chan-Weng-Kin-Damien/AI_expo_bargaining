import pyspiel as sp
import random

def new_game():
    """
    Initializes a new bargaining game instance using OpenSpiel.
    """
    game = sp.load_game("bargaining(max_turns=30,discount=0.98)")
    return game

def play(game, player_actions):
    """
    Play the game by passing a dictionary of player action functions.
    
    Args:
        game (pyspiel.Game): The bargaining game instance.
        player_actions (dict): A dictionary where keys are player indices (0 or 1)
                               and values are functions that take a state and return an action.
    """
    turn = 1
    state = game.new_initial_state()
    state.apply_action(random.randint(0, 9))  # Initialize with chance node action
    
    while not state.is_terminal():
        print(f"Turn: {turn}")
        print(state)
        print(f"Current player: {state.current_player()}")
        print("Possible actions:", [state.action_to_string(a) for a in state.legal_actions()])
        
        current_player = state.current_player()
        if current_player in player_actions:
            # Call the action function for the current player
            action = player_actions[current_player](state)
        else:
            raise ValueError(f"No action function provided for player {current_player}")
        
        print(f"Player {current_player} chose: {state.action_to_string(action)}")
        state.apply_action(action)
        turn += 1
        print("____________________________________________________ \n\n\n")
    
    print(f"Final state:\n{state}")
    print(f"Player0 reward: {state.rewards()[0]} \nPlayer1 reward: {state.rewards()[1]}")

class RandomAgent:
    """
    A random agent that selects actions randomly from the legal actions.
    """
    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state
    
    def calc_action(self):
        return self.state.legal_actions()[random.randint(0, len(self.state.legal_actions()) - 1)]


class HumanAgent:
    """
    A placeholder class for human action handling. To be replaced with Streamlit UI.
    """
    def __init__(self):
        pass

    def calc_action(self, state):
        raise NotImplementedError("Human action must be handled by Streamlit or other UI framework.")


# Example usage
if __name__ == "__main__":
    g = new_game()

    # Example: Define actions programmatically
    def human_action(state):
        legal_actions = state.legal_actions()
        return legal_actions[0]  # Replace with dynamic input (e.g., Streamlit UI)

    def random_agent_action(state):
        legal_actions = state.legal_actions()
        return random.choice(legal_actions)

    # Define action handlers for each player
    player_actions = {
        0: human_action,  # Human player
        1: random_agent_action  # Random agent
    }

    # Play the game
    play(g, player_actions)


