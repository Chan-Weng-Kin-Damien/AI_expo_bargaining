import pyspiel as sp
import random

def new_game():
    game = sp.load_game("bargaining")
    return game

def play(game, *args): # optional args are agent objects (up to 2)
    turn = 1
    state = game.new_initial_state()
    state.apply_action(random.randint(0,9)) # Initial state is chance node which generates the random valuations and pool
    players = 2-len(args) # Number of human players
    while not state.is_terminal():
        print(f"Turn: {turn}")
        print(state)
        print(f"Current player:{state.current_player()}")
        print("Possible action strings and corresponding action numbers")
        for action in state.legal_actions():
            print(f"{state.action_to_string(action)}, num: {action}")
        if players == 2:
            x = int(input(state.legal_actions())) # if 2 human players, then input
        elif players ==1:
            if state.current_player() == 0: # player 0 defaults to the bot agent
                args[0].set_state(state) # Update state of agent
                x = args[0].calc_action() # Call function of agent to calculate next action
            else:
                x = int(input(f"{state.legal_actions()} \nEnter action (number) ")) # If human player's (player 1) turn then input
        else:
            if state.current_player() == 0:
                args[0].set_state(state)
                x = args[0].calc_action()
            else:
                args[1].set_state(state)
                x = args[1].calc_action()
        print(f"player {state.current_player()} picked action: {state.action_to_string(x)}")
        state.apply_action(x)
        turn+=1
        print("____________________________________________________ \n\n\n")

    print(f"Final state:\n{state}")
    print(f"Player0 reward: {state.rewards()[0]} \nPlayer1 reward: {state.rewards()[1]}")

class RandomAgent:
    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state
    
    def calc_action(self): # calculate next action to take given current state
        return self.state.legal_actions()[random.randint(0, len(self.state.legal_actions())-1)] # Pick random legal action


g = new_game()
play(g,RandomAgent()) # Play game with 1 human player and 1 random agent (agent acts first)
#play(g, RandomAgent(), RandomAgent()) # play game with 2 random agents

