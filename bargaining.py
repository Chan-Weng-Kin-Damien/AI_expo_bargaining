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

class HeuristicAgent:
    def __init__(self):
        """
        Initialize the agent with valuation for objects, number of rounds and alpha.
        Alpha controls the balance between maximizing own utility and opponent's utility.
        """
        self.valuations = None
        self.pool = None
        self.state = None
        self.inferred_opponent_valuations = [2, 2, 2]  # Initial guess for opponent's valuations
        self.round = 0  # Start at the first round
        self.acceptance_threshold = 7  # Initial acceptance threshold (70%)
        self.alpha = 0.8  # Weight for agent's utility in combined utility calculation

    def set_state(self, state):
        self.state = state
        # Call get_pool_and_value after setting the state
        if self.pool is None or self.valuations is None:  # Only call if not already set
            self.pool, self.valuations = self.get_pool_and_value()

    def get_pool_and_value(self):
        # Extract relevant information from the state
        pool = None
        valuations = None
        info_lines = self.state.information_state_string().splitlines()
        # Search for lines containing "Pool" and "Valuations"
        for line in info_lines:
            if "Pool:" in line:
                pool = list(map(int, line.split(":")[1].strip().split(" ")))
            if "values:" in line:
                valuations = list(map(int, line.split(":")[1].strip().split(" ")))
        if pool is None or valuations is None:
            raise ValueError("Failed to extract pool or valuations from state information.")
        return pool, valuations

    def evaluate_offer(self, offer, valuations):
        """Calculate the utility of an offer based on given valuations."""
        return sum(valuations[i] * offer[i] for i in range(len(offer)))

    def adjust_acceptance_threshold(self):
        """Dynamically adjust the acceptance threshold based on how many rounds have passed."""
        if self.round > 7:
            self.acceptance_threshold = 3  # Late game, lower threshold to 30%
            self.alpha = 0.4  # Lower alpha for late game
        elif self.round > 4:
            self.acceptance_threshold = 5  # Middle game, lower threshold to 50%
            self.alpha = 0.6  # Lower alpha for middle game

    def infer_opponent_valuations(self, opponent_offer):
        """Update guessed opponent valuations based on their offers."""
        for i, quantity in enumerate(opponent_offer):
            if quantity > 0:  # If the opponent is asking for more of this object
                # Increase the inferred valuation for this object
                self.inferred_opponent_valuations[i] += quantity

    def calc_action(self):
        """Calculate the next action (offer) based on current state."""
        # Get all possible legal actions (offers or counter-offers) the agent can take
        legal_actions = self.state.legal_actions()
        best_action = None
        best_combined_utility = -float('inf')

        #first decide whether accept the offer or not
        if 120 in legal_actions:
            #get the list of offer
            decision = self.make_decision()
            if decision == "accept":
                return 120
            legal_actions.remove(120)

        # Evaluate each legal action and choose the one that maximizes combined utility
        for action in legal_actions:
            # Convert the action to a human-readable offer format
            offer = self.parse_offer(self.state.action_to_string(action))

            # Calculate the agent's utility for this offer
            agent_utility = self.evaluate_offer(offer, self.valuations)

            # Estimate opponent's utility for this offer based on inferred valuations
            opponent_utility = self.evaluate_offer(offer, self.inferred_opponent_valuations)

            # Combined utility: Maximize both agent's and opponent's utility
            combined_utility = self.alpha * agent_utility + (1 - self.alpha) * opponent_utility

            # If this combined value is better, choose this action
            if combined_utility > best_combined_utility:
                best_combined_utility = combined_utility
                best_action = action

        # Increase the round counter after making a decision
        self.round += 1
        self.adjust_acceptance_threshold()  # Adjust the acceptance threshold for the next round43
        return best_action

    def parse_offer(self, offer_string):
        #Convert the offer string into [offer]
        return list(map(int, offer_string.split(":")[1].strip().split(" ")))

    def make_decision(self):
        """Decide whether to accept or reject the opponent's offer."""
        #extract the offer from the state info
        opponent_offer = None
        info_lines = self.state.information_state_string().splitlines()
        for line in info_lines:
            if "Offer:" in line:
                opponent_offer = list(map(int, line.split(":")[2].strip().split(" ")))
        # In each round, infer the opponent's preferences from their offer
        self.infer_opponent_valuations(opponent_offer)

        # Evaluate the opponent's offer and decide whether to accept or reject
        agent_utility = self.evaluate_offer(opponent_offer, self.valuations)

        # Calculate the minimum utility required to accept the offer
        required_utility = self.acceptance_threshold

        # Accept the offer if it meets the acceptance threshold and provides good value to both
        if agent_utility >= required_utility:
            print(agent_utility)
            return "accept"
        else:
            return "reject"

g = new_game()
play(g,RandomAgent()) # Play game with 1 human player and 1 random agent (agent acts first)
#play(g, RandomAgent(), RandomAgent()) # play game with 2 random agents

