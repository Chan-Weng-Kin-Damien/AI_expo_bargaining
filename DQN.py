import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
import pyspiel as sp
import copy

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, h3_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)    # Second hidden layer
        self.fc3 = nn.Linear(h2_nodes, h3_nodes)    # Third hidden layer
        self.out = nn.Linear(h3_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)         # Calculate output

        return x
    

class Heuristics():
    def __init__(self):
        pass

    def valueAgent(self, state):
        obs = state.information_state_string().splitlines()
        #print(f"Value agent obs = \n{obs}")
        rounds_passed = len(obs) -3
        pool = []
        pool.append(int(obs[0][6]))
        pool.append(int(obs[0][8]))
        pool.append(int(obs[0][10]))
        valuations = list(map(int, obs[1][11:].split()))
        last_offer = []
        if len(obs)>3:
            last_offer.append(int(obs[-1][18]))
            last_offer.append(int(obs[-1][20]))
            last_offer.append(int(obs[-1][22]))
            offered_items = [a-b for a,b in zip(pool, last_offer)]
            valuation = sum([a*b for a,b in zip(valuations, offered_items)])
        
            if valuation >=6:
                return 120
            if rounds_passed> 7 and valuation >=5:
                return 120
        valid_offers = []
        for action in state.legal_actions():
            action_string =  state.action_to_string(action)
            if action_string == "Agree":
                break
            offer = []
            offer.append(int(action_string[7]))
            offer.append(int(action_string[9]))
            offer.append(int(action_string[11]))
            new_val = [a*b for a,b in zip(offer, valuations)]
            if sum(new_val)>=6:
                valid_offers.append(action)

        return valid_offers[random.randint(0,len(valid_offers)-1)]
        
                


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    
class BargainingDQN():
    #HYPERPARAMETERS
    heuristic_train_rate = 1000
    learning_rate = 0.0001
    discount_factor = 0.99
    model_sync_rate = 1000     # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 100000 # size of replay memory
    mini_batch_size = 128     # size of the training data set sampled from the replay memory
    tempGame = sp.load_game("bargaining(max_turns=30,discount=0.98)")
    information_tensor_shape = tempGame.information_state_tensor_shape()
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.
    

    def train(self, episodes):
        h = Heuristics()
        game = sp.load_game("bargaining(max_turns=30,discount=0.98)")
        state = game.new_initial_state()
        state.apply_action(random.randint(0,9))
        num_states = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()
        
        epsilon = 1 # 1 = 100% random actions
        epsilon_decay = 0.999986
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=512, h2_nodes = 256 ,h3_nodes= 128, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=512, h2_nodes = 256 ,h3_nodes= 128, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy netw:ork optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
        for i in range(episodes): #episodes
            if i%5000 == 0:
                print(f"EPISODE {i}")
            if i % self.heuristic_train_rate == 0:
                #print(f"Training against heurisitc on episode {i}")
                state = game.new_initial_state()
                state.apply_action(random.randint(0,9))  # Initialize to first non chance node state
                terminated = False      # True when agent agrees on a deal or 10 turns have passed
                turn = 0
                
                agent_states = []
                agent_actions = []
                agent_rewards = []
                agent_num = random.randint(0,1)
                while(not terminated):
                    currentPlayer = state.current_player()
                    if currentPlayer != agent_num:
                        action = h.valueAgent(state)
                        state.apply_action(action)
                        terminated = state.is_terminal()
                        if terminated:
                            agent_rewards.append(state.rewards()[agent_num])
                            agent_states.append(copy.deepcopy(state))
                
                    else:
                    # Select action based on epsilon-greedy
                        if random.random() < epsilon:
                            # select random action
                            action = state.legal_actions()[random.randint(0,len(state.legal_actions())-1)]
                        else:
                            # select best action            
                            with torch.no_grad():

                                q_values = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
                                action_mask = torch.zeros_like(q_values, dtype=torch.bool)
                                action_mask[state.legal_actions()] = True  # Mask for valid actions only
                                masked_q_values = q_values.masked_fill(~action_mask, -float('inf'))
                                action = masked_q_values.argmax().item()
                                
                                '''
                                if i % 100 == 0:
                                    print(f"episode (against heuristic) {i}")
                                    print(f"state: {state}")
                                    print(f"actions_mask :{action_mask} ")
                                    print(f"masked_q_values : {masked_q_values}")
                                    print(f"action : {action} \n")
                                '''
                        #obs = state.information_state_string().splitlines()
                        '''
                        if obs[-1][18:24] == "0 0 0":
                            action = 120
                            print(f"offered 0 0 0, forcing accept on episode {i}")
                        '''
                        agent_actions.append(action)
                        agent_states.append(copy.deepcopy(state))
                        
                        # Execute action
                        state.apply_action(action)
                        terminated = state.is_terminal()
                        agent_rewards.append(state.rewards()[agent_num])
                        if terminated:
                            agent_states.append(copy.deepcopy(state))

                        # Increment step counter
                    turn += 1
                    step_count+=1
                #game termianted
                for t in range(len(agent_states)-1):
                    term  = False
                    if agent_states[t+1].is_terminal() == True:
                        term = True
        
                    memory.append((agent_states[t], agent_actions[t],agent_states[t+1], agent_rewards[t], term))
                rewards_per_episode[i] = (agent_rewards[-1])
                #print(f"rewards_per_episode[{i}]: {rewards_per_episode[i]}")

            else:
                #print(f"Training normally on episode {i}")
            
                state = game.new_initial_state()
                state.apply_action(random.randint(0,9))  # Initialize to first non chance node state
                terminated = False      # True when agent agrees on a deal or 10 turns have passed
                turn = 0
                
                player0_states = []
                player1_states= []
                player0_actions = []
                player1_actions = []
                player0_rewards = []
                player1_rewards = []
                while(not terminated):
                    currentPlayer = state.current_player()
                    # Select action based on epsilon-greedy
                    if random.random() < epsilon:
                        # select random action
                        action = state.legal_actions()[random.randint(0,len(state.legal_actions())-1)]
                    else:
                        # select best action            
                        with torch.no_grad():

                            q_values = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
                            action_mask = torch.zeros_like(q_values, dtype=torch.bool)
                            action_mask[state.legal_actions()] = True  # Mask for valid actions only
                            masked_q_values = q_values.masked_fill(~action_mask, -float('inf'))
                            action = masked_q_values.argmax().item()
                            
                            '''
                            if i % 100 == 0:
                                print(f"episode {i}")
                                print(f"state: {state}")
                                print(f"actions_mask :{action_mask} ")
                                print(f"masked_q_values : {masked_q_values}")
                                print(f"action : {action} \n")
                            '''
                    #obs = state.information_state_string().splitlines()
                    '''
                    if obs[-1][18:24] == "0 0 0":
                        action = 120
                        print(f"offered 0 0 0, forcing accept on episode {i}")
                    '''
                    
                    if currentPlayer == 0:
                        player0_states.append(copy.deepcopy(state))
                        player0_actions.append(action)
                    
                    else:
                        player1_states.append(copy.deepcopy(state))
                        player1_actions.append(action)

                    # Execute action
                    state.apply_action(action)
                    terminated = state.is_terminal()
                        
                    if currentPlayer == 0:
                        player0_rewards.append(state.rewards()[0])
                    else:
                        player1_rewards.append(state.rewards()[1])
                    if terminated:
                        player0_states.append(copy.deepcopy(state))
                        player1_states.append(copy.deepcopy(state))
                        if currentPlayer == 0:
                            player1_rewards[-1] = state.rewards()[1]
                        else:
                            player0_rewards[-1] = state.rewards()[0]

                    # Increment step counter
                    turn += 1
                    step_count+=1
                # Game ended
                '''
                print(f"FINAL STATE: {state}")
                print(f"Player 0 states: {player0_states} \n player 0 actions {player0_actions} \n player 0 rewards {player0_rewards}")
                print(f"Player 1 states: {player1_states} \n player 1 actions {player1_actions} \n player 1 rewards {player1_rewards}")
                '''

                for t in range(len(player0_states)-1):
                    term  = False
                    obs = player0_states[t].information_state_string().splitlines()
                    #print(f"obs: {obs}")
                    if player0_states[t+1].is_terminal() == True:
                        term = True
        
                    memory.append((player0_states[t], player0_actions[t],player0_states[t+1], player0_rewards[t], term))
                
                for k in range(len(player1_states)-1):
                    term  = False
                    #obs = player1_states[k].information_state_string().splitlines()
                    if player1_states[k+1].is_terminal() == True:
                        term = True
                    #print(f"mem appended: {(player1_states[k], player1_actions[k],player1_states[k+1], player1_rewards[k], term)}")
                    memory.append((player1_states[k], player1_actions[k],player1_states[k+1], player1_rewards[k], term))

                rewards_per_episode[i] = (player0_rewards[-1] + player1_rewards[-1])/2
                #print(f"rewards_per_episode[{i}]: {rewards_per_episode[i]}")

        # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>=self.mini_batch_size: #and np.sum(rewards_per_episode)>0
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon_decay ** (i), 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.model_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Save policy
        torch.save(policy_dqn.state_dict(), "bargaining_dqn2.pt")

        # Create new graph 
        plt.figure(1)
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])/100
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('bargaining_dqn2.png')    

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes

        current_q_list = []
        target_q_list = []
        #print("STARTING OPTIMISATION")
        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor * target_dqn(torch.FloatTensor(new_state.information_state_tensor()).view(self.information_tensor_shape)).max()
                    )
               

            # Get the current set of Q values
            current_q = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
            current_q_list.append(current_q)
            # Get the target set of Q values
            target_q = target_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
            '''
            print(f"State: {state}, action: {action}")
            print(f"reward: {reward} \n terminated : {terminated} ")
            print(f"Current q list:{current_q}")
            print(f"Target q list: {target_q} ")
            '''     
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

            

    def test(self, episodes):
        # Create FrozenLake instance
        game = sp.load_game("bargaining(max_turns=30,discount=0.98)")
        state = game.new_initial_state()
        state.apply_action(random.randint(0,9)) 
        num_states = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()
        
        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=512, h2_nodes = 256 ,h3_nodes= 128, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("bargaining_dqn2.pt"))
        policy_dqn.eval()    # switch model to evaluation mode


        for i in range(episodes):
            state = game.new_initial_state()
            state.apply_action(random.randint(0,9))  # Initialize to first non chance node state
            terminated = False
            agent_num = 0#random.randint(0,1)
            if agent_num == 1:
                self.player_input(state)
            while(not terminated):   
                
                with torch.no_grad():
                    q_values = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
                    action_mask = torch.zeros_like(q_values, dtype=torch.bool)
                    action_mask[state.legal_actions()] = True  # Mask for valid actions only
                    masked_q_values = q_values.masked_fill(~action_mask, -float('inf'))
                    for action in state.legal_actions():
                        print(f"{state.action_to_string(action)}, num: {action}")
                    print("q values")
                    print(masked_q_values)
                    action = masked_q_values.argmax().item()
                    print(f"action: {action}")


                # Execute action
                state.apply_action(action)
                terminated = state.is_terminal()

                if not terminated:
                    self.player_input(state)
                terminated = state.is_terminal()
            print(f"Final state:\n{state}")
            print(f"Player0 reward: {state.rewards()[0]} \nPlayer1 reward: {state.rewards()[1]}")

    def player_input(self, state):
        print(f"You are player {state.current_player()}")
        print(state)
        print("Possible action strings and corresponding action numbers")
        for action in state.legal_actions():
            print(f"{state.action_to_string(action)}, num: {action}")
        x = int(input(state.legal_actions()))
        print(f"player {state.current_player()} picked action: {state.action_to_string(x)}")
        state.apply_action(x)
        print("____________________________________________________ \n\n\n")
    
    def test_self_play(self, episodes):
        game = sp.load_game("bargaining(max_turns=30,discount=0.98)")
        state = game.new_initial_state()
        state.apply_action(random.randint(0,9)) 
        num_states = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()
        
        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=512, h2_nodes = 256 ,h3_nodes= 128, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("bargaining_dqn2.pt"))
        policy_dqn.eval()    # switch model to evaluation mode


        for i in range(episodes):
            state = game.new_initial_state()
            state.apply_action(random.randint(0,9))  # Initialize to first non chance node state
            terminated = False
            while(not terminated):   
                print("STATE:", state)
                with torch.no_grad():
                    q_values = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
                    action_mask = torch.zeros_like(q_values, dtype=torch.bool)
                    action_mask[state.legal_actions()] = True  # Mask for valid actions only
                    masked_q_values = q_values.masked_fill(~action_mask, -float('inf'))
                    for action in state.legal_actions():
                        print(f"{state.action_to_string(action)}, num: {action}")
                    print("q values")
                    print(masked_q_values)
                    action = masked_q_values.argmax().item()
                    print(f"action: {action}")


                # Execute action
                state.apply_action(action)
                terminated = state.is_terminal()

            print(f"Final state:\n{state}")
            print(f"Player0 reward: {state.rewards()[0]} \nPlayer1 reward: {state.rewards()[1]}")



    def testing(self):
        game = sp.load_game("bargaining(max_turns=30,discount=0.98)")
        state = game.new_initial_state()
        state.apply_action(0)
        num_states = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()
        state.apply_action(0)
        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=512, h2_nodes = 256 ,h3_nodes= 128, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("bargaining_dqn.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        with torch.no_grad():
            q_values = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
            action_mask = torch.zeros_like(q_values, dtype=torch.bool)
            action_mask[state.legal_actions()] = True  # Mask for valid actions only
            masked_q_values = q_values.masked_fill(~action_mask, -float('inf'))
        
        print(f"maseskd q values after updazting : \n{masked_q_values}")

        #print(state.legal_actions())
        #state.apply_action(0) # Offer 0 0 0
    def state_to_action(self, state, game, model):
        num_states = game.information_state_tensor_size()
        num_actions = game.num_distinct_actions()
        policy_dqn = DQN(in_states=num_states, h1_nodes=512, h2_nodes = 256 ,h3_nodes= 128, out_actions=num_actions)
        if model == 1:
            policy_dqn.load_state_dict(torch.load("bargaining_dqn.pt"))
        else:
            policy_dqn.load_state_dict(torch.load("bargaining_dqn2.pt"))
        
        policy_dqn.eval()    # switch model to evaluation mode

        with torch.no_grad():
            q_values = policy_dqn(torch.FloatTensor(state.information_state_tensor()).view(self.information_tensor_shape))
            action_mask = torch.zeros_like(q_values, dtype=torch.bool)
            action_mask[state.legal_actions()] = True  # Mask for valid actions only
            masked_q_values = q_values.masked_fill(~action_mask, -float('inf'))
            action = masked_q_values.argmax().item()
        return action

if __name__ == '__main__':

    bargaining = BargainingDQN()
    #bargaining.train(250000)
    #bargaining.test(1)
    #bargaining.testing()
    bargaining.test_self_play(1)