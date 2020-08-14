import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import dpsgd
import argparse
# Use a double ended queue (deque) for memory
# When memory is full, this will replace the oldest value with the new one
from collections import deque

# Gym environment to use
ENV_NAME = "CartPole-v0"
# Set whether to display on screen (slows model)
DISPLAY_ON_SCREEN = False
# Discount rate of future rewards
GAMMA = 0.99
# Learing rate for neural network
LEARNING_RATE = 0.001
# Maximum number of game steps (state, action, reward, next state) to keep
MEMORY_SIZE = 5000
# Sample batch size for policy network update
BATCH_SIZE = 5
# Number of game steps to play before starting training (all random actions)
REPLAY_START_SIZE = 500
# Number of steps between policy -> target network update
SYNC_TARGET_STEPS = 100
# Exploration rate (episolon) is probability of choosign a random action
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.999
# Target rewards
TARGET_REWARD = 200
# Number of times target must be met without break
TARGET_PERIOD = 5

class DQN(nn.Module):

    """Deep Q Network. Udes for both policy (action) and target (Q) networks."""

    def __init__(self, observation_space, action_space):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = EXPLORATION_MAX
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_space)
            )
        
        # Set loss function and optimizer
        # self.objective = nn.MSELoss()
        # self.optimizer = optim.Adam(
        #     params=self.net.parameters(), lr=LEARNING_RATE)
        
    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.net(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
        
  
    def forward(self, x):
        return self.net(x)

def optimize(policy_net, target_net, memory):
    """
    Update  model by sampling from memory.
    Uses policy network to predict best action (best Q).
    Uses target network to provide target of Q for the selected next action.
    """
      
    # Do not try to train model if memory is less than reqired batch size
    if len(memory) < BATCH_SIZE:
        return    
 
    # Reduce exploration rate (exploration rate is stored in policy net)
    policy_net.exploration_rate *= EXPLORATION_DECAY
    policy_net.exploration_rate = max(EXPLORATION_MIN, 
                                      policy_net.exploration_rate)
    # Sample a random batch from memory
    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
        
        state_action_values = policy_net(torch.FloatTensor(state))
        
        # Get target Q for policy net update
       
        if not terminal:
            # For non-terminal actions get Q from policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next state action with best Q from the policy net (double DQN)
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get target net next state
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            updated_q = reward + (GAMMA * best_next_q)      
            expected_state_action_values[0][action] = updated_q
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[0] = reward
 
        # Reset net gradients
        policy_net.optimizer.zero_grad()  
        # calculate loss
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Backpropogate loss
        loss_v.backward()
        # Update network gradients
        policy_net.optimizer.step()  

    return



class Memory():
    """
    Replay memory used to train model.
    Limited length memory (using deque, double ended queue from collections).
      - When memory full deque replaces oldest data with newest.
    Holds, state, action, reward, next state, and episode done.
    """
    
    def __init__(self):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, done):
        """state/action/reward/next_state/done"""
        self.memory.append((state, action, reward, next_state, done))


def plot_results(run, exploration, score):
    """Plot results at end of run"""
    
    # Set up chart (ax1 and ax2 share x-axis to combine two plots on one graph)
    fig = plt.figure(figsize=(6,6,))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    # Plot results
    ax1.plot(run, exploration, label='exploration', color='g')
    ax2.plot(run, score, label='score', color='r')
    
    # Set axes
    ax1.set_xlabel('run')
    ax1.set_ylabel('exploration', color='g')
    ax2.set_ylabel('score', color='r')
    
    # Show
    plt.show()
 #train( policy_net, device, memory.memory, optimizer, epoch)
def train( policy_net, device, memory, optimizer, epoch,target_net):
    policy_net.train()
    total_loss = 0 
    counter = 0
    if len(memory) < BATCH_SIZE:
        return  
    batch = random.sample(memory, BATCH_SIZE)
   
 
    # Reduce exploration rate (exploration rate is stored in policy net)
    policy_net.exploration_rate *= EXPLORATION_DECAY
    policy_net.exploration_rate = max(EXPLORATION_MIN, 
                                      policy_net.exploration_rate)
    #print("train")
    for state, action, reward, state_next, terminal in batch:
        
        state_action_values = policy_net(torch.FloatTensor(state))
        
        # Get target Q for policy net update
       
        if not terminal:
            # For non-terminal actions get Q from policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next state action with best Q from the policy net (double DQN)
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get target net next state
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            updated_q = reward + (GAMMA * best_next_q)      
            expected_state_action_values[0][action] = updated_q
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[0] = reward
 ##################################
        # policy_net.optimizer.zero_grad()  
        # loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # loss_v.backward()
        # policy_net.optimizer.step() 
#########################
        optimizer.zero_grad()  
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        loss_v.backward() 
        optimizer.step() 


def cartpole():
    """Main program loop"""
    
    ############################################################################
    #                          8 Set up environment                            #
    ############################################################################
        
    # Set up game environemnt
    env = gym.make(ENV_NAME)
    
    # Get number of observations returned for state
    observation_space = env.observation_space.shape[0]
    
    # Get number of actions possible
    action_space = env.action_space.n
    
    ############################################################################
    #                    9 Set up policy and target nets                       #
    ############################################################################
    
    # Set up policy and target neural nets
    policy_net = DQN(observation_space, action_space)
    target_net = DQN(observation_space, action_space)
    
    # Copy weights from policy_net to target
    target_net.load_state_dict(policy_net.state_dict())
    
    # Set target net to eval rather than training mode
    # We do not train target net - ot is copied from policy net at intervals
    target_net.eval()
    
    ############################################################################
    #                            10 Set up memory                              #
    ############################################################################
        
    # Set up memomry
    memory = Memory()
    
    ############################################################################
    #                     11 Set up + start training loop                      #
    ############################################################################
    
    # Set up run counter and learning loop    
    run = 0
    all_steps = 0
    continue_learning = True
    
    # Set up list for results
    results_run = []
    results_exploration = []
    results_score = []
    parser = argparse.ArgumentParser(description='DPSGD')
    parser.add_argument('--batchsize', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--norm-clip', type=float, default=1.0, metavar='M',
                        help='L2 norm clip (default: 1.0)')
    parser.add_argument('--noise-multiplier', type=float, default=1.0, metavar='M',
                        help='Noise multiplier (default: 1.0)') 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--microbatches',type=int, default=1, metavar='N',
                        help='Majority Thresh')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_batch = args.microbatches
    test_batch = args.test_batch_size
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),

    #                    ])),
        
        
    #batch_size=BATCH_SIZE, shuffle=True, kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),


    #                    ])),
    #     batch_size=test_batch , shuffle=True, **kwargs)

    # train_test_loader = torch.utils.data.DataLoader(

    #     datasets.MNIST('./data', train=True, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                    ])),
        # batch_size=test_batch , shuffle=True, **kwargs)
    
    # Continue repeating games (episodes) until target complete
    while continue_learning:
        
        ########################################################################
        #                           12 Play episode                            #
        ########################################################################
        
        # Increment run (episode) counter
        run += 1
        
        ########################################################################
        #                             13 Reset game                            #
        ########################################################################
        
        # Reset game environment and get first state observations
        state = env.reset()
        
        # Reshape state into 2D array with state obsverations as first 'row'
        state = np.reshape(state, [1, observation_space])
        
        # Reset step count for episode
        step = 0
        
        # Continue loop until episode complete
        while True:
            
        ########################################################################
        #                       14 Game episode loop                           #
        ########################################################################
            
            # Incrememnt step counts
            step += 1
            all_steps += 1
            
            ####################################################################
            #                       15 Get action                              #
            ####################################################################
            
            # Get action to take
            action = policy_net.act(state)
            
            ####################################################################
            #                 16 Play action (get S', R, T)                    #
            ####################################################################
            
            # Act
            state_next, reward, terminal, info = env.step(action)
            
            # Update display if needed
            if DISPLAY_ON_SCREEN:
                env.render()
            
            # Convert step reward to negative if end of run
            if terminal and step < TARGET_REWARD:
                reward *= -1
            
            # Get observations for new state
            state_next = np.reshape(state_next, [1, observation_space])
            
            ####################################################################
            #                  17 Add S/A/R/S/T to memory                      #
            ####################################################################
            
            # Record state, action, reward, new state & terminal
            memory.remember(state, action, reward, state_next, terminal)
            
            # Update state
            state = state_next
            
            ####################################################################
            #                  18 Check for end of episode                     #
            ####################################################################
            
            # Actions to take if end of game episode
            if terminal:
                # Get exploration rate
                exploration = policy_net.exploration_rate
                # Clear print row content
                clear_row = '\r' + ' '*79 + '\r'
                print (clear_row, end ='')
                print (f'Run: {run}, ', end='')
                print (f'exploration: {exploration: .3f}, ', end='')
                print (f'score: {step}', end='')
                
                # Add to results lists
                results_run.append(run)
                results_exploration.append(exploration)
                results_score.append(step)
                
                ################################################################
                #             18b Check for end of learning                    #
                ################################################################
                
                # Get average of last 5 games
                if len(results_score) >= TARGET_PERIOD:
                    average_reward = np.mean(results_score[-TARGET_PERIOD:])
                    # Stop if target met in all required episodes
                    #if average_reward >= TARGET_REWARD:
                    if run == 150 :    
                        # Stop learning loop
                        continue_learning = False
                
                # End episode loop
                break
            
            
            ####################################################################
            #                        19 Update policy net                      #
            ####################################################################
            
            # Avoid training model if memory is not of sufficient length
            if len(memory.memory) > REPLAY_START_SIZE:

                # Update policy net 
                optimizer = dpsgd.DPSGD(policy_net.parameters(),lr=args.lr,batch_size=args.batchsize//args.microbatches,C=args.norm_clip,noise_multiplier=args.noise_multiplier)
                for epoch in range(args.epochs):
                    policy_net = DQN(observation_space, action_space) 
                    #model = Net().to(device) 
                    train( policy_net, device, memory.memory, optimizer, epoch,target_net)

                    #test( model, device, test_loader)
                    #test( model, device, train_test_loader)


                ################################################################
                #             20 Update target net periodically                #
                ################################################################
                
                # Use load_state_dict method to copy weights from policy net
                if all_steps % SYNC_TARGET_STEPS == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
    ############################################################################
    #                      21 Learning complete - plot results                 #
    ############################################################################
        
    # Target reached. Plot results
    plot_results(results_run, results_exploration, results_score)


cartpole()    