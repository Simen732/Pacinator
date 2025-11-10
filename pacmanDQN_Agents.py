# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from DQN import *

params = {
    # Model backups
    'load_file': None,       # Start fresh with new reward function
    'save_file': 'Pac-inator-WIN-FOCUSED',
    'save_interval': 50000,  # Save checkpoints every 50k steps
    
    # Logging
    'log_interval': 50,     # Log every 50 episodes
    
    # Training parameters - OPTIMIZED FOR LEARNING TO WIN
    'train_start': 500,      # Start training after 500 steps
    'batch_size': 64,        # Good batch size for CPU
    'mem_size': 20000,       # Larger memory for diverse experiences
    
    'discount': 0.99,        # High discount = plan for long-term (winning)
    'lr': .0001,             # Good learning rate
    
    # Epsilon decay - Allow exploration to find winning strategies
    'eps': 1.0,              # Start with full exploration
    'eps_final': 0.1,        # End with 10% exploration (was 0.05)
    'eps_step': 50000        # Slower decay - explore more to find wins
}                     



class PacmanDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args.get('numTraining', 0)  # Default to 0 if not provided

        # Start Tensorflow session - CPU optimized
        import os
        # Set environment variables for maximum CPU usage
        os.environ['OMP_NUM_THREADS'] = '16'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
        os.environ['TF_NUM_INTEROP_THREADS'] = '16'
        
        config = tf.ConfigProto(
            intra_op_parallelism_threads=16,  # Max CPU cores
            inter_op_parallelism_threads=16,  # Parallel operations
            allow_soft_placement=True,
            device_count={'CPU': 1}
        )
        self.sess = tf.Session(config=config)
        self.qnet = DQN(self.params)
        
        # Print device info
        print("Running on CPU (GPU not available)")
        print("CPU parallelism MAXIMIZED (16 threads) for best performance")

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)  # Training step counter (from TF)
        # Use training steps (cnt) for epsilon decay - this persists across checkpoint loads
        # This ensures epsilon continues from correct value when resuming training
        self.local_cnt = 0    # Steps in current episode

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()
        
        # Track all episode rewards for cumulative averaging
        self.all_episode_rewards = []  # Track all rewards for cumulative average
        
        # Frame stacking: Keep last 4 frames for temporal information
        self.frame_buffer = deque(maxlen=4)  # Buffer for frame stacking


    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 24)), 
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST
    
    def getNearestFoodDistance(self, state):
        """Get Manhattan distance to nearest food pellet"""
        pacman_pos = state.getPacmanPosition()
        food_grid = state.getFood()
        food_list = food_grid.asList()
        
        if len(food_list) == 0:
            return 0
        
        distances = [util.manhattanDistance(pacman_pos, food) for food in food_list]
        return min(distances)
    
    def getNearestGhostDistance(self, state):
        """Get Manhattan distance to nearest ghost"""
        pacman_pos = state.getPacmanPosition()
        ghost_states = state.getGhostStates()
        
        if len(ghost_states) == 0:
            return 999  # No ghosts
        
        ghost_positions = [ghost.getPosition() for ghost in ghost_states]
        distances = [util.manhattanDistance(pacman_pos, pos) for pos in ghost_positions]
        return min(distances)
            
    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            current_frame = self.getStateMatrices(state)
            self.current_state = self.getStackedState(current_frame)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score
            
            food_count = state.getNumFood()

            # ============================================================
            # SIMPLIFIED REWARD: FOCUS ON WINNING
            # Use actual game score changes + massive win bonus + death penalty
            # ============================================================
            
            # Base reward is the actual game score change
            self.last_reward = reward
            
            # Check if this is a death (large negative score)
            if reward < -10:
                # DEATH - Massive penalty to discourage dying
                self.last_reward = -500  # Heavy death penalty
                self.won = False
            
            # Small penalty for each step to encourage efficiency
            self.last_reward -= 1
            
            # WIN BONUS - Applied when game ends with a win
            if(self.terminal and self.won):
                # HUGE win bonus - make winning extremely desirable
                self.last_reward += 5000
            self.ep_rew += self.last_reward

            # Store last experience into memory 
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        # Use training steps (cnt) for epsilon decay - persists across checkpoints
        # Skip decay if both eps and eps_final are 0 (testing mode)
        if not (self.params['eps'] == 0 and self.params['eps_final'] == 0):
            self.params['eps'] = max(self.params['eps_final'],
                                     1.00 - float(self.cnt) / float(self.params['eps_step']))


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)
        
        # Track episode reward for cumulative averaging
        self.all_episode_rewards.append(self.ep_rew)
        
        # Calculate average reward over ALL episodes so far
        avg_reward = np.mean(self.all_episode_rewards)

        # Print stats (only every N episodes to reduce I/O)
        if self.numeps % self.params['log_interval'] == 0:
            log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
            log_file.write("# %4d | steps_t: %5d | t: %4f | r: %12f | avg_r: %12f | e: %10f " %
                             (self.numeps, self.cnt, time.time()-self.s, self.ep_rew, avg_reward, self.params['eps']))
            log_file.write("| Q: %10f | won: %r | game_score: %d \n" % ((max(self.Q_global, default=float('nan')), self.won, self.current_score)))
            log_file.close()
            sys.stdout.write("# %4d | steps_t: %5d | t: %4f | r: %12f | avg_r: %12f | e: %10f " %
                             (self.numeps, self.cnt, time.time()-self.s, self.ep_rew, avg_reward, self.params['eps']))
            sys.stdout.write("| Q: %10f | won: %r | game_score: %d \n" % ((max(self.Q_global, default=float('nan')), self.won, self.current_score)))
            sys.stdout.flush()
        
        # Save model when training is complete
        if self.numeps == self.params['num_training'] and params['save_file']:
            final_model_name = 'saves/model-' + params['save_file'] + '_FINAL_' + str(self.cnt) + '_' + str(self.numeps)
            self.qnet.save_ckpt(final_model_name)
            print('\n' + '='*60)
            print('TRAINING COMPLETE!')
            print('Final model saved: ' + final_model_name)
            print('Total episodes: %d | Total steps: %d' % (self.numeps, self.cnt))
            print('Final average reward: %.2f' % avg_reward)
            print('='*60 + '\n')

    def train(self):
        # Train every 8 steps for CPU efficiency
        # Less frequent = faster on CPU, still effective learning
        if (self.local_cnt > self.params['train_start']) and \
           (len(self.replay_mem) >= self.params['batch_size']) and \
           (self.local_cnt % 8 == 0):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation
    
    def getStackedState(self, current_frame):
        """
        Stack the last 4 frames together to provide temporal information.
        This allows the network to infer velocity and direction.
        """
        # Add current frame to buffer
        self.frame_buffer.append(current_frame)
        
        # If we don't have 4 frames yet (start of episode), duplicate the current frame
        while len(self.frame_buffer) < 4:
            self.frame_buffer.append(current_frame)
        
        # Stack frames along the channel dimension
        # Shape: (height, width, 6) x 4 frames -> (height, width, 24)
        stacked = np.concatenate(list(self.frame_buffer), axis=2)
        
        return stacked

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        
        # Clear frame buffer and initialize with current frame
        self.frame_buffer.clear()
        current_frame = self.getStateMatrices(state)
        self.current_state = self.getStackedState(current_frame)

        # Reset actions
        self.last_action = None
        self.action_history = deque(maxlen=5)  # Track last 5 actions for oscillation detection

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0
        
        # Track total pellets at start for progress calculation
        self.total_food = state.getNumFood()
        
        # Initialize food distance tracking
        self.last_food_distance = self.getNearestFoodDistance(state)
        
        # Initialize pellet value decay tracking
        self.steps_since_pellet = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Ensure move is legal - prefer STOP over invalid moves
        # This maintains consistency between chosen action and executed action
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP  # Deterministic fallback

        return move
