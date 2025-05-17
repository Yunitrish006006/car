import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not found, running without JIT acceleration")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not found, running without progress bar. To install: pip install tqdm")


# set the rows and columns length
BOARD_ROWS = 8
BOARD_COLS = 10

# initalise start, win and lose states
START = (0, 0)
WIN_STATE = (7, 9)

# class state defines the board and decides reward, end and next position
class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False

    def getReward(self):
        if self.state == WIN_STATE:
            return 100
        else:
            return 0

    def isEndFunc(self):
        if self.state == WIN_STATE:
            self.isEnd = True

    def nxtPosition(self, action):
        if action == 0:
            nxtState = (self.state[0] - 1, self.state[1])  # up
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])  # down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)  # left
        else:
            nxtState = (self.state[0], self.state[1] + 1)  # right

        if (nxtState[0] >= 0) and (nxtState[0] < BOARD_ROWS) and (nxtState[1] >= 0) and (nxtState[1] < BOARD_COLS):
            return nxtState

        return self.state


# More efficient array-based agent
class OptimizedAgent:
    def __init__(self):
        # initialize states and actions
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.State = State()
        self.alpha = 0.8
        self.epsilon = 0.5
        self.isEnd = self.State.isEnd

        # array to retain reward values for plot
        self.plot_reward = []

        # initialize Q values as numpy arrays for faster access
        self.Q = np.zeros((BOARD_ROWS, BOARD_COLS, 4))
        self.new_Q = np.zeros((BOARD_ROWS, BOARD_COLS, 4))
        self.rewards = 0
        
        # Pre-load discount factors to avoid pandas lookups in the loop
        self.discount_factors = None
    
    def load_discount_factors(self, path):
        """Pre-load all discount factors into a numpy array for faster access"""
        df = pd.read_csv(path, index_col="Grid 座標")
        self.discount_factors = np.zeros((BOARD_ROWS, BOARD_COLS, 4))
        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                idx = f"{i}_{j}"
                if idx in df.index:
                    factors = df.loc[idx, "discount factor"].split("/")
                    for a in range(4):
                        self.discount_factors[i, j, a] = float(factors[a])
        
        print("Discount factors loaded successfully")

    def Action(self):
        """Choose action based on epsilon-greedy policy"""
        i, j = self.State.state
        
        # Choose exploitation vs exploration
        if random.random() > self.epsilon:
            # Choose the best action based on Q values
            action = np.argmax(self.Q[i, j, :])
        else:
            # Choose random action
            action = np.random.choice(self.actions)
            
        # Get next position
        position = self.State.nxtPosition(action)
        return position, action

    def Q_Learning(self, episodes):
        """Optimized Q-learning implementation"""
        if self.discount_factors is None:
            print("Error: Must load discount factors first with load_discount_factors()")
            return
            
        episode = 0
        
        # Create progress bar
        if HAS_TQDM:
            pbar = tqdm(total=episodes, desc="Training Progress", ncols=100)
        else:
            print(f"Starting training for {episodes} episodes...")
            
        last_episode_report = 0
        report_interval = max(1, episodes // 100)  # Report every 1% of progress
        
        # Add a step counter to update progress more frequently
        steps = 0
        steps_per_update = 1000  # Update progress every 1000 steps
        max_steps_per_episode = 1000  # Limit to prevent infinite loops
        
        while episode < episodes:
            if self.isEnd:
                # At terminal state
                reward = self.State.getReward()
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                
                # Update Q values for terminal state
                i, j = self.State.state
                self.new_Q[i, j, :] = reward
                
                # Reset for next episode
                self.State = State()
                self.isEnd = self.State.isEnd
                self.rewards = 0
                episode += 1
                
                # Update progress bar
                if HAS_TQDM:
                    pbar.update(1)
                elif episode - last_episode_report >= report_interval:
                    print(f"Progress: {episode}/{episodes} episodes ({episode/episodes*100:.1f}%)")
                    last_episode_report = episode
            else:
                # Choose action and get next state
                next_state, action = self.Action()
                i, j = self.State.state
                
                # Get current reward
                reward = self.State.getReward()
                self.rewards += reward
                
                # Calculate new Q value
                next_i, next_j = next_state
                max_next_q = np.max(self.Q[next_i, next_j, :])
                df = self.discount_factors[i, j, action]
                
                # Q-value update
                new_q = (1 - self.alpha) * self.Q[i, j, action] + self.alpha * (
                    reward + df * max_next_q
                )
                
                # Store new Q value
                self.new_Q[i, j, action] = round(new_q, 3)
                
                # Update state
                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                
                # Increment steps and check for limit to avoid infinite loops
                steps += 1
                if steps % steps_per_update == 0 and HAS_TQDM:
                    # Update progress bar description to show we're not stuck
                    pbar.set_description(f"Training: Episode {episode+1}/{episodes}, Steps: {steps}")
                
                # Break if stuck in an episode for too long
                if steps % max_steps_per_episode == 0 and steps > 0:
                    # Reset and move to next episode
                    self.State = State()
                    self.isEnd = self.State.isEnd
                    self.rewards = 0
                    episode += 1
                    if HAS_TQDM:
                        pbar.update(1)
                        pbar.set_description(f"Training Progress (Forced next: {episode}/{episodes})")
                    steps = 0
                
            # Update Q table after each step
            self.Q = self.new_Q.copy()
            
        # Close progress bar
        if HAS_TQDM:
            pbar.close()
    
    def saveQTable(self, filename="full_q_table_optimized.npy"):
        """Save the Q-table to a file"""
        np.save(filename, self.Q)
        print(f"Full Q-table saved as {filename}")
    
    def showBestPath(self):
        """Display the best path strategy"""
        print("\n===== Best Path Strategy =====")
        direction = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        path_grid = [[' ' for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                best_a = np.argmax(self.Q[i, j, :])
                path_grid[i][j] = direction[best_a]
        
        path_grid[WIN_STATE[0]][WIN_STATE[1]] = 'G'
        path_grid[START[0]][START[1]] = 'S'
        
        for row in path_grid:
            print(' '.join(row))
        print("=========================\n")
    
    def showValues(self, arr):
        """Display the max Q-values for each state"""
        outArr = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=str)
        
        for i in range(BOARD_ROWS):
            print("-----------------------------------------------------------------------------------------------------------")
            out = "| "
            for j in range(BOARD_COLS):
                max_val = np.max(self.Q[i, j, :])
                out += str(max_val).ljust(6) + " | "
                outArr[i, j] = str(max_val)
            print(out)
        print("-----------------------------------------------------------------------------------------------------------")
        
        # Store values for neighboring cells of START
        i, j = START
        if i - 1 >= 0:
            arr[0] = float(outArr[i-1, j])
        else:
            arr[0] = -1
            
        if i + 1 < BOARD_ROWS:
            arr[1] = float(outArr[i+1, j])
        else:
            arr[1] = -1
            
        if j - 1 >= 0:
            arr[2] = float(outArr[i, j-1])
        else:
            arr[2] = -1
            
        if j + 1 < BOARD_COLS:
            arr[3] = float(outArr[i, j+1])
        else:
            arr[3] = -1
            
        print(arr)


def test_epsilon_performance(agent_class=OptimizedAgent):
    """Test performance with different epsilon values"""
    print("\n===== Testing Epsilon Performance =====")
    epsilons = np.linspace(0, 1, 5)  # 0.0, 0.25, 0.5, 0.75, 1.0
    times = []
    test_episodes = 1000  # Significantly reduced for faster testing
    
    # Create a progress bar for the epsilon tests if tqdm is available
    if HAS_TQDM:
        epsilon_pbar = tqdm(epsilons, desc="Testing Epsilons", ncols=100)
    else:
        print(f"Testing {len(epsilons)} different epsilon values...")
        epsilon_pbar = epsilons
    
    for eps in epsilon_pbar:
        test_ag = agent_class()
        test_ag.epsilon = eps
        test_ag.load_discount_factors("hw/hw6/demo code/discount_factor.csv")
        
        start_time = time.time()
        test_ag.Q_Learning(test_episodes)
        end_time = time.time()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        # Update description with current result
        if HAS_TQDM:
            epsilon_pbar.set_description(f"Epsilon: {eps:.1f}, Time: {execution_time:.2f}s")
        else:
            print(f"Epsilon: {eps:.1f}, Execution Time: {execution_time:.2f} seconds")
    
    # Plot epsilon vs. execution time
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, times, marker='o', linestyle='-')
    plt.xlabel('Epsilon Value')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Epsilon vs. Execution Time')
    plt.grid(True)
    plt.savefig("epsilon_performance_optimized.png")
    plt.show()


if __name__ == "__main__":
    # Generate discount factors
    from gen_discount_factor import generate_discount_factors
    generate_discount_factors()
    
    # Create agent with optimized implementation
    print("Starting optimized Q-learning with array-based implementation")
    episodes = 10000
    start_time = time.time()
    
    ag = OptimizedAgent()
    ag.load_discount_factors("hw/hw6/demo code/discount_factor.csv")
    
    filename = "q_table_optimized"
    q_value = np.zeros(4, dtype=np.float64)
    
    print(f"Start: {START}, Goal: {WIN_STATE}")
    
    # Run Q-learning
    ag.Q_Learning(episodes)
    
    end_time = time.time()
    print(f"Optimized Q-learning completed in {end_time - start_time:.2f} seconds")
    
    # Show results
    ag.showValues(q_value)
    np.save(os.path.join(filename), q_value)
    ag.showBestPath()
    ag.saveQTable("full_q_table_optimized.npy")
    
    # Test epsilon performance
    test_epsilon_performance()
