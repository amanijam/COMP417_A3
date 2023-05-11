import numpy as np


class RL_controller:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lr = args.lr
        self.Q_value = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps, 3)) # state-action values
        self.V_values = np.zeros((args.theta_discrete_steps, args.theta_dot_discrete_steps)) # state values
        self.prev_a = 0 # previous action
        # Use a previous_state = None to detect the beginning of the new round e.g. if not(self.prev_s is None): ...
        self.prev_state = None # Previous state

    def reset(self):
        self.prev_a = 1 # NO ACTION
        self.prev_state = None

    def get_action(self, state, image_state, random_controller=False, episode=0):
        terminal, timestep, theta, theta_dot, reward = state

        if random_controller:
            action = np.random.randint(0, 3) # you have three possible actions (0,1,2)
        else:
            curr_state = [theta, theta_dot]

            # Only use random variables for first 200 episodes
            if np.random.rand() > 0.8 and episode < 200:
                # Explore
                action = np.random.randint(0, 3)
            else:
                # Exploit
                action = np.argmax(self.Q_value[curr_state[0], curr_state[1]])

        self.V_values[curr_state[0], curr_state[1]] = np.max(self.Q_value[curr_state[0], curr_state[1]])

        if not(self.prev_state is None or self.prev_state == [theta, theta_dot]):
            # Update Q values 
            self.Q_value[self.prev_state[0], self.prev_state[1], self.prev_a] += self.lr * (reward + self.gamma * (self.V_values[curr_state[0], curr_state[1]] - self.Q_value[self.prev_state[0], self.prev_state[1], self.prev_a]))
        #############################################################
        
        # print V values
        # if (episode+1) % 10 == 0 and timestep == 2500:
        #     print("V value matrix for episode " + str(episode+1) + ":")
        #     for row in self.V_values:
        #         print(*row, sep="   ")
        #     print("")
        
        self.prev_state = [theta, theta_dot]
        self.prev_a = action
        return action

