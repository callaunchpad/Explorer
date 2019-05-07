import numpy as np
import matplotlib.pyplot as plt

"""
init()
reset()
render()
step(action) -> obs, reward, done, garbage
"""

class ObservationSpace:
    def __init__(self, shape):
        self.shape = shape

class ActionSpace:
    def __init__(self, n):
        self.n = n

class GradEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.time = 0
        self.array = np.fromfunction(lambda r, c: np.clip(1, (r + c) * 2, 255), (50, 50))
        self.player = np.array([25, 25])
        self.action_space = ActionSpace(4)
        self.observation_space = ObservationSpace(self.get_array(reshape=True).shape)
        return self.get_array(reshape=True)
    
    def render(self):
        # @source https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
        plt.imshow(self.get_array(reshape=False), cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.ion()
        plt.show()
        plt.pause(0.1)
        plt.close()
    
    def step(self, action):
        self.time += 1
        if action == 0:
            self.player[0] += 1
        elif action == 1:
            self.player[0] -= 1
        elif action == 2:
            self.player[1] += 1
        elif action == 3:
            self.player[1] -= 1
        else:
            raise ValueError(f'{action} is not valid')
        self.clip_player()
        return self.get_array(reshape=True), self.get_reward(), self.get_done(), None
        
    def get_array(self, reshape):
        array = np.copy(self.array)
        array[tuple(self.player)] = 255
        if reshape:
            array = array.reshape((50, 50, 1))
        return array
    
    def get_reward(self):
        return self.array[tuple(self.player)] / 255.0 * 10.0
    
    def get_done(self):
        return self.time >= 100
    
    def clip_player(self):
        self.player = np.clip(self.player, 0, self.array.shape[0] - 1)