
import numpy as np


class SmoothStream:

    def __init__(self, window_size=10):

        self.smoothed_stream = []
        self.stream = []
        self.current_sum = 0.
        self.window_size = window_size


    def get_moving_avg(self):
        if len(self.stream) == 0:
            return 0.
        if len(self.stream) < self.window_size:
            return self.current_sum / len(self.stream)
        return self.current_sum / self.window_size

    def insert(self, x):
        if self.stream == None:
            self.stream = np.array(x)
        else:
            self.stream.append(x)

        self.current_sum += x

        if len(self.stream) > self.window_size:
            self.current_sum -= self.stream[-(self.window_size+1)]

        self.smoothed_stream.append(self.get_moving_avg())