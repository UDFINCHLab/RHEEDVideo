import numpy as np

class MyModel:
    def __init__(self):
        # Later: load your ML weights here
        pass

    def run(self, frame: np.ndarray):
        # Example inference: compute average intensity
        return float(frame.mean())
