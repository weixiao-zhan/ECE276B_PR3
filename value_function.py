import numpy as np


class ValueFunction:
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = ex_space
        self.ey_space = ey_space
        self.etheta_space = etheta_space

    def copy_from(self, other):
        """
        Update the underlying value function storage with another value function
        """
        # TODO: your implementation
        raise NotImplementedError

    def update(self, t, ex, ey, etheta, target_value):
        """
        Update the value function at given states
        Args:
            t: time step
            ex: x position error
            ey: y position error
            etheta: theta error
            target_value: target value
        """
        # TODO: your implementation
        raise NotImplementedError

    def __call__(self, t, ex, ey, etheta):
        """
        Get the value function results at given states
        Args:
            t: time step
            ex: x position error
            ey: y position error
            etheta: theta error
        Returns:
            value function results
        """
        # TODO: your implementation
        raise NotImplementedError

    def copy(self):
        """
        Create a copy of the value function
        Returns:
            a copy of the value function
        """
        # TODO: your implementation
        raise NotImplementedError


class GridValueFunction(ValueFunction):
    """
    Grid-based value function
    """
    # TODO: your implementation
    raise NotImplementedError


class FeatureValueFunction(ValueFunction):
    """
    Feature-based value function
    """
    # TODO: your implementation
    raise NotImplementedError


