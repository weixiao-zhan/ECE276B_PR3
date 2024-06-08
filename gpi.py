from dataclasses import dataclass
import numpy as np
from value_function import ValueFunction
import utils


@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int  # number of policy evaluations in each iteration
    collision_margin: float
    V: ValueFunction  # your value function implementation
    output_dir: str
    # used by feature-based value function
    v_ex_space: np.ndarray
    v_ey_space: np.ndarray
    v_etheta_space: np.ndarray
    v_alpha: float
    v_beta_t: float
    v_beta_e: float
    v_lr: float
    v_batch_size: int  # batch size if GPU memory is not enough


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        # TODO: other initialization code

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # TODO: your implementation
        raise NotImplementedError

    def state_metric_to_index(self, metric_state: np.ndarray) -> tuple:
        """
        Convert the metric state to grid indices according to your descretization design.
        Args:
            metric_state (np.ndarray): metric state
        Returns:
            tuple: grid indices
        """
        # TODO: your implementation
        raise NotImplementedError

    def state_index_to_metric(self, state_index: tuple) -> np.ndarray:
        """
        Convert the grid indices to metric state according to your descretization design.
        Args:
            state_index (tuple): grid indices
        Returns:
            np.ndarray: metric state
        """
        # TODO: your implementation
        raise NotImplementedError

    def control_metric_to_index(self, control_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            control_metric: [2, N] array of controls in metric space
        Returns:
            [N, ] array of indices in the control space
        """
        v: np.ndarray = np.digitize(control_metric[0], self.config.v_space, right=True)
        w: np.ndarray = np.digitize(control_metric[1], self.config.w_space, right=True)
        return v, w

    def control_index_to_metric(self, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            v: [N, ] array of indices in the v space
            w: [N, ] array of indices in the w space
        Returns:
            [2, N] array of controls in metric space
        """
        return self.config.v_space[v], self.config.w_space[w]

    def compute_transition_matrix(self):
        """
        Compute the transition matrix in advance to speed up the GPI algorithm.
        """
        # TODO: your implementation
        raise NotImplementedError

    def compute_stage_costs(self):
        """
        Compute the stage costs in advance to speed up the GPI algorithm.
        """
        # TODO: your implementation
        raise NotImplementedError

    def init_value_function(self):
        """
        Initialize the value function.
        """
        # TODO: your implementation
        raise NotImplementedError

    def evaluate_value_function(self):
        """
        Evaluate the value function. Implement this function if you are using a feature-based value function.
        """
        # TODO: your implementation
        raise NotImplementedError

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        # TODO: your implementation
        raise NotImplementedError

    @utils.timer
    def policy_evaluation(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        # TODO: your implementation
        raise NotImplementedError

    def compute_policy(self, num_iters: int) -> None:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        # TODO: your implementation
        raise NotImplementedError
        

