import casadi
import numpy as np
import utils

class CEC:
    def __init__(self, traj, Q = np.eye(2), q = 1, R = 0.01*np.eye(2), look_ahead_steps=50) -> None:
        self.traj = traj
        self.Q = Q
        self.q = q
        self.R = R
        self.look_ahead_steps = look_ahead_steps
        self.epsilon = 1e-4

    def car_next_state(self, time_step, cur_state, control):
        theta = cur_state[2]
        G = casadi.vertcat(
            casadi.horzcat(time_step * casadi.cos(theta), 0),
            casadi.horzcat(time_step * casadi.sin(theta), 0),
            casadi.horzcat(0, time_step)
        )
        f =  casadi.mtimes(G, control)
        return cur_state + f
        
    def __call__(self, cur_iter: int, cur_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            cur_iter (int):
            cur_state (np.ndarray): current state
        Returns:
            np.ndarray: control input
        """
        # define optimization variables 
        controls = casadi.MX.sym('U', utils.CONTROL_DIM, self.look_ahead_steps)

        # Propagate states using the motion model
        states = [cur_state]
        for t in range(self.look_ahead_steps):
            next_state = self.car_next_state(utils.time_step, states[-1], controls[:, t])
            states.append(next_state)

        # objective
        objective = 0
        for t in range(1, self.look_ahead_steps+1):
            error_state = states[t] - self.traj(cur_iter + t)
            objective += utils.GAMMA**t * (
                casadi.mtimes(error_state[:2].T, casadi.mtimes(self.Q, error_state[:2]))
                + self.q * (1 - casadi.cos(error_state[2]))**2
                + casadi.mtimes(controls[:, t-1].T, casadi.mtimes(self.R, controls[:, t-1]))
            )
        
        # Define the constrains
        constrains = []
        lbg, ubg = [], []

        for t in range(1, self.look_ahead_steps+1):
            constrains.append(states[t])
            lbg.extend([-3,-3,-casadi.pi])
            ubg.extend([3,3,casadi.pi])
        
            constrains.append(casadi.norm_2(states[t][:2] - casadi.vertcat(-2,-2)))
            lbg.append(0.5)
            ubg.append(float('inf'))
            constrains.append(casadi.norm_2(states[t][:2] - casadi.vertcat(1,2)))
            lbg.append(0.5)
            ubg.append(float('inf'))
        # Lower and upper bounds of variables
        lb_controls = np.tile(np.array([[utils.v_min], [utils.w_min]]), (1, self.look_ahead_steps))
        ub_controls = np.tile(np.array([[utils.v_max], [utils.w_max]]), (1, self.look_ahead_steps))

        # init guesses
        init_guess_controls = np.zeros([2, self.look_ahead_steps])

        # optimization solver
        nlp = { 'x': casadi.reshape(controls, -1, 1),
                'f': objective,
                'g': casadi.vertcat(*constrains)}
        opts = {
            'ipopt': {
                'print_level': 0,
            },
            'print_time': 0
        }
        solver = casadi.nlpsol("S", "ipopt", nlp, opts)
        sol = solver(
            # x0 =init_guess_controls.reshape(-1),  # TODO: initial guess
            lbx=lb_controls.reshape(-1), # TODO: lower bound on optimization variables
            ubx=ub_controls.reshape(-1), # TODO: upper bound on optimization variables
            lbg=lbg, # TODO: lower bound on optimization constraints
            ubg=ubg, # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution
        x_np = np.array(x.full())
        control_solution = x_np.reshape((utils.CONTROL_DIM, self.look_ahead_steps))
        return control_solution[:, 0]

if __name__ == "__main__":
    solver = CEC(utils.lissajous)
    solver(0, utils.lissajous(0))