import casadi
import numpy as np
import utils

class CEC:
    def __init__(self, traj, 
                 Q , q , 
                 R, r ,
                 look_ahead_steps) -> None:
        self.traj = traj
        self.Q = Q
        self.q = q
        self.R = R
        self.r = r
        self.look_ahead_steps = look_ahead_steps

    def car_next_state(self, cur_state, control):
        theta = cur_state[2]
        G = casadi.vertcat(
            casadi.horzcat(utils.time_step * casadi.cos(theta), 0),
            casadi.horzcat(utils.time_step * casadi.sin(theta), 0),
            casadi.horzcat(0, utils.time_step)
        )
        f =  G @ control
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
            next_state = self.car_next_state(states[-1], controls[:, t])
            states.append(next_state)

        # objective
        objective = 0
        for t in range(1, self.look_ahead_steps+1):
            error_state = states[t] - self.traj(cur_iter + t)
            p = error_state[:2]
            theta = error_state[2]
            objective += utils.GAMMA**t * (
                p.T @ self.Q @ p
                + self.q * (1 - casadi.cos(theta))**2
                + controls[:, t-1].T @ self.R @ controls[:, t-1]
            )
        
        for t in range(self.look_ahead_steps-1):
            delta_control = controls[:,t] - controls[:,t-1]
            objective += utils.GAMMA**t * (
               casadi.mtimes(delta_control.T, casadi.mtimes(self.r, delta_control))
            )
        
        # Define the constrains
        constrains = []
        lbg, ubg = [], []

        for t in range(1, self.look_ahead_steps+1):
            constrains.append(states[t][:2])
            lbg.extend([-3,-3])
            ubg.extend([3,3])
        
            constrains.append(casadi.norm_2(states[t][:2] - casadi.vertcat(-2,-2)))
            lbg.append(0.5)
            ubg.append(casadi.inf)
            constrains.append(casadi.norm_2(states[t][:2] - casadi.vertcat(1,2)))
            lbg.append(0.5)
            ubg.append(casadi.inf)

        # Lower and upper bounds of variables
        lb_controls = [utils.v_min, utils.w_min]*self.look_ahead_steps #+ [utils.w_min]*self.look_ahead_steps
        ub_controls = [utils.v_max, utils.w_max]*self.look_ahead_steps #+ [utils.w_max]*self.look_ahead_steps 
        # lb_controls = np.tile(np.array([[utils.v_min], [utils.w_min]]), (1, self.look_ahead_steps)).T.reshape(-1)
        # ub_controls = np.tile(np.array([[utils.v_max], [utils.w_max]]), (1, self.look_ahead_steps)).T.reshape(-1)

        # init guesses
        init_guess_controls = [0.2, 0.01]*self.look_ahead_steps

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
            x0 =init_guess_controls,  # TODO: initial guess
            lbx=lb_controls, # TODO: lower bound on optimization variables
            ubx=ub_controls, # TODO: upper bound on optimization variables
            lbg=lbg, # TODO: lower bound on optimization constraints
            ubg=ubg, # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution
        return np.array(x)[:2].squeeze()

if __name__ == "__main__":
    solver = CEC(
        utils.lissajous,
        Q = casadi.MX(np.array([[2,0],[0,2]])),
        q = 2,
        R = 0.05*casadi.MX(np.eye(2)),
        r = 0.05*np.eye(2),
        look_ahead_steps=5
    )
    solver(0, utils.lissajous(0))