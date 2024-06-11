from dataclasses import dataclass
import numpy as np
import torch
import utils
import os
from tqdm import tqdm
import inspect

@dataclass
class GpiConfig:
    traj: callable
    device: torch.device
    dtype: torch.dtype
    obstacles: torch.Tensor
    epsilon: float
    
    nt: int
    nx: int
    ny: int
    nth: int
    nv: int
    nw: int
    
    Q: torch.Tensor
    q: float
    R: torch.Tensor
    gamma: float

    num_iters: int
    num_eval_iters: int  # number of policy evaluations in each iteration
    
    # V: ValueFunction  # your value function implementation
    # output_dir: str
    # # used by feature-based value function
    # v_ex_space: torch.Tensor
    # v_ey_space: torch.Tensor
    # v_etheta_space: torch.Tensor
    # v_alpha: float
    # v_beta_t: float
    # v_beta_e: float
    # v_lr: float
    # v_batch_size: int  # batch size if GPU memory is not enough


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config

        self.ref_states = self.compute_ref_states()
        self.transition_new_states, self.transition_probabilities = self.compute_transition_matrix()
        self.stage_cost = self.compute_stage_costs()
        print(self.check_nan(self.ref_states), self.check_nan(self.transition_new_states), self.check_nan(self.transition_probabilities), self.check_nan(self.stage_cost))
        
        self.policy = torch.zeros((self.config.nt, self.config.nx, self.config.ny, self.config.nth, 2),
                                  dtype=torch.int, device=self.config.device)
         
        self.Q      = torch.full((self.config.nt, self.config.nx, self.config.ny, self.config.nth, self.config.nv, self.config.nw),
                                 4096.0,
                                 dtype=self.config.dtype, device=self.config.device, )
        self.compute_policy()

    @staticmethod
    def check_nan(tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        return "has_nan" if has_nan else "" + "has_inf" if has_inf else ""

    def __call__(self, t: int, cur_state: torch.Tensor) -> torch.Tensor:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (torch.Tensor): current state
            cur_ref_state (torch.Tensor): reference state
        Returns:
            torch.Tensor: control input
        """
        my_cur_state = torch.tensor(cur_state,device=self.config.device, dtype=self.config.dtype)
        state_index = self.state_metric_to_index(my_cur_state)
        return self.policy[t%self.config.nt, state_index[0], state_index[1], state_index[2]].cpu().numpy().astype(cur_state.dtype)

    def compute_ref_states(self, file = "ref_states.pt"):
        if os.path.exists(file):
            re = torch.load(file).to(self.config.device)
            # print("ref states: loading pre-computed", re.shape, re.dtype)
            return re
        ref_states = torch.zeros([self.config.nt, 3],device=self.config.device, dtype=self.config.dtype)
        for t in range(self.config.nt):
            ref_states[t,:] = torch.tensor(self.config.traj(t), dtype=self.config.dtype)
        torch.save(ref_states, file)
        return ref_states

    def compute_transition_matrix(self, file = "transition.pt"):
        """
        Compute the transition matrix in advance to speed up the GPI algorithm.
        """

        if os.path.exists(file):
            # print("transition loading pre-computed")
            loaded_tensors = torch.load(file)
            transition_new_states = loaded_tensors['transition_new_states']
            transition_probabilities = loaded_tensors['transition_probabilities']
            # print("transition_new_states", transition_new_states.shape, transition_new_states.dtype)
            # print("transition_probabilities", transition_probabilities.shape, transition_probabilities.dtype)
            return transition_new_states, transition_probabilities
        
        # the mesh grid of states
        xx, yy, thth, vv, ww = torch.meshgrid(
            torch.linspace(-3, 3, self.config.nx, 
                           device=self.config.device, dtype=self.config.dtype), 
            torch.linspace(-3, 3, self.config.ny, 
                           device=self.config.device, dtype=self.config.dtype),
            torch.linspace(-torch.pi, torch.pi, self.config.nth,
                           device=self.config.device, dtype=self.config.dtype),
            torch.linspace(0, 1, self.config.nv,
                           device=self.config.device, dtype=self.config.dtype),
            torch.linspace(-1, 1, self.config.nw, 
                           device=self.config.device, dtype=self.config.dtype),
            indexing='ij')

        new_states = self.batch_motion_model(xx,yy,thth,vv,ww)
        transition_new_states, transition_probabilities = self.evaluate_discretized_probabilities(new_states)

        # print("transition_new_states", transition_new_states.shape, transition_new_states.dtype)
        # print("transition_probabilities", transition_probabilities.shape, transition_probabilities.dtype)
        # torch.save({'transition_new_states': transition_new_states, 'transition_probabilities': transition_probabilities}, file)
        return transition_new_states, transition_probabilities

    def batch_motion_model(self, xx, yy, thth, vv, ww):
        states = torch.stack((xx, yy, thth), dim=-1)
        # Compute cos(theta) and sin(theta) for each theta in the batch
        cos_th = torch.cos(thth)
        sin_th = torch.sin(thth)

        G = torch.stack([
            torch.stack([cos_th, torch.zeros_like(cos_th)], dim=-1),
            torch.stack([sin_th, torch.zeros_like(sin_th)], dim=-1),
            torch.stack([torch.zeros_like(cos_th), torch.ones_like(cos_th)], dim=-1)
        ], dim=-2)  # Shape: (*batch_size, 3, 2)
                
        # print("G", G.shape)
        controls = torch.stack([vv, ww], dim=-1)  # Shape: (*batch_size, 2)
        controls_expanded = controls.unsqueeze(-1)  # Shape: (*batch_size, 2, 1)
        new_state = states + utils.time_step * torch.matmul(G, controls_expanded).squeeze(-1)  # Shape: (*batch_size, 3)
        # print("new_state",  new_state.shape)
        new_state[...,2] = (new_state[..., 2] + torch.pi) % (2*torch.pi) - torch.pi
        return new_state
    
    def evaluate_discretized_probabilities(self, new_states):
        # print("new_states", new_states.shape)
        # Convert new states to grid indices
        state_indices = self.state_metric_to_index(new_states)
        discrete_state = self.state_index_to_metric(state_indices)
        # print("state_indices", state_indices.shape)

        # Find 6 neighboring points
        dx = 6 / (self.config.nx - 1)
        dy = 6 / (self.config.ny - 1)
        dtheta = (2 * torch.pi) / (self.config.nth - 1)
        
        neighbors = torch.stack([
            discrete_state,
            discrete_state + torch.tensor([dx, 0, 0], device=new_states.device, dtype=self.config.dtype),
            discrete_state - torch.tensor([dx, 0, 0], device=new_states.device, dtype=self.config.dtype),
            discrete_state + torch.tensor([0, dy, 0], device=new_states.device, dtype=self.config.dtype),
            discrete_state - torch.tensor([0, dy, 0], device=new_states.device, dtype=self.config.dtype),
            discrete_state + torch.tensor([0, 0, dtheta], device=new_states.device, dtype=self.config.dtype),
            discrete_state - torch.tensor([0, 0, dtheta], device=new_states.device, dtype=self.config.dtype)
        ], dim=0) # stack the neighbors on dim 0
        neighbors[..., 0] = torch.clamp(neighbors[..., 0], -3, 3)
        neighbors[..., 1] = torch.clamp(neighbors[..., 1], -3, 3)
        neighbors[..., 2] = (neighbors[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi
        # print("neighbors", neighbors.shape, neighbors.dtype)
        
        # Evaluate probability densities
        sigma = torch.diag(torch.tensor(utils.sigma, dtype=self.config.dtype, device=new_states.device))
        dist = torch.distributions.MultivariateNormal(loc=new_states, covariance_matrix=sigma)
        # print("dist: batch_shape", dist.batch_shape, "event shape", dist.event_shape)
        probabilities = dist.log_prob(neighbors).exp().to(self.config.dtype)
        # print("probabilities", probabilities.shape, self.check_nan(probabilities)) # False
        # collision mask
        for i in range(self.config.obstacles.shape[0]):
            r = torch.norm(neighbors[...,:2] - self.config.obstacles[i,:2], p=2, dim=-1)
            # print("r", r.shape)
            mask = r < self.config.obstacles[i,2]
            # print("collision mask", mask.shape)
            probabilities[mask] = 0

        # print("probabilities", probabilities.shape, self.check_nan(probabilities)) 

        # # Normalize probabilities
        normalized_probabilities = torch.nn.functional.normalize(probabilities, p=1, dim=0)
        # print("normalized_probabilities", normalized_probabilities.shape, self.check_nan(normalized_probabilities))

        return neighbors, normalized_probabilities


    def compute_stage_costs(self, file="stage_cost.pt"):
        """
        Compute the stage costs in advance to speed up the GPI algorithm.
        """
        if os.path.exists(file):
            re = torch.load(file).to(self.config.device)
            # print("stage cost (using pre-computed)", re.shape, re.dtype)
            return re
        
        tt, xx, yy, thth, vv, ww = torch.meshgrid(
            torch.linspace(0, self.config.nt-1, self.config.nt,
                         device=self.config.device, dtype=self.config.dtype),
            torch.linspace(-3, 3, self.config.nx, 
                           device=self.config.device, dtype=self.config.dtype), 
            torch.linspace(-3, 3, self.config.ny, 
                           device=self.config.device, dtype=self.config.dtype),
            torch.linspace(-torch.pi, torch.pi, self.config.nth,
                           device=self.config.device, dtype=self.config.dtype),
            torch.linspace(0, 1, self.config.nv,
                           device=self.config.device, dtype=self.config.dtype),
            torch.linspace(-1, 1, self.config.nw, 
                           device=self.config.device, dtype=self.config.dtype),
            indexing='ij')
        tt = tt.to(torch.int)
        xxyy = torch.stack((xx, yy), dim=-1)
        controls = torch.stack([vv,ww], dim=-1)
        ref_states = self.ref_states[tt] 

        delta_p = xxyy - ref_states[...,:2]
        delta_theta = thth - ref_states[...,2]
        cost_term1 = torch.einsum('...i,ij,...j->...', delta_p, self.config.Q, delta_p)
        cost_term2 = torch.pow(1-torch.cos(delta_theta), 2)
        cost_term3 = torch.einsum('...i,ij,...j->...', controls, self.config.R, controls)

        stage_cost = cost_term1 + cost_term2 + cost_term3
        # print("stage_cost", stage_cost.shape, stage_cost.dtype)

        # mask out boundary
        mask = (
            (xx < -3 + self.config.epsilon) | (xx > 3 - self.config.epsilon) |
            (yy < -3 + self.config.epsilon) | (yy > 3 - self.config.epsilon)
        )
        # print("stage_cost boundary mask size", mask.sum().item())
        stage_cost[mask] = 4096.0 # torch.inf
        
        for i in range(self.config.obstacles.shape[0]):
            r = torch.norm(xxyy - self.config.obstacles[i,:2], p=2, dim=-1)
            # print("r", r.shape)
            mask = r < self.config.obstacles[i,2]
            # print("stage_cost obstacle mask size", mask.sum().item())
            stage_cost[mask] = 4096.0 # torch.inf

        # torch.save(stage_cost, file)
        return stage_cost

    # ----- helper converter ----- #
    def state_metric_to_index(self, metric_state: torch.Tensor) -> torch.Tensor:
        """
        Convert the metric state to grid indices according to your descretization design.
        Args:
            metric_state [..., 3]: metric state
        Returns:
            torch.tensor, shape [..., 3]: grid indices
        """
        x, y  = metric_state[..., 0], metric_state[..., 1]
        theta = (metric_state[..., 2]+torch.pi) % (2*torch.pi) - torch.pi

        x_index = torch.round((x + 3) * (self.config.nx - 1) / 6).int()
        y_index = torch.round((y + 3) * (self.config.ny - 1) / 6).int()
        theta_index = torch.round((theta + torch.pi) * (self.config.nth - 1) / (2 * torch.pi)).int()

        x_index = torch.clamp(x_index, 0, self.config.nx - 1)
        y_index = torch.clamp(y_index, 0, self.config.ny - 1)
        theta_index = torch.clamp(theta_index, 0, self.config.nth - 1)

        return torch.stack([x_index, y_index, theta_index], dim=-1)
    
    def control_metric_to_index(self, control_metric: torch.Tensor) -> torch.Tensor:
        """
        Convert the control metrics to grid indices according to your discretization design.
        Args:
            control_metric [..., 2]: controls in metric space
        Returns:
            torch.Tensor, shape [..., 2]: grid indices
        """
        v, w = control_metric[..., 0], control_metric[..., 1]
        
        v_index = torch.round((v + utils.v_min) * (self.config.nv - 1) / (utils.v_max - utils.v_min)).int()
        w_index = torch.round((w + utils.w_min) * (self.config.nw - 1) / (utils.w_max - utils.w_min)).int()

        v_index = torch.clamp(v_index, 0, self.config.nv - 1)
        w_index = torch.clamp(w_index, 0, self.config.nw - 1)
        
        return torch.stack([v_index, w_index], dim=-1)

    def state_index_to_metric(self, state_index: torch.Tensor) -> torch.Tensor:
        """
        Convert the grid indices to metric state according to your descretization design.
        Args:
            state_index [...,3]: grid indices
        Returns:
            torch.Tensor, shape [...,3]: metric state
        """
        x_index, y_index, theta_index = state_index[..., 0], state_index[..., 1], state_index[..., 2]
        
        x = (x_index * 6 / (self.config.nx - 1) - 3).to(self.config.dtype)
        y = (y_index * 6 / (self.config.ny - 1) - 3).to(self.config.dtype)
        theta = (theta_index * (2 * torch.pi) / (self.config.nth - 1) - torch.pi).to(self.config.dtype)
        
        return torch.stack([x, y, theta], dim=-1)

    def control_index_to_metric(self, control_index: torch.Tensor) -> torch.Tensor:
        """
        Convert the grid indices to control metrics according to your discretization design.
        Args:
            control_index [..., 2]: grid indices
        Returns:
            torch.Tensor, shape [..., 2]: controls in metric space
        """
        v_index, w_index = control_index[..., 0], control_index[..., 1]
        
        v_metric = (v_index * (utils.v_max - utils.v_min) / (self.config.nv - 1) + utils.v_min).to(self.config.dtype)
        w_metric = (w_index * (utils.w_max - utils.w_min) / (self.config.nw - 1) + utils.w_min).to(self.config.dtype)
        
        return torch.stack([v_metric, w_metric], dim=-1)
 
    # ----- policy iteration ----- #
    def compute_policy(self) -> None:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        for _ in tqdm(range(self.config.num_iters)):
            self.policy_evaluation(self.config.num_eval_iters)
            self.policy_improvement()

    # @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        # Flatten the last two dimensions
        Q_flattened = self.Q.view(*self.Q.shape[:-2], -1)  # Shape: [100, 31, 31, 19, 66]
        
        # Find the indices of the minimum values along the flattened dimension
        best_flattened_indices = torch.argmin(Q_flattened, dim=-1)  # Shape: [100, 31, 31, 19]

        # Convert the flattened indices back to 2D indices
        last_dim_size = self.Q.shape[-2]  # 6
        best_v_indices = best_flattened_indices // last_dim_size  # Indices for the second last dimension (11)
        best_w_indices = best_flattened_indices % last_dim_size  # Indices for the last dimension (6)

        # Stack the indices together along a new dimension
        best_indices = torch.stack((best_v_indices, best_w_indices), dim=-1)  # Shape: [100, 31, 31, 19, 2]
       
        print("policy_improvement delta", torch.norm((self.policy - best_indices).to(self.config.dtype)).item())
        self.policy = best_indices

    # @utils.timer
    def policy_evaluation(self, num_eval_iters):
        """
        Policy evaluation step of the GPI algorithm.
        """
        # Create a mesh grid of all states
        tt, xx, yy, thth = torch.meshgrid(
            torch.linspace(0, self.config.nt-1, self.config.nt,
                         device=self.config.device),
            torch.linspace(-3, 3, self.config.nx, 
                           device=self.config.device), 
            torch.linspace(-3, 3, self.config.ny, 
                           device=self.config.device),
            torch.linspace(-torch.pi, torch.pi, self.config.nth,
                           device=self.config.device),
            indexing='ij')
        tt = tt.to(torch.int)
        # print("tt", tt.shape)
        # Get the current policy controls for all states
        state_indices = self.state_metric_to_index(torch.stack([xx, yy, thth], dim=-1))
        # print("state_indices", state_indices.shape)

        policy_control_indices = self.policy[tt, state_indices[..., 0], state_indices[..., 1], state_indices[..., 2], :]
        # print("policy_controls", policy_control_indices.shape)

        # Compute the new states using the batch motion model 
        next_state_tt = tt.clone()
        next_state_tt = torch.remainder(next_state_tt + 1, self.config.nt)
        next_state_tt = next_state_tt.unsqueeze(0).expand(7, -1, -1, -1, -1)
        next_states =  self.transition_new_states[
                :, # all neighbors
                state_indices[...,0], state_indices[...,1], state_indices[...,2], 
                policy_control_indices[...,0], policy_control_indices[...,1],
                : # [x,y,th]
            ]
        # print("next_states", next_states.shape)
        next_state_indices = self.state_metric_to_index(next_states)
        next_state_prob = self.transition_probabilities[
            :, # all neighbors
            state_indices[...,0], state_indices[...,1], state_indices[...,2], 
            policy_control_indices[...,0], policy_control_indices[...,1]
        ]
        # print('next_state_prob', next_state_prob.shape)

        # print('stage_cost', self.stage_cost.shape)
        stage_cost = self.stage_cost[
                tt,
                state_indices[...,0], state_indices[...,1], state_indices[...,2], 
                policy_control_indices[...,0], policy_control_indices[...,1]
            ]
        # print("stage_cost_slice", stage_cost.shape)
        
        for eval_iter in range(num_eval_iters):
            next_state_V = torch.min(
                self.Q[next_state_tt, next_state_indices[...,0], next_state_indices[...,1], next_state_indices[...,2]], 
                dim=-1).values
            next_state_V = torch.min(next_state_V, dim=-1).values
            # print('next_state_V', next_state_V.shape, self.check_nan(next_state_V))
            next_state_returns = torch.einsum('aijkl,aijkl->ijkl', next_state_V, next_state_prob)
            # print('new_state_returns', next_state_returns.shape, self.check_nan(next_state_returns))
            new_Q = stage_cost + self.config.gamma * next_state_returns
            
            # if eval_iter % 10 == 0:
            delta = torch.norm(self.Q[
                tt,
                state_indices[...,0], state_indices[...,1], state_indices[...,2], 
                policy_control_indices[...,0], policy_control_indices[...,1]] - new_Q
            ).item()
            print("policy evaluation delta", delta)
            
            self.Q[
                tt,
                state_indices[...,0], state_indices[...,1], state_indices[...,2], 
                policy_control_indices[...,0],policy_control_indices[...,1]
            ] = new_Q

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cpu")
    dtype = torch.float # must use float 32 for probability normalization work correctly

    cfg = GpiConfig(
        utils.line,
        device,
        dtype,
        epsilon=1e-5,
        obstacles=torch.tensor([[-2, -2, 0.5], [1, 2, 0.5]], device=device, dtype=dtype),
        nt=utils.T,
        nx=6*5+1,
        ny=6*5+1,
        nth=2*9+1,
        nv=4+1,
        nw=2*4+1,
        Q=torch.tensor([[10,1],[1,10]], device=device, dtype=dtype),
        q=2,
        R=torch.tensor([[0.05,0],[0,0.05]], device=device, dtype=dtype),
        gamma=utils.GAMMA,
        num_iters=5,
        num_eval_iters=1)

    solver = GPI(cfg)