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
    idx_dtype: torch.dtype
    obstacles: torch.Tensor
    
    nt: int
    nx: int
    ny: int
    nth: int
    nv: int
    nw: int
    num_neighbors: int

    Q: torch.Tensor
    q: float
    R: torch.Tensor
    gamma: float

    num_iters: int
    delta: float

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

        self.t_space  = torch.linspace(0, self.config.nt-1, self.config.nt,
                         device=self.config.device, dtype=self.config.dtype)
        self.x_space  = torch.linspace(-3, 3, self.config.nx, 
                           device=self.config.device, dtype=self.config.dtype)
        self.y_space  = torch.linspace(-3, 3, self.config.ny, 
                           device=self.config.device, dtype=self.config.dtype) 
        self.th_space = torch.linspace(-torch.pi, torch.pi, self.config.nth,
                           device=self.config.device, dtype=self.config.dtype)
        self.v_space  = torch.linspace(0, 1, self.config.nv,
                           device=self.config.device, dtype=self.config.dtype)
        self.w_space  = torch.linspace(-1, 1, self.config.nw, 
                           device=self.config.device, dtype=self.config.dtype) 

        self.t_index_space  = torch.arange(self.config.nt,  device=self.config.device)
        self.x_index_space  = torch.arange(self.config.nx,  device=self.config.device)
        self.y_index_space  = torch.arange(self.config.ny,  device=self.config.device)
        self.th_index_space = torch.arange(self.config.nth, device=self.config.device)
        self.v_index_space  = torch.arange(self.config.nv,  device=self.config.device)
        self.w_index_space  = torch.arange(self.config.nw,  device=self.config.device)

        self.ref_states = self.compute_ref_states()
        self.transition_neighborhood_indices, self.transition_probabilities = \
            self.compute_transition_matrix()
        self.stage_cost = self.compute_stage_costs()
        print(self.check_nan(self.ref_states), self.check_nan(self.transition_neighborhood_indices), self.check_nan(self.transition_probabilities), self.check_nan(self.stage_cost))
        
        self.policy = torch.zeros((self.config.nt, self.config.nx, self.config.ny, self.config.nth, 2),
                                  dtype=torch.int, device=self.config.device)
         
        self.Q      = torch.zeros((self.config.nt, self.config.nx, self.config.ny, self.config.nth, self.config.nv, self.config.nw),
                                 dtype=self.config.dtype, device=self.config.device, )
        self.compute_policy()

    @staticmethod
    def check_nan(tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        return "has_nan" if has_nan else "" + "has_inf" if has_inf else ""

    def __call__(self, t: int, cur_state: np.ndarray) -> np.ndarray:
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
        index = self.state_metric_to_index(my_cur_state)
        return self.policy[t%self.config.nt, index[0], index[1], index[2]].numpy().astype(cur_state.dtype)

    def compute_ref_states(self, file = "ref_states.pt"):
        if os.path.exists(file):
            re = torch.load(file).to(self.config.device)
            # print("ref states: loading pre-computed", re.shape, re.dtype)
            return re
        ref_states = torch.zeros([self.config.nt, 3],device=self.config.device, dtype=self.config.dtype)
        for t in range(self.config.nt):
            ref_states[t,:] = torch.tensor(self.config.traj(t), dtype=self.config.dtype)
        # torch.save(ref_states, file)
        return ref_states

    def compute_transition_matrix(self, file = "transition.pt"):
        """
        Compute the transition matrix in advance to speed up the GPI algorithm.
        """
        if os.path.exists(file):
            # print("transition loading pre-computed")
            loaded_tensors = torch.load(file)
            transition_neighborhood_indices = loaded_tensors['transition_neighborhood_indices']
            transition_probabilities = loaded_tensors['transition_probabilities']
            print("transition_neighborhood_indices", transition_neighborhood_indices.shape, transition_neighborhood_indices.dtype)
            print("transition_probabilities", transition_probabilities.shape, transition_probabilities.dtype)
            return transition_neighborhood_indices, transition_probabilities
        
        # the mesh grid of states
        xx, yy, thth, vv, ww = torch.meshgrid(
            self.x_space,
            self.y_space,
            self.th_space,
            self.v_space,
            self.w_space,
            indexing='ij')
    
        new_states = self.batch_motion_model(
            states=torch.stack([xx, yy, thth], dim=-1), 
            controls=torch.stack([vv, ww], dim=-1))
        transition_neighborhood_indices, transition_probabilities = \
            self.evaluate_discretized_probabilities(new_states)
        
        print("transition_neighborhood_indices", transition_neighborhood_indices.shape, transition_neighborhood_indices.dtype)
        print("transition_probabilities", transition_probabilities.shape, transition_probabilities.dtype)
        return transition_neighborhood_indices, transition_probabilities
        # torch.save({'transition_neighborhood_indices': transition_neighborhood_indices, 'transition_probabilities': transition_probabilities}, file)

    def batch_motion_model(self, states, controls):
        # Compute cos(theta) and sin(theta) for each theta in the batch
        cos_th = torch.cos(states[...,2])
        sin_th = torch.sin(states[...,2])

        G = torch.stack([
            torch.stack([cos_th, torch.zeros_like(cos_th)], dim=-1),
            torch.stack([sin_th, torch.zeros_like(sin_th)], dim=-1),
            torch.stack([torch.zeros_like(cos_th), torch.ones_like(cos_th)], dim=-1)
        ], dim=-2)  # Shape: (*batch_size, 3, 2)
        # print("G", G.shape)

        controls_expanded = controls.unsqueeze(-1)  # Shape: (*batch_size, 2, 1)
        new_state = states + utils.time_step * torch.matmul(G, controls_expanded).squeeze(-1)  # Shape: (*batch_size, 3)
        new_state[...,2] = (new_state[..., 2] + torch.pi) % (2*torch.pi) - torch.pi
        print("new_state",  new_state.shape, new_state.dtype)
        return new_state
    
    def evaluate_discretized_probabilities(self, new_states):
        # print("new_states", new_states.shape)
        # Convert new states to grid indices
        new_state_indices = self.state_metric_to_index(new_states)
        print("new_state_indices", new_state_indices.shape, new_state_indices.dtype)

        return new_state_indices.unsqueeze(0), torch.ones(new_state_indices.shape[:-1], device=self.config.device, dtype=self.config.dtype).unsqueeze(0)

        # Find 6 neighboring points
        neighbor_indices = torch.stack([
            new_state_indices,
            # new_state_indices + torch.tensor([1, 0, 0], device=self.config.device, dtype=self.config.idx_dtype),
            # new_state_indices - torch.tensor([1, 0, 0], device=self.config.device, dtype=self.config.idx_dtype),
            # new_state_indices + torch.tensor([0, 1, 0], device=self.config.device, dtype=self.config.idx_dtype),
            # new_state_indices - torch.tensor([0, 1, 0], device=self.config.device, dtype=self.config.idx_dtype),
            # new_state_indices + torch.tensor([0, 0, 1], device=self.config.device, dtype=self.config.idx_dtype),
            # new_state_indices - torch.tensor([0, 0, 1], device=self.config.device, dtype=self.config.idx_dtype)
        ], dim=0) # stack the neighbors on dim 0

        neighbor_indices[..., 0] = torch.clamp(neighbor_indices[..., 0], 0, self.config.nx)
        neighbor_indices[..., 1] = torch.clamp(neighbor_indices[..., 1], 0, self.config.ny)
        neighbor_indices[..., 2] = neighbor_indices[..., 2] % self.config.nth
        print("neighbor_indices", neighbor_indices.shape, neighbor_indices.dtype)
        
        neighbor_states = torch.stack([
            self.x_space[neighbor_indices[...,0]],
            self.y_space[neighbor_indices[...,1]],
            self.th_space[neighbor_indices[...,2]]
        ],dim=-1)
        print("neighbor_states", neighbor_states.shape)

        # Evaluate probability densities
        sigma = torch.tensor([[0.4,0,0],[0,0.4,0],[0,0,0.04]], dtype=self.config.dtype, device=self.config.device)
        dist = torch.distributions.MultivariateNormal(loc=new_states, covariance_matrix=sigma)
        print("dist: batch_shape", dist.batch_shape, "event shape", dist.event_shape)

        probabilities = dist.log_prob(neighbor_states).exp().to(self.config.dtype)

        # collision mask
        for i in range(self.config.obstacles.shape[0]):
            r = torch.norm(neighbor_states[...,:2] - self.config.obstacles[i,:2], p=2, dim=-1)
            # print("r", r.shape)
            mask = r < self.config.obstacles[i,2]
            # print("collision mask", mask.shape)
            probabilities[mask] = 0
        print("probabilities", probabilities.shape, self.check_nan(probabilities)) 

        # # Normalize probabilities
        normalized_probabilities = torch.nn.functional.normalize(probabilities, p=1, dim=0)
        # print("normalized_probabilities", normalized_probabilities.shape, self.check_nan(normalized_probabilities))

        return neighbor_indices, normalized_probabilities


    def compute_stage_costs(self, file="stage_cost.pt"):
        """
        Compute the stage costs in advance to speed up the GPI algorithm.
        """
        if os.path.exists(file):
            re = torch.load(file).to(self.config.device)
            # print("stage cost (using pre-computed)", re.shape, re.dtype)
            return re
        
        tt, xx, yy, thth, vv, ww = torch.meshgrid(
            self.t_space, self.x_space, self.y_space, self.th_space, self.v_space, self.w_space,
            indexing='ij')
        tt = tt.to(torch.int)

        p = torch.stack((xx, yy), dim=-1)
        controls = torch.stack([vv,ww], dim=-1)
        ref_states = self.ref_states[tt] 

        delta_p = p - ref_states[...,:2]
        delta_theta = thth - ref_states[...,2]
        cost_term1 = torch.einsum('...i,ij,...j->...', delta_p, self.config.Q, delta_p)
        cost_term2 = torch.pow(1-torch.cos(delta_theta), 2)
        cost_term3 = torch.einsum('...i,ij,...j->...', controls, self.config.R, controls)
        stage_cost = cost_term1 + cost_term2 + cost_term3
        print("stage_cost", stage_cost.shape, stage_cost.dtype)
        
        for i in range(self.config.obstacles.shape[0]):
            r = torch.norm(p - self.config.obstacles[i,:2], p=2, dim=-1)
            # print("r", r.shape)
            mask = r < self.config.obstacles[i,2]
            # print("stage_cost obstacle mask size", mask.sum().item())
            stage_cost[mask] = 1e10 #torch.inf

        # torch.save(stage_cost, file)
        return stage_cost

    # ----- helper converter ----- #
    def state_metric_to_index(self, states: torch.Tensor) ->torch.Tensor:
        """
        Convert the metric state to grid indices according to your descretization design.
        """
        x = torch.clamp(states[...,0], -3, 3)
        y = torch.clamp(states[...,1], -3, 3)
        theta = (states[...,2]+torch.pi) % (2*torch.pi) - torch.pi

        x_index = torch.round((x + 3) * (self.config.nx - 1) / 6).to(self.config.idx_dtype)
        y_index = torch.round((y + 3) * (self.config.ny - 1) / 6).to(self.config.idx_dtype)
        theta_index = torch.round((theta + torch.pi) * (self.config.nth - 1) / (2 * torch.pi)).to(self.config.idx_dtype)

        return torch.stack([x_index, y_index, theta_index],dim=-1)
    
    def control_metric_to_index(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Convert the control metrics to grid indices according to your discretization design.
        """
        v_index = torch.round((controls[...,0] + utils.v_min) * (self.config.nv - 1) / (utils.v_max - utils.v_min)).to(self.config.idx_dtype)
        w_index = torch.round((controls[...,1] + utils.w_min) * (self.config.nw - 1) / (utils.w_max - utils.w_min)).to(self.config.idx_dtype)
        
        return torch.stack([v_index, w_index], dim=-1)
 
    # ----- policy iteration ----- #
    def compute_policy(self) -> None:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        self.Q_iteration()
        self.policy_extraction()

    # @utils.timer
    def Q_iteration(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        # Create a mesh grid of all states
        t_idx, x_idx, y_idx, th_idx, v_idx, w_idx = torch.meshgrid(
            self.t_index_space, self.x_index_space, self.y_index_space, self.th_index_space, self.v_index_space, self.w_index_space,
            indexing='ij')


        # Efficient next_state_t for all next states' neighbors
        next_neighborhood_t_idx = ((t_idx + 1) % self.config.nt).unsqueeze(0)\
            .expand(self.transition_neighborhood_indices.shape[0], -1, -1, -1, -1, -1, -1)  

        next_neighborhood_indices =  self.transition_neighborhood_indices[
                :, # all neighbors
                x_idx, y_idx, th_idx, 
                v_idx, w_idx,
                : # [x,y,th]
            ]
        print("next_states", next_neighborhood_indices.shape)
        next_neighborhood_prob = self.transition_probabilities[
            :, # all neighbors
            x_idx, y_idx, th_idx, 
            v_idx, w_idx,
        ]
        print('next_state_prob', next_neighborhood_indices.shape)
        
        print("* --- start Q-value iteration --- * ")
        bar = tqdm(range(self.config.num_iters))
        for eval_iter in bar:
            V = torch.min(torch.min(self.Q, dim=-1).values, dim=-1).values
            next_neighborhood_V = V[
                next_neighborhood_t_idx, 
                next_neighborhood_indices[...,0], next_neighborhood_indices[...,1], next_neighborhood_indices[...,2]
            ]
            # print('next_neighborhood_V', next_neighborhood_V.shape, self.check_nan(next_neighborhood_V))
            next_state_returns = torch.einsum('a...,a...->...', next_neighborhood_V, next_neighborhood_prob)
            # print('new_state_returns', next_state_returns.shape, self.check_nan(next_state_returns))
            next_Q = self.stage_cost + self.config.gamma * next_state_returns
            
            if eval_iter %10 == 0:
                delta = torch.norm(self.Q - next_Q).item()
                bar.set_postfix({"delta-Q norm": f"{delta:.3f}"}) 
                if delta < self.config.delta:
                    self.Q = next_Q
                    return
            self.Q = next_Q
    
    # @utils.timer
    def policy_extraction(self):
        """
        extract policy from Q
        """
        # Flatten the last two dimensions
        Q_flattened = self.Q.view(*self.Q.shape[:-2], -1)  # Shape: [100, 31, 31, 19, 66]
        
        # Find the indices of the minimum values along the flattened dimension
        best_flattened_indices = torch.argmin(Q_flattened, dim=-1)  # Shape: [100, 31, 31, 19]

        # Convert the flattened indices back to 2D indices
        last_dim_size = self.Q.shape[-1]  # 6
        best_v_indices = best_flattened_indices // last_dim_size  # Indices for the second last dimension (11)
        best_w_indices = best_flattened_indices % last_dim_size  # Indices for the last dimension (6)
        #  [100, 31, 31, 19
        self.policy = torch.stack(
            [self.v_space[best_v_indices], self.w_space[best_w_indices]]
            ,dim=-1).cpu()
        return

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cpu")
    dtype = torch.float # must use float 32 for probability normalization work correctly

    cfg = GpiConfig(
        utils.line,
        device,
        dtype,
        idx_dtype=torch.int32,
        obstacles=torch.tensor([[-2, -2, 0.5], [1, 2, 0.5]], device=device, dtype=dtype),
        nt=utils.T,
        nx=6*5+1,
        ny=6*5+1,
        nth=2*9+1,
        nv=4+1,
        nw=2*4+1,
        num_neighbors=1,
        Q=torch.tensor([[10,1],[1,10]], device=device, dtype=dtype),
        q=2,
        R=torch.tensor([[0.05,0],[0,0.05]], device=device, dtype=dtype),
        gamma=utils.GAMMA,
        num_iters=3,
        delta=10)

    solver = GPI(cfg)