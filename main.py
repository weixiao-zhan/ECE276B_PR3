from time import time
import numpy as np
import torch
import utils
import cec, gpi
import tqdm

def main(which_solver):
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    # Params
    traj = utils.circle 
    traj = utils.line
    traj = utils.lissajous
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state

    if which_solver == "cec":
        solver = cec.CEC(
            traj,
            Q = cec.casadi.MX(np.array([[2,0],[0,2]])),
            q = 2,
            R = 0.05*cec.casadi.MX(np.eye(2)),
            r = 0.05*np.eye(2),
            look_ahead_steps=5
        )
    elif which_solver == "gpi":
        device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cpu")
        print("using", device)
        dtype = torch.float32
        cfg = gpi.GpiConfig(
            traj,
            device,
            dtype,
            torch.int32,
            obstacles=torch.tensor(obstacles, device=device, dtype=dtype),
            nt=utils.T,
            nx=6*5+1,
            ny=6*5+1,
            nth=2*5+1,
            nv=5+1,
            nw=2*5+1,
            num_neighbors=1,
            Q=torch.tensor([[1,0],[0,1]], device=device, dtype=dtype),
            q=1,
            R=0.05*torch.tensor([[1,0],[0,1]], device=device, dtype=dtype),
            gamma=utils.GAMMA,
            num_iters=100,
            delta=5)
        solver = gpi.GPI(cfg)
    else:
        raise NotImplemented
    # Main loop
    start_iter = 0
    cur_state = traj(start_iter)
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init]) # 
    bar = tqdm.tqdm(range(start_iter, int(utils.sim_time / utils.time_step)))
    for cur_iter in bar:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = utils.simple_controller(cur_state, traj(cur_iter))
        control = solver(cur_iter, cur_state)
        ###############################################################

        # Apply control input
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        # print(cur_err, error_trans, error_rot)
        # print("======================")
        bar.set_postfix({
            "[v,w]": control,
            'cur_err': cur_err,
            'error_trans': error_trans,
            'error_rot': error_rot,
        })

        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)

if __name__ == "__main__":
    main("gpi")

