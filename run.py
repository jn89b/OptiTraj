import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl

from typing import Dict, List

from aircraftsim import DataVisualizer

from optitraj.models.plane import Plane, JSBPlane
from optitraj.close_loop import CloseLoopSim
from optitraj.mpc.PlaneOptControl import PlaneOptControl, Obstacle
from optitraj.utils.data_container import MPCParams
from optitraj.utils.report import Report
from optitraj.dynamics_adapter import JSBSimAdapter
from scipy.spatial import distance


try:
    # import aircraftsim
    from aircraftsim import (
        SimInterface,
        AircraftIC
    )


except ImportError:
    print('aircraftsim not installed')

GOAL_X = -125
GOAL_Y = 125
GOAL_Z = 25


def knn_obstacles(obs: np.ndarray, ego_pos: np.ndarray,
                  K: int = 3, use_2d: bool = False) -> tuple:
    """
    Find the K nearest obstacles to the ego vehicle and return the 
    obstacle positions and distances
    """
    if use_2d:
        ego_pos = ego_pos[:2]
        obs = obs[:, :2]

    nearest_indices = distance.cdist(
        [ego_pos], obs).argsort()[:, :K]
    nearest_obstacles = obs[nearest_indices]
    return nearest_obstacles[0], nearest_indices


def find_inline_obstacles(ego_unit_vector: np.ndarray, obstacles: np.ndarray,
                          ego_position: np.ndarray,
                          dot_product_threshold: float = 0.0,
                          use_2d: bool = False) -> tuple:
    '''
    compute the obstacles that are inline with the ego vehicle
    the dot product threshold is used to determine if the obstacle is inline
    with the ego vehicle
    '''
    # check size of ego_position
    if use_2d:
        if ego_position.shape[0] != 2:
            ego_position = ego_position[:2]
        obstacles = obstacles[:, :2]

    inline_obstacles = []
    dot_product_vals = []
    for i, obs in enumerate(obstacles):
        los_vector = obs - ego_position
        los_vector /= np.linalg.norm(los_vector)
        dot_product = np.dot(ego_unit_vector, los_vector)
        if dot_product >= dot_product_threshold:
            inline_obstacles.append(obstacles[i])
            dot_product_vals.append(dot_product)

    return inline_obstacles, dot_product_vals


def find_danger_zones(obstacles: np.ndarray,
                      ego_position: np.ndarray,
                      min_radius_turn: float,
                      dot_products: np.ndarray,
                      distance_buffer: float = 10.0,
                      use_2d: bool = False) -> np.ndarray:
    '''
    compute the obstacles that are inline with the ego 
    we check if they are within the minimum radius of turn with the buffer
    '''
    # check size of ego_position
    if use_2d:
        if ego_position.shape[0] != 2:
            ego_position = ego_position[:2]

    danger_zones = []
    new_dot_products = []
    for i, obs, in enumerate(obstacles):
        obs_position = obs[:2]
        obs_radius = obs[-1]
        distance_to_obstacle = np.linalg.norm(obs_position - ego_position)
        delta_r = distance_to_obstacle - \
            (min_radius_turn + obs_radius + distance_buffer)
        if delta_r <= distance_buffer:
            # compute the danger zone
            new_dot_products.append(dot_products[i])
            danger_zones.append(obs)

    return danger_zones, new_dot_products


class Test():

    def __init__(self):

        self.plane = Plane()
        Q = np.array([1, 1, 1, 0, 0, 0, 0])
        # Q = np.eye(plane.n_states)
        Q = np.diag(Q)
        R = np.eye(self.plane.n_controls)

        # let's define the limits for the states and controls
        control_limits_dict = {
            'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'u_theta': {'min': -np.deg2rad(30), 'max': np.deg2rad(30)},
            'u_psi': {'min': -np.deg2rad(180), 'max': np.deg2rad(180)},
            'v_cmd': {'min': 15.0, 'max': 30.0}
        }
        state_limits_dict = {
            'x': {'min': -np.inf, 'max': np.inf},
            'y': {'min': -np.inf, 'max': np.inf},
            'z': {'min': 0, 'max': 50},
            'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'theta': {'min': -np.deg2rad(15), 'max': np.deg2rad(15)},
            'psi': {'min': -np.pi, 'max': np.pi},
            'v': {'min': 15, 'max': 30.0}
        }
        self.plane.set_control_limits(control_limits_dict)
        self.plane.set_state_limits(state_limits_dict)

        self.params = MPCParams(Q=Q, R=R, N=10, dt=0.1)
        self.mpc = None
        self.closed_loop_sim = None

    def run_kinematics(self) -> None:
        x_init = np.array([0, 0, 15, 0, 0, 0, 15])
        x_final = np.array([GOAL_X, GOAL_Y, 40, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 0, 15])

        obstacle_list: List[Obstacle] = []
        obstacle_list.append(Obstacle(center=[50, 50, 20], radius=5))

        self.mpc = PlaneOptControl(mpc_params=self.params,
                                   casadi_model=self.plane,
                                   use_obs_avoidance=True,
                                   obs_params=obstacle_list)

        def custom_stop_criteria(state: np.ndarray,
                                 final_state: np.ndarray) -> bool:
            distance = np.linalg.norm(state[0:2] - final_state[0:2])
            if distance < 3.0:
                return True

        self.closed_loop_sim = CloseLoopSim(
            optimizer=self.mpc,
            x_init=x_init,
            x_final=x_final,
            u0=u_0,
            N=200,
            log_data=True,
            stop_criteria=custom_stop_criteria,
            file_name='simple_sim'
        )

        self.closed_loop_sim.run()
        report = self.closed_loop_sim.report
        self.plot(report, self.closed_loop_sim)

    def run_jsbsim(self):
        x_init = np.array([0, 0, 15, 0, 0, np.deg2rad(40), 15])
        x_final = np.array([GOAL_X, GOAL_Y, GOAL_Z, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 20])

        obstacle_list: List[Obstacle] = []
        obstacle_list.append(Obstacle(center=[50, 50, 20], radius=10))
        self.plane = JSBPlane(dt_val=1/60)

        # let's define the limits for the states and controls
        control_limits_dict = {
            'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'u_z': {'min': 0, 'max': 50},
            'v_cmd': {'min': 15.0, 'max': 30.0},
        }

        state_limits_dict = {
            'x': {'min': -np.inf, 'max': np.inf},
            'y': {'min': -np.inf, 'max': np.inf},
            'z': {'min': 0, 'max': 50},
            'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'theta': {'min': -np.deg2rad(25), 'max': np.deg2rad(20)},
            'psi': {'min': np.deg2rad(-360), 'max': np.deg2rad(360)},
            'v': {'min': 15, 'max': 30.0}
        }
        self.plane.set_control_limits(control_limits_dict)
        self.plane.set_state_limits(state_limits_dict)

        Q = np.array([1, 1, 1, 0, 0, 0, 0])
        Q = np.diag(Q)
        R = np.array([0, 0, 0])  # np.eye(self.plane.n_controls)
        R = np.diag(R)

        params = MPCParams(Q=Q, R=R, N=10, dt=1/60)
        self.mpc = PlaneOptControl(
            mpc_params=params,
            casadi_model=self.plane,
            use_obs_avoidance=False,
            obs_params=obstacle_list,
            robot_radius=10.0)

        init_cond = AircraftIC(
            x=x_init[0], y=x_init[1], z=x_init[2],
            roll=np.deg2rad(0),
            pitch=np.deg2rad(0),
            yaw=x_init[5],
            airspeed_m=x_init[6])

        sim = SimInterface(
            aircraft_name='x8',
            init_cond=init_cond,
            sim_freq=60,
        )

        x_init = np.array([init_cond.x,
                           init_cond.y,
                           init_cond.z,
                           init_cond.roll,
                           init_cond.pitch,
                           init_cond.yaw,
                           init_cond.airspeed_m])

        def custom_stop_criteria(state: np.ndarray,
                                 final_state: np.ndarray) -> bool:
            distance = np.linalg.norm(state[0:2] - final_state[0:2])
            if distance < 5.0:
                return True

        self.closed_loop_sim = CloseLoopSim(
            optimizer=self.mpc,
            x_init=x_init,
            x_final=x_final,
            u0=u_0,
            dynamics_adapter=JSBSimAdapter(sim),
            N=1000,
            log_data=True,
            stop_criteria=custom_stop_criteria,
            file_name='jsbsim_sim'
        )
        self.closed_loop_sim.run()

        report: Report = self.closed_loop_sim.report
        self.plot(report, self.closed_loop_sim, True)

    def plot(self, report: Report, cl_sim: CloseLoopSim,
             plt_jsbsim: bool = False) -> None:
        states = report.current_state
        # controls = report.current_control
        time = report.time_dict['time']
        traj = report.state_traj
        idx = 1
        next_state = {}

        for key in traj.keys():
            length = len(traj[key])
            next_state[key] = []
            for i in range(length):
                next_state[key].append(traj[key][i][idx])
            # next_state[key] = traj[key][idx]
            # print(next_state[key])

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(time, states['x'], label='x')
        ax[1].plot(time, states['y'], label='y')
        ax[2].plot(time, states['z'], label='z')

        ax[0].plot(time, next_state['x'], label='u_x', linestyle='--')
        ax[1].plot(time, next_state['y'], label='u_y', linestyle='--')
        ax[2].plot(time, next_state['z'], label='u_z', linestyle='--')

        for a in ax:
            a.legend()

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(time, np.rad2deg(states['phi']), label='phi')
        ax[0].plot(time, np.rad2deg(next_state['phi']),
                   label='u_phi', linestyle='--')

        ax[1].plot(time, np.rad2deg(states['theta']), label='theta')
        ax[1].plot(time, np.rad2deg(next_state['theta']),
                   label='u_theta', linestyle='--')

        ax[2].plot(time, np.rad2deg(states['psi']), label='psi')
        ax[2].plot(time, np.rad2deg(next_state['psi']),
                   label='u_psi', linestyle='--')

        for a in ax:
            a.legend()

        # plot as 3d trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states['x'], states['y'], states['z'], label='actual')
        ax.plot(GOAL_X, GOAL_Y, GOAL_Z, 'ro', label='goal')

        if cl_sim.optimizer.use_obs_avoidance:
            for obs in cl_sim.optimizer.obs_params:
                ax.plot([obs.center[0]], [obs.center[1]], [
                        obs.center[2]], 'go', label='obstacle')
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = obs.radius * np.outer(np.cos(u), np.sin(v)) + obs.center[0]
                y = obs.radius * np.outer(np.sin(u), np.sin(v)) + obs.center[1]
                z = obs.radius * \
                    np.outer(np.ones(np.size(u)), np.cos(v)) + obs.center[2]
                ax.plot_surface(x, y, z, color='b', alpha=0.5)

        pkl.dump(states, open('states.pkl', 'wb'))
        pkl.dump(next_state, open('next_state.pkl', 'wb'))
        pkl.dump(time, open('time.pkl', 'wb'))

        # plot 3d trajectory
        if plt_jsbsim:
            jsb_sim_report = cl_sim.dynamics_adapter.simulator.report
            data_vis = DataVisualizer(jsb_sim_report)
            fig, ax = data_vis.plot_3d_trajectory()
            # plot goal location
            x_final = cl_sim.x_final
            ax.scatter(x_final[0], x_final[1], x_final[2], label='goal')
            buffer = 30
            max_x = max(states['x'])
            min_x = min(states['x'])

            max_y = max(states['y'])
            min_y = min(states['y'])

            max_z = max(states['z'])
            min_z = min(states['z'])

            ax.set_xlim([min_x-buffer, max_x+buffer])
            ax.set_ylim([min_y-buffer, max_y+buffer])

            ax.set_zlim([0, 50])

            distance = np.sqrt((x_final[0] - states['x'][-1])**2 +

                               (x_final[1] - states['y'][-1])**2)

        ax.plot(states['x'], states['y'], states['z'],
                label='from mpc', color='g')
        ax.plot(GOAL_X, GOAL_Y, GOAL_Z, 'ro', label='goal')
        ax.legend()
        plt.show()

    def test_knn_obstacles(self) -> None:
        obs = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]])
        ego_pos = np.array([0, 0, 0])
        nearest_obstacles, indices = knn_obstacles(
            obs, ego_pos, K=2, use_2d=True)
        print(nearest_obstacles)
        print(indices)

    def test_avoidance(self) -> None:

        x_init = np.array([0, 0, 15, 0, 0, np.deg2rad(40), 15])
        x_final = np.array([GOAL_X, GOAL_Y, GOAL_Z, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 20])

        obstacle_list: List[Obstacle] = []
        obstacle_list.append(Obstacle(center=[50, 50, 20], radius=10))
        self.plane = JSBPlane(dt_val=1/60)

        # let's define the limits for the states and controls
        control_limits_dict = {
            'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'u_z': {'min': 0, 'max': 50},
            'v_cmd': {'min': 15.0, 'max': 30.0},
        }

        state_limits_dict = {
            'x': {'min': -np.inf, 'max': np.inf},
            'y': {'min': -np.inf, 'max': np.inf},
            'z': {'min': 0, 'max': 50},
            'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'theta': {'min': -np.deg2rad(25), 'max': np.deg2rad(20)},
            'psi': {'min': np.deg2rad(-360), 'max': np.deg2rad(360)},
            'v': {'min': 15, 'max': 30.0}
        }
        self.plane.set_control_limits(control_limits_dict)
        self.plane.set_state_limits(state_limits_dict)

        Q = np.array([1, 1, 1, 0, 0, 0, 0])
        Q = np.diag(Q)
        R = np.array([0, 0, 0])  # np.eye(self.plane.n_controls)
        R = np.diag(R)

        params = MPCParams(Q=Q, R=R, N=10, dt=1/60)
        self.mpc = PlaneOptControl(
            mpc_params=params,
            casadi_model=self.plane,
            use_obs_avoidance=False,
            obs_params=obstacle_list,
            robot_radius=10.0)

        init_cond = AircraftIC(
            x=x_init[0], y=x_init[1], z=x_init[2],
            roll=np.deg2rad(0),
            pitch=np.deg2rad(0),
            yaw=x_init[5],
            airspeed_m=x_init[6])

        sim = SimInterface(
            aircraft_name='x8',
            init_cond=init_cond,
            sim_freq=60,
        )

        x_init = np.array([init_cond.x,
                           init_cond.y,
                           init_cond.z,
                           init_cond.roll,
                           init_cond.pitch,
                           init_cond.yaw,
                           init_cond.airspeed_m])

        def custom_stop_criteria(state: np.ndarray,
                                 final_state: np.ndarray) -> bool:
            distance = np.linalg.norm(state[0:2] - final_state[0:2])
            if distance < 5.0:
                return True

        self.closed_loop_sim = CloseLoopSim(
            optimizer=self.mpc,
            x_init=x_init,
            x_final=x_final,
            u0=u_0,
            dynamics_adapter=JSBSimAdapter(sim),
            N=1000,
            log_data=True,
            stop_criteria=custom_stop_criteria,
            file_name='jsbsim_sim'
        )

        obstacles = self.mpc.obs_params

        def vectorize_obs(obstacles: List[Obstacle]) -> np.ndarray:
            obs_array = []
            for obs in obstacles:
                obs_array.append([obs.center[0], obs.center[1], obs.radius])
            return np.array(obs_array)

        obstacles = vectorize_obs(obstacles)

        for i in range(self.closed_loop_sim.N):
            solution = self.closed_loop_sim.run_single_step()
            state = solution['states']

            ego_position = state[:3]
            ego_unit_vector = state[3:6]
            inline_obstacles, dot_products = find_inline_obstacles(
                ego_unit_vector, obstacles, ego_position, use_2d=True)
            print(inline_obstacles)
            print(dot_products)

            danger_zones, new_dot_products = find_danger_zones(
                inline_obstacles, ego_position, 10.0, dot_products, use_2d=True)
            print(danger_zones)
            print(new_dot_products)


def run_close_loop_sim():
    test = Test()
    # test.run_kinematics()
    # test.run_jsbsim()
    test.test_knn_obstacles()


if __name__ == '__main__':
    run_close_loop_sim()
