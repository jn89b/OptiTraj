import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl

from typing import Dict

from aircraftsim import DataVisualizer

from optitraj.models.plane import Plane, JSBPlane
from optitraj.close_loop import CloseLoopSim
from optitraj.mpc.PlaneOptControl import PlaneOptControl
from optitraj.utils.data_container import MPCParams
from optitraj.utils.report import Report
from optitraj.dynamics_adapter import JSBSimAdapter

# check if aircraftsim is installed
try:
    # import aircraftsim
    from aircraftsim import (
        SimInterface,
        AircraftIC
    )


except ImportError:
    print('aircraftsim not installed')

GOAL_X = 125
GOAL_Y = 80
GOAL_Z = 15


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

        params = MPCParams(Q=Q, R=R, N=10, dt=0.1)
        self.mpc = PlaneOptControl(mpc_params=params,
                                   casadi_model=self.plane)

        self.closed_loop_sim = None

    def run_kinematics(self) -> None:
        x_init = np.array([0, 0, 15, 0, 0, 0, 15])
        x_final = np.array([GOAL_X, GOAL_Y, 40, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 0, 15])

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

        x_init = np.array([0, 0, 15, 0, 0, np.deg2rad(-225), 15])
        x_final = np.array([GOAL_X, GOAL_Y, 30, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 20])

        self.plane = JSBPlane()

        # let's define the limits for the states and controls
        control_limits_dict = {
            'u_psi': {'min': -np.deg2rad(180), 'max': np.deg2rad(180)},
            'u_z': {'min': 0, 'max': 50},
            'v_cmd': {'min': 15.0, 'max': 30.0},
        }

        state_limits_dict = {
            'x': {'min': -np.inf, 'max': np.inf},
            'y': {'min': -np.inf, 'max': np.inf},
            'z': {'min': 0, 'max': 50},
            'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'theta': {'min': -np.deg2rad(25), 'max': np.deg2rad(20)},
            'psi': {'min': -np.pi, 'max': np.pi},
            'v': {'min': 15, 'max': 30.0}
        }
        self.plane.set_control_limits(control_limits_dict)
        self.plane.set_state_limits(state_limits_dict)

        Q = np.array([1, 1, 1, 0, 0, 0, 0])
        # Q = np.eye(plane.n_states)
        Q = np.diag(Q)
        R = np.array([0, 0, 0])  # np.eye(self.plane.n_controls)
        R = np.diag(R)

        params = MPCParams(Q=Q, R=R, N=10, dt=0.1)
        self.mpc = PlaneOptControl(mpc_params=params,
                                   casadi_model=self.plane)

        init_cond = AircraftIC(
            x=0, y=0, z=x_init[2],
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
            N=750,
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
        # ax.plot(states['x'][-1], states['y'][-1], states['z'][-1], 'ro', label='goal')
        ax.plot(GOAL_X, GOAL_Y, 40, 'ro', label='goal')

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

        plt.show()


def run_close_loop_sim():
    test = Test()
    # test.run_kinematics()
    test.run_jsbsim()


if __name__ == '__main__':
    run_close_loop_sim()
