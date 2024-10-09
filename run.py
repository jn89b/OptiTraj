import numpy as np
import matplotlib.pyplot as plt
import math 

from typing import Dict

from aircraftsim import DataVisualizer

from optitraj.models.plane import Plane
from optitraj.close_loop import CloseLoopSim, DynamicsAdapter
from optitraj.mpc.PlaneOptControl import PlaneOptControl
from optitraj.utils.data_container import MPCParams
from optitraj.utils.report import Report


#check if aircraftsim is installed
try:
    # import aircraftsim
    
    from aircraftsim import (
        SimInterface
    )    

    from aircraftsim import (
        AircraftStateLimits,
        HighLevelControlLimits,
        HighControlInputs,
        AircraftIC,
        AircraftState,
    )

except ImportError:
    print('aircraftsim not installed')


GOAL_X = -150
GOAL_Y = 65

class JSBSimAdapter(DynamicsAdapter):
    def __init__(self, simulator: SimInterface) -> None:
        super().__init__(simulator)
        self.simulator:SimInterface = simulator
        self.current_control:HighControlInputs = None
        
    def initialize(self) -> None:
        self.simulator.initialize()
        
    def set_controls(self, x:Dict, u:dict, idx:int) -> None:
        """
        For this u we will have to convert it to the high level control
        which is  alt_ref_m, heading_ref_deg, vel_cmd
        
        JSBSIM's heading controller is oriented NED 
        Where 0 degrees is North
        90 degrees is East
        
        Since our trajectory planner considers 
        everything in ENU convention we're going to 
        need to transform the information 
        """

        
        x_ref_m = x['x'][-1]
        y_ref_m = x['y'][-1]
        # roll_ref = x['phi'][idx]
        # pitch_ref = x['theta'][idx]
        print("xref and yref", x_ref_m, y_ref_m)
        aircraft_state = self.get_state_information()
        print(f"x: {aircraft_state[0]}, y: {aircraft_state[1]}, \
            z: {aircraft_state[2]}, \
                yaw: {np.rad2deg(aircraft_state[5])}, \
                    airspeed: {aircraft_state[6]}")
        
        #FIX THE COORDINATE TRANSFORMATION
        # dy = GOAL_X - aircraft_state.x
        # dx = GOAL_Y   - aircraft_state.y
        dx = x_ref_m - aircraft_state[0]
        dy = y_ref_m - aircraft_state[1]
        
        dz = -x['z'][idx]
            
        los = np.arctan2(dy,dx)
        # los = np.pi/2 - los
        los_dg = np.rad2deg(los) 
        print(f"los: {np.rad2deg(los)}")
        
        vel_cmd = u['v_cmd'][idx]
        self.current_control = HighControlInputs(
            ctrl_idx=1,
            alt_ref_m=50,
            heading_ref_deg=los_dg,
            vel_cmd=vel_cmd
        )
        
    def run(self) -> None:
        self.simulator.step(self.current_control)
        
    def get_state_information(self) -> np.ndarray:
        """
        """
        aircraft_state:AircraftState = self.simulator.get_states()
        current_ctrl = self.current_control
                
        transformed_yaw = np.pi/2 - aircraft_state.yaw
        return np.array([
            aircraft_state.x,
            aircraft_state.y,
            aircraft_state.z,
            aircraft_state.roll,
            aircraft_state.pitch,
            aircraft_state.yaw,
            aircraft_state.airspeed
        ])

    def get_control_information(self) -> np.ndarray:
        aircraft_state:AircraftState = self.simulator.get_states()
        p_q_r = self.simulator.sim.get_rates()
        p_q_r[2] = np.pi/2 - p_q_r[2]
        
        transformed_yaw = np.pi/2 - aircraft_state.yaw
        
        return np.array([
            aircraft_state.roll,
            aircraft_state.pitch,
            aircraft_state.yaw,
            self.current_control.vel_cmd
        ])
        
class Test():
    
    def __init__(self):
        
        self.plane = Plane()
        Q = np.array([1, 1, 1, 0, 0, 0, 0])
        # Q = np.eye(plane.n_states)
        Q = np.diag(Q)
        R = np.eye(self.plane.n_controls)
            
        #let's define the limits for the states and controls
        control_limits_dict = {
            'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
            'u_theta': {'min': -np.deg2rad(30), 'max': np.deg2rad(30)},
            'u_psi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
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
        
        params = MPCParams(Q=Q, R=R, N=10, dt=0.1)
        self.mpc = PlaneOptControl(mpc_params=params, 
                                casadi_model=self.plane,
                                state_limits=state_limits_dict,
                                ctrl_limits=control_limits_dict)
        
        self.closed_loop_sim = None
        
    def run_kinematics(self):
        
        x_init = np.array([0, 0, 15, 0, 0, 0, 15])
        x_final = np.array([GOAL_X, GOAL_Y, 40, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 0, 15])
        
        
        def custom_stop_criteria(state:np.ndarray, 
                                 final_state:np.ndarray) -> bool:
            distance = np.linalg.norm(state[0:2] - final_state[0:2])
            if distance < 3.0:
                return True
        
        self.closed_loop_sim = CloseLoopSim(
            optimizer=self.mpc,
            x_init=x_init,
            x_final= x_final,
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
        
        x_init = np.array([0, 0, 15, 0, 0, 0, 15])
        x_final = np.array([GOAL_X, GOAL_Y, 30, 0, 0, 0, 30])
        u_0 = np.array([0, 0, 0, 15])
        
        state_limits = AircraftStateLimits(
            x_bounds=[-100,100],
            y_bounds=[-100,100],
            z_bounds=[-100,100],
            roll_bounds=[-np.deg2rad(45), np.deg2rad(45)],
            pitch_bounds=[-np.deg2rad(45), np.deg2rad(45)],
            yaw_bounds = [-np.deg2rad(180), np.deg2rad(180)],
            airspeed_bounds=[15, 30])

        hl_ctrl_limits = HighLevelControlLimits(
            roll_rate=[-np.deg2rad(45), np.deg2rad(45)],
            pitch_rate=[-np.deg2rad(45), np.deg2rad(45)],
            yaw_rate=[-np.deg2rad(45), np.deg2rad(45)],
            vel_cmd=[15,30])

        init_cond = AircraftIC(
            x=0, y=0, z= 30, 
            roll=np.deg2rad(0),
            pitch=np.deg2rad(0),
            yaw=np.deg2rad(0),
            airspeed_m=20)

        sim = SimInterface(
            aircraft_name='x8',
            init_cond=init_cond,
            high_control_lim=hl_ctrl_limits,
            state_lim=state_limits,
            sim_freq=60,
        )

        x_init = np.array([init_cond.x, 
                           init_cond.y, 
                           init_cond.z, 
                           init_cond.roll, 
                           init_cond.pitch, 
                           init_cond.yaw, 
                           init_cond.airspeed_m])
        
        self.closed_loop_sim = CloseLoopSim(
            optimizer=self.mpc,
            x_init=x_init,
            x_final= x_final,
            u0=np.array([0, 0, 0, 15]),
            dynamics_adapter=JSBSimAdapter(sim),
            N=750,
            log_data=True,
            stop_criteria=None,
            file_name='jsbsim_sim'
)
        self.closed_loop_sim.run()
        
        report:Report = self.closed_loop_sim.report
        self.plot(report, self.closed_loop_sim, True)

        
    def plot(self, report:Report, cl_sim:CloseLoopSim,
             plt_jsbsim:bool=False) -> None:
        states = report.current_state
        controls = report.current_control
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
            
        fig ,ax = plt.subplots(3,1, figsize=(10,10))
        ax[0].plot(time, states['x'], label='x')
        ax[1].plot(time, states['y'], label='y')
        ax[2].plot(time, states['z'], label='z')
        
        ax[0].plot(time, next_state['x'], label='u_x', linestyle='--')
        ax[1].plot(time, next_state['y'], label='u_y', linestyle='--')
        ax[2].plot(time, next_state['z'], label='u_z', linestyle='--')
        
        for a in ax:
            a.legend()
        
        fig, ax = plt.subplots(3,1, figsize=(10,10))
        ax[0].plot(time, np.rad2deg(states['phi']), label='phi')
        ax[0].plot(time, np.rad2deg(next_state['phi']), label='u_phi', linestyle='--')
        
        ax[1].plot(time, np.rad2deg(states['theta']), label='theta')
        ax[1].plot(time, np.rad2deg(next_state['theta']), label='u_theta', linestyle='--')
        
        ax[2].plot(time, np.rad2deg(states['psi']), label='psi')
        ax[2].plot(time, np.rad2deg(next_state['psi']), label='u_psi', linestyle='--')
        
        for a in ax:
            a.legend()
            
        # plot as 3d trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states['x'], states['y'], states['z'], label='actual')
        #ax.plot(states['x'][-1], states['y'][-1], states['z'][-1], 'ro', label='goal')
        ax.plot(GOAL_X, GOAL_Y, 40, 'ro', label='goal')
               
        import pickle as pkl
        pkl.dump(states, open('states.pkl', 'wb'))
        pkl.dump(next_state, open('next_state.pkl', 'wb'))
        pkl.dump(time, open('time.pkl', 'wb'))
        
               
        #plot 3d trajectory
        if plt_jsbsim:
            jsb_sim_report = cl_sim.dynamics_adapter.simulator.report
            data_vis = DataVisualizer(jsb_sim_report)
            fig, ax = data_vis.plot_3d_trajectory()
            #plot goal location
            x_final = cl_sim.x_final
            ax.scatter(x_final[0], x_final[1], x_final[2], label='goal')
            buffer = 30
            max_x = max(states['x'])
            min_x = min(states['x'])
            max_y = max(states['y'])
            min_y = min(states['y'])
            max_z = max(states['z'])
            min_z = min(states['z'])
            ax.set_xlim([min_x-buffer,max_x+buffer])
            ax.set_ylim([min_y-buffer,max_y+buffer])
            ax.set_zlim([0,50])
            
            distance = np.sqrt((x_final[0] - states['x'][-1])**2 + \
                (x_final[1] - states['y'][-1])**2)

    
        plt.show()
        
def run_close_loop_sim():
    test = Test()
    #test.run_kinematics()
    test.run_jsbsim()

if __name__ == '__main__':
    run_close_loop_sim()
    