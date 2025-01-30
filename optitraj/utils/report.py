
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from optitraj.utils.data_container import MPCParams
from typing import Dict, List


class Report():
    def __init__(self,
                 state_limits: dict,
                 control_limits: dict,
                 control_names: List[str],
                 file_name: str = None) -> None:
        self.state_limits: dict = state_limits
        self.control_limits: dict = control_limits
        self.control_names = control_names

        self.state_traj: Dict = self.create_dict(self.state_limits)
        self.ctrl_traj: Dict = self.create_dict(self.control_limits)
        self.time_dict = self.create_time_dict()

        self.current_state: Dict = self.create_dict(self.state_limits)
        self.current_control: Dict = self.create_dict(self.control_limits)
        self.visualizer = None
        self.file_name = file_name

    def create_time_dict(self) -> Dict:
        time_dict = {}
        time_dict['time'] = []

        return time_dict

    def create_dict(self, limit: Dict) -> Dict:
        """
        This stores the mpc trajectories 
        for each of the states where:
        key: state name (str)
        value: list of the state trajectory at each time step
        """
        state_trajectories = {}
        for state in limit.keys():
            state_trajectories[state] = []

        return state_trajectories

    def create_ctrl_dict(self) -> Dict:
        ctrl_dict = {}
        for ctrl in self.control_names:
            ctrl_dict[ctrl] = []

        return ctrl_dict

    def log(self, sol: Dict, dict_traj: Dict) -> None:
        """
        Log the current state and control
        """
        for key in sol.keys():
            dict_traj[key].append(sol[key])

    def log_state_traj(self, state_sol: Dict) -> None:
        """
        Stash the trajectories and
        """
        self.log(state_sol, self.state_traj)

    def log_control_traj(self, ctrl_sol: Dict) -> None:
        """
        Stash the control trajectories
        """
        self.log(ctrl_sol, self.ctrl_traj)

    def log_current_state(self, x0: np.ndarray) -> None:
        """
        Log the current state
        """
        for i, state_name in enumerate(self.state_limits.keys()):
            self.current_state[state_name].append(x0[i])

    def log_current_control(self, ctrl_sol: Dict) -> None:
        """
        Log the current control
        """
        for i, ctrl_name in enumerate(self.control_names):
            self.current_control[ctrl_name].append(ctrl_sol[i])

    def log_time(self, current_time: float) -> None:
        self.time_dict['time'].append(current_time)

    def save_everything(self) -> None:
        """
        Save the state and control trajectories
        Future to do 
        """
        # save as json file
        # convert to list
        for key in self.state_traj.keys():
            self.state_traj[key] = np.array(self.state_traj[key]).tolist()

        for key in self.ctrl_traj.keys():
            self.ctrl_traj[key] = np.array(self.ctrl_traj[key]).tolist()

        data = {
            'state_traj': self.state_traj,
            'ctrl_traj': self.ctrl_traj
        }

        with open(self.file_name+'.json', 'w') as f:
            json.dump(data, f, indent=4)
