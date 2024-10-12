"""
Want to implement a close loop control for the robot.
The close loop control will use the MPC to compute the optimal control input.
The MPC will be used to compute the optimal control input.
User can feed in their own dynamics system to put into a plant
"""
import numpy as np
import time
from typing import Tuple, Dict, Any, List, Optional, Callable
from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.utils.report import Report
from abc import ABC, abstractmethod
from optitraj.dynamics_adapter import DynamicsAdapter


class CloseLoopSim():
    def __init__(self,
                 optimizer: OptimalControlProblem,
                 x_init: np.ndarray,
                 x_final: np.ndarray,
                 u0: np.ndarray,
                 dynamics_adapter: Optional["DynamicsAdapter"] = None,
                 N: int = 100,
                 log_data: bool = True,
                 stop_criteria: Optional[Callable[[np.ndarray], bool]] = None,
                 file_name: str = '',
                 print_every: int = 10) -> None:

        self.optimizer: OptimalControlProblem = optimizer
        self.state_limits: Dict = self.optimizer.state_limits
        self.ctrl_limits: Dict = self.optimizer.ctrl_limits
        self.optimizer.init_optimization()
        self.x_init: np.ndarray = x_init
        self.x_final: np.ndarray = x_final
        self.u0: np.ndarray = u0
        self.dynamics_adapter: DynamicsAdapter = dynamics_adapter
        self.N: int = N
        self.log_data: bool = log_data
        self.control_names = self.get_control_names()
        self.stop_criteria: Optional[Callable[[
            np.ndarray], bool]] = stop_criteria
        self.ctrl_idx: int = 1  # next control index
        self.time = 0.0
        self.print_every = print_every
        # this is used to update the controller, used for the JSBSimAdapter
        self.update_controller: bool = True
        if self.log_data:
            self.report: Report = Report(self.state_limits,
                                         self.ctrl_limits, self.control_names,
                                         file_name=file_name)

    def get_control_names(self) -> List[str]:
        controls = self.optimizer.casadi_model.controls
        n_controls = self.optimizer.casadi_model.n_controls
        control_names = [str(controls[i].name()) for i in range(n_controls)]

        return control_names

    def update_x_final(self, x_final: np.ndarray) -> None:
        self.x_final = x_final

    def update_x_init(self, x_init: np.ndarray) -> None:
        self.x_init = x_init

    def run(self) -> None:

        mpc_time = self.optimizer.mpc_params.dt
        next_time = self.time + mpc_time

        for i in range(self.N):
            if i % self.print_every == 0:
                print("state: ", self.x_init)
            if self.dynamics_adapter is not None and self.update_controller:
                sol = self.optimizer.solve_and_get_solution(
                    self.x_init, self.x_final, self.u0)
                self.dynamics_adapter.set_controls(sol['states'],
                                                   sol['controls'],
                                                   self.ctrl_idx,
                                                   xF=self.x_final)
                self.update_controller = False
            else:
                sol = self.optimizer.solve_and_get_solution(
                    self.x_init, self.x_final, self.u0)

            if self.dynamics_adapter is not None:
                self.time += self.dynamics_adapter.simulator.dt
                # check if the time is greater than the mpc time
                if self.time >= next_time:
                    next_time = self.time + mpc_time
                    self.update_controller = True
            else:
                self.time += self.optimizer.mpc_params.dt

            if self.log_data:
                self.report.log_state_traj(sol['states'])
                self.report.log_control_traj(sol['controls'])
                self.report.log_current_state(self.x_init)
                self.report.log_current_control(self.u0)
                self.report.log_time(self.time)

            if self.dynamics_adapter is not None:
                self.dynamics_adapter.run()
                self.x_init = self.dynamics_adapter.get_state_information()
                self.u0 = self.dynamics_adapter.get_control_information()
            else:
                self.shift_next(sol)

            # def check criteria
            if self.stop_criteria is not None and self.stop_criteria(
                    self.x_init, self.x_final):
                print('Stopping criteria met')
                self.report.save_everything()
                break

    def shift_next(self, sol: dict) -> None:
        """
        Shift over the next initial state
        """
        states = sol['states']
        controls = sol['controls']
        idx = 1
        x_init = []
        u0 = []
        for i, state_name in enumerate(self.state_limits.keys()):
            x_init.append(states[state_name][idx])

        for i, ctrl_name in enumerate(self.ctrl_limits.keys()):
            u0.append(controls[ctrl_name][idx])

        self.x_init = np.array(x_init)
        self.u0 = np.array(u0)

        return

    def reset(self, x0: np.ndarray,
              xF: np.ndarray, u0: np.ndarray,
              file_name: str) -> None:
        self.x_init = x0
        self.x_final = xF
        self.u0 = u0
        self.optimizer.init_optimization()
        self.report = Report(self.state_limits,
                             self.ctrl_limits, self.control_names,
                             file_name)
        self.time = 0.0
        if self.dynamics_adapter is not None:
            self.dynamics_adapter.reset()
