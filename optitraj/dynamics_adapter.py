from typing import Tuple, Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from aircraftsim import SimInterface, AircraftState, HighControlInputs
import numpy as np


class DynamicsAdapter():
    """
    The DynamicsAdapter class is an abstract class that defines the interface
    for the dynamics system to be used in the MPC. The user must implement the
    methods in this class to interface with the MPC. Refer to the JSBSimAdapter
    class in the dynamics_adapter module for an example.
    """

    def __init__(self,
                 simulator: Any) -> None:
        self.simulator: Any = simulator

    @abstractmethod
    def get_time_step(self) -> float:
        """
        Get the time step of the dynamics system
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the dynamics system
        """
        pass

    @abstractmethod
    def set_controls(self, x: Dict, u: Dict, ctrl_idx: int) -> None:
        """
        Set the controls for the dynamics system
        so that it can interface with the Simulator
        """
        pass

    @abstractmethod
    def get_state_information(self) -> Dict:
        """
        Method used to get information about the dynamics system
        to use for the MPC
        """
        pass

    @abstractmethod
    def get_control_information(self) -> Dict:
        """
        Method used to get information about the dynamics system
        to use for the MPC
        """

    @abstractmethod
    def run(self, **kwargs) -> None:
        """
        Run the dynamics system
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the dynamics system
        """
        pass


class JSBSimAdapter(DynamicsAdapter):
    def __init__(self, simulator: SimInterface) -> None:
        super().__init__(simulator)
        self.simulator: SimInterface = simulator
        self.current_control: HighControlInputs = None

    def initialize(self) -> None:
        self.simulator.initialize()

    def wrap_yaw(self, yaw_rad: float) -> float:
        """
        Used to wrap the yaw from -pi to pi
        """
        if yaw_rad > np.pi:
            yaw_rad -= 2*np.pi
        elif yaw_rad < -np.pi:
            yaw_rad += 2*np.pi
        return yaw_rad

    def get_time_step(self):
        return self.simulator.dt

    def set_controls(self, x: Dict, u: dict, idx: int,
                     xF: np.ndarray) -> None:
        """
        This will set the controls for the JSBSimAdapter class

        """
        idx = 1
        psi_ref = np.pi/2 - x['psi'][idx]
        theta_ref = x['theta'][idx]
        vel_cmd = u['v_cmd'][idx]
        phi_cmd = u['u_phi'][idx]
        # z_cmd = u['u_z'][idx]
        z_ref_m = x['z'][idx]

        self.current_control = HighControlInputs(
            ctrl_idx=0,
            pitch=theta_ref,
            alt_ref_m=z_ref_m,
            roll=phi_cmd,
            yaw=psi_ref,
            vel_cmd=vel_cmd
        )

    def run(self) -> None:
        self.simulator.step(self.current_control)

    def get_state_information(self) -> np.ndarray:
        """
        """
        aircraft_state: AircraftState = self.simulator.get_states()
        # wrap aircraft yaw from -pi to pi
        # transformed_yaw = np.pi/2 - aircraft_state.yaw
        transformed_yaw = aircraft_state.yaw
        # wrap yaw between 0 to 2pi
        if transformed_yaw > np.pi:
            transformed_yaw -= 2*np.pi
        elif transformed_yaw < -np.pi:
            transformed_yaw += 2*np.pi

        # need to reverse the x and y coordinates
        return np.array([
            aircraft_state.y,
            aircraft_state.x,
            aircraft_state.z,
            aircraft_state.roll,
            aircraft_state.pitch,
            aircraft_state.yaw,
            aircraft_state.airspeed
        ])

    def get_control_information(self) -> np.ndarray:
        aircraft_state: AircraftState = self.simulator.get_states()
        p_q_r = self.simulator.sim.get_rates()
        p_q_r[2] = np.pi/2 - p_q_r[2]
        transformed_yaw = np.pi/2 - aircraft_state.yaw
        transformed_yaw = np.pi/2 - aircraft_state.yaw
        transformed_yaw = aircraft_state.yaw
        if transformed_yaw > np.pi:
            transformed_yaw -= 2*np.pi
        elif transformed_yaw < -np.pi:
            transformed_yaw += 2*np.pi

        return np.array([
            aircraft_state.roll,
            aircraft_state.z,
            aircraft_state.airspeed,
        ])

    def reset(self) -> None:
        pass
