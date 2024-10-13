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

    def set_controls(self, x: Dict, u: dict, idx: int,
                     xF: np.ndarray) -> None:
        """
        This will set the controls for the JSBSimAdapter class

        """
        idx = -1
        x_ref_m = x['x'][idx]
        y_ref_m = x['y'][idx]
        aircraft_state = self.get_state_information()
        # FIX THE COORDINATE TRANSFORMATION
        dx = x_ref_m - aircraft_state[0]
        dy = y_ref_m - aircraft_state[1]

        # for some reason the reference for height doesn't work well
        # so we're going to set z height based on the goal location
        los = np.arctan2(dy, dx)
        los = np.pi/2 - los
        vel_cmd = u['v_cmd'][idx]
        self.current_control = HighControlInputs(
            ctrl_idx=1,
            alt_ref_m=xF[2],
            heading_ref_deg=np.rad2deg(los),
            vel_cmd=vel_cmd
        )

    def run(self) -> None:
        self.simulator.step(self.current_control)

    def get_state_information(self) -> np.ndarray:
        """
        """
        aircraft_state: AircraftState = self.simulator.get_states()
        # wrap aircraft yaw from -pi to pi
        transformed_yaw = np.pi/2 - aircraft_state.yaw
        if transformed_yaw > np.pi:
            transformed_yaw -= 2*np.pi
        elif transformed_yaw < -np.pi:
            transformed_yaw += 2*np.pi

        return np.array([
            aircraft_state.x,
            aircraft_state.y,
            aircraft_state.z,
            aircraft_state.roll,
            aircraft_state.pitch,
            transformed_yaw,
            aircraft_state.airspeed
        ])

    def get_control_information(self) -> np.ndarray:
        aircraft_state: AircraftState = self.simulator.get_states()
        p_q_r = self.simulator.sim.get_rates()
        p_q_r[2] = np.pi/2 - p_q_r[2]
        current_heading = aircraft_state.yaw

        return np.array([
            current_heading,
            aircraft_state.z,
            aircraft_state.airspeed,
        ])

    def reset(self) -> None:
        pass
