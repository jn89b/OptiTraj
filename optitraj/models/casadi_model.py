"""
This module defines the CasadiModel abstract class, which serves as a template for creating system models 
in CasADi. Users must inherit from this class and implement the methods to define the system's states, 
controls, and dynamic equations.

Inheriting from this class allows the user to easily integrate their model into optimization 
and control frameworks that rely on CasADi symbolic computation.
"""

from typing import Dict
import casadi as ca
import numpy as np
from abc import ABC, abstractmethod


class CasadiModel(ABC):
    """
    An abstract base class for defining a dynamic system model in CasADi.

    This class outlines the basic structure of a model, including defining system states, controls, 
    state space, and dynamic equations. The core function of this model, `function`, will describe 
    the dynamics of the system using CasADi symbolic expressions.

    **How to Use:**
    - Inherit from this class.
    - Implement the abstract methods to define states, controls, and the dynamic model (state space).
    - Optionally, use the provided `rk45` method for integrating system dynamics.

    Attributes:
        function (ca.Function): CasADi function describing system dynamics.
        controls (ca.MX): Symbolic variable representing control inputs.
        states (ca.MX): Symbolic variable representing system states.
        n_states (int): Number of state variables.
        n_controls (int): Number of control inputs.
        control_limits (Dict): Dictionary defining the limits of control inputs.
        state_limits (Dict): Dictionary defining the limits of states.
    """

    def __init__(self) -> None:
        self.function: ca.Function = None
        self.controls: ca.MX = None
        self.states: ca.MX = None
        self.n_states: int = None
        self.n_controls: int = None
        self.control_limits: Dict = None
        self.state_limits: Dict = None

    @abstractmethod
    def define_states(self) -> None:
        """
        Abstract method for defining the state variables of the system.
        This method should create symbolic representations of the states and set `self.states`.
        It should also define `self.n_states`, the number of states in the system.

        Example:
            self.states = ca.MX.sym('x', 4)  # For a 4-state system
            self.n_states = 4
        """
        pass

    @abstractmethod
    def define_controls(self) -> None:
        """
        Abstract method for defining the control variables of the system.
        This method should create symbolic representations of the controls and set `self.controls`.
        It should also define `self.n_controls`, the number of control inputs.

        Example:
            self.controls = ca.MX.sym('u', 2)  # For a system with 2 control inputs
            self.n_controls = 2
        """
        pass

    @abstractmethod
    def define_state_space(self) -> None:
        """
        Abstract method for defining the dynamic equations (state space) of the system.
        This method should create a CasADi function that represents the system dynamics as:
            f(x, u) -> dx/dt

        Example:
            dxdt = ca.MX([some_expression])  # Define the dynamic model equations
            self.function = ca.Function('dynamics', [self.states, self.controls], [dxdt])
        """
        pass

    def set_control_limits(self, limits: Dict) -> None:
        """
        Set the control limits for the system.

        Parameters:
            limits (Dict): A dictionary where each control input is assigned a 'min' and 'max' value.

        Example:
            limits = {
                'throttle': {'min': 0, 'max': 1},
                'steering': {'min': -1, 'max': 1}
            }
        """
        self.control_limits = limits
        if len(limits.keys()) != self.n_controls:
            raise ValueError(
                "Number of control limits do not match the number of controls.")

    def set_state_limits(self, limits: Dict) -> None:
        """
        Set the state limits for the system.

        Parameters:
            limits (Dict): A dictionary where each state variable is assigned a 'min' and 'max' value.

        Example:
            limits = {
                'position_x': {'min': -10, 'max': 10},
                'position_y': {'min': -10, 'max': 10}
            }
        """
        self.state_limits = limits
        if len(limits.keys()) != self.n_states:
            raise ValueError(
                "Number of state limits do not match the number of states.")

    def rk45(self, x: np.ndarray, u: np.ndarray, dt: float, use_numeric: bool = True) -> np.ndarray:
        """
        Perform one step of the Runge-Kutta 4th order (RK4) integration method to advance the system dynamics.

        Parameters:
            x (np.ndarray): Current state vector.
            u (np.ndarray): Current control input vector.
            dt (float): Time step for integration.
            use_numeric (bool): Whether to return the result as a numpy array (True) or as CasADi symbols (False).

        Returns:
            np.ndarray: The next state after applying the RK4 integration step (as a numpy array if `use_numeric` is True).

        Note:
            This method requires that the `self.function` attribute be defined (i.e., the dynamic equations are set).
        """
        if self.function is None:
            raise RuntimeError(
                "The function attribute is not defined. Ensure you have defined the state space dynamics.")

        # Runge-Kutta integration steps
        k1 = self.function(x, u)
        k2 = self.function(x + dt/2 * k1, u)
        k3 = self.function(x + dt/2 * k2, u)
        k4 = self.function(x + dt * k3, u)

        # Compute the next state
        next_step = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        if use_numeric:
            # Return the result as a numpy array
            next_step = np.array(next_step).flatten()
            return next_step
        else:
            # Return the result as CasADi symbolic expressions
            return next_step
