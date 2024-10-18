# OptiTraj: Optimal Trajectory Control Framework

OptiTraj is a versatile framework designed for optimal control and trajectory optimization. Built with CasADi, it provides tools for dynamic simulation, closed-loop control, and path planning, especially in aerospace contexts such as UAVs. OptiTraj allows users to define models, constraints, and optimal control problems that can be solved using Model Predictive Control (MPC) and other optimization-based techniques.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
  - [1. Defining the Model](#1-defining-the-model)
  - [2. Setting Up the MPC Problem](#2-setting-up-the-mpc-problem)
  - [3. Running a Simulation](#3-running-a-simulation)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Dynamic Models**: Includes customizable plane models, such as `Plane` and `JSBPlane`, to model UAV and aircraft dynamics.
- **Model Predictive Control (MPC)**: Integrated with CasADi to provide a robust MPC framework for trajectory optimization.
- **Closed-Loop Simulation**: Allows users to simulate control actions in real-time environments.
- **State and Control Constraints**: Easily define constraints for states (e.g., position, velocity) and controls (e.g., pitch, roll, yaw).
- **Obstacle Avoidance**: Incorporate static obstacles into the environment and compute obstacle-free trajectories.
- **JSBSim Integration**: Leverage JSBSim, an open-source flight dynamics model, for high-fidelity simulations.

## Installation
To install OptiTraj, you will need Python 3.7 or above. Follow these steps to set up the package.

### Prerequisites
- [CasADi](https://web.casadi.org/get/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

