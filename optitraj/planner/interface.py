import numpy as np

from typing import List
from optitraj.planner.sparse_astar import SparseAstar, Route
from optitraj.planner.position_vector import PositionVector
from optitraj.planner.grid import FWAgent, Grid
from optitraj.planner.grid_obs import Obstacle as GridObstacle


class Planner():
    """
    Planner Class for Pathfinding in a Grid Environment using Sparse A* Algorithm.

    This class facilitates the initialization and configuration of a pathfinding 
    system using a fixed-wing agent navigating in a grid environment. It supports 
    customizing agent parameters, grid configuration, and planning routes with 
    optional obstacles.

    Attributes:
        planner_type (str): The type of planner to use. Default is "sparse_astar".
        start_position (PositionVector): The starting position of the agent.
        goal_position (PositionVector): The goal position of the agent.
        fw_agent (FWAgent): The fixed-wing agent navigating the grid.
        grid (Grid): The grid environment for the agent's navigation.

    Methods:
        init_agent(start_position, goal_position, start_theta_dg, start_psi_dg, 
                speed_ms=20.0, max_roll_dg=45.0, leg_m=5.0) -> FWAgent:
            Initializes the fixed-wing agent with its constraints and goal state.

        init_grid(grid_size, obstacles=None, use_obstacles=False, 
                random_obstacles=False) -> Grid:
            Configures the grid environment with optional obstacles.

        plan(velocity_ms, time_search_sec) -> Route:
            Plans a path using the configured planner type and grid.

        update_start_position(position, start_theta_dg, start_psi_dg) -> None:
            Updates the agent's starting position and orientation.

        update_goal_position(position) -> None:
            Updates the agent's goal position.
    """

    def __init__(self,
                 planner_type: str = "sparse_astar",):

        self.planner_type: str = planner_type
        self.start_position: PositionVector = None
        self.goal_position: PositionVector = None
        self.fw_agent: FWAgent = None
        self.grid: Grid = None

    def init_agent(self, start_position: List[float],
                   goal_position: List[float],
                   start_theta_dg: float,
                   start_psi_dg: float,
                   speed_ms: float = 20.0,
                   max_roll_dg: float = 45.0,
                   leg_m: float = 5.0) -> FWAgent:
        """
        Initializes the fixed-wing agent with its constraints and goal state.

        Args:
            start_position (List[float]): The [x, y, z] starting coordinates of the agent.
            goal_position (List[float]): The [x, y, z] goal coordinates for the agent.
            start_theta_dg (float): The initial pitch angle of the agent in degrees.
            start_psi_dg (float): The initial yaw angle of the agent in degrees.
            speed_ms (float): The speed of the agent in meters per second. Default is 20.0.
            max_roll_dg (float): The maximum roll angle of the agent in degrees. Default is 45.0.
            leg_m (float): The minimum leg length for the agent's turns in meters. Default is 5.0.

        Returns:
            FWAgent: The initialized fixed-wing agent.
        """

        self.start_position = PositionVector(start_position[0],
                                             start_position[1],
                                             start_position[2])
        self.goal_position = PositionVector(goal_position[0],
                                            goal_position[1],
                                            goal_position[2])
        # create a fixed wing agent
        r: float = speed_ms**2 / (g * np.tan(np.deg2rad(max_roll_dg)))

        fw_agent = FWAgent(
            start_position, theta_dg=start_theta_dg, psi_dg=start_psi_dg)
        fw_agent.vehicle_constraints(horizontal_min_radius_m=r,
                                     max_climb_angle_dg=10,
                                     max_psi_turn_dg=45)
        g: float = 9.81
        fw_agent.leg_m = leg_m
        fw_agent.set_goal_state(goal_position)

    def init_grid(self, grid_size: List[int],
                  obstacles: List[GridObstacle] = None,
                  use_obstacles: bool = False,
                  random_obstacles: bool = False) -> Grid:
        """
        Configures the grid environment with optional obstacles.

        Args:
            grid_size (List[int]): The [width, height] dimensions of the grid.
            obstacles (List[GridObstacle], optional): A list of obstacles to add to the grid. Defaults to None.
            use_obstacles (bool, optional): Whether to add obstacles to the grid. Defaults to False.
            random_obstacles (bool, optional): Whether to generate random obstacles. Defaults to False.

        Returns:
            Grid: The configured grid environment.
        """

        self.grid = Grid(grid_size, obstacles)

        if use_obstacles:
            if not random_obstacles:
                for obstacle in obstacles:
                    obstacle: GridObstacle
                    self.grid.insert_obstacles(obstacle)
            else:
                # generate random obstacles
                random_obstacles: List[GridObstacle]

        return self.grid

    def plan(self, velocity_ms: float, time_search_sec: float) -> Route:
        """
        Plans a path using the configured planner type and grid.

        Args:
            velocity_ms (float): The agent's velocity in meters per second.
            time_search_sec (float): The maximum search time for the planner in seconds.

        Returns:
            Route: The planned route from the start to the goal position.
        """

        if self.planner_type == "sparse_astar":
            astar = SparseAstar(grid=self.grid,
                                velocity=velocity_ms,
                                max_time_search=time_search_sec)
            path: Route = astar.search()

        return path

    def update_start_position(self, position: List[float],
                              start_theta_dg: float,
                              start_psi_dg: float) -> None:
        """
        Updates the agent's starting position and orientation.

        Args:
            position (List[float]): The [x, y, z] coordinates of the new starting position.
            start_theta_dg (float): The pitch angle of the agent in degrees.
            start_psi_dg (float): The yaw angle of the agent in degrees.

        Returns:
            None
        """

        self.start_position = PositionVector(position[0],
                                             position[1],
                                             position[2])
        self.fw_agent.set_current_state(position, start_theta_dg, start_psi_dg)

    def update_goal_position(self, position: List[float]) -> None:
        """
        Updates the agent's goal position.

        Args:
            position (List[float]): The [x, y, z] coordinates of the new goal position.

        Returns:
            None
        """

        self.goal_position = PositionVector(position[0],
                                            position[1],
                                            position[2])
        self.fw_agent.set_goal_state(position)
