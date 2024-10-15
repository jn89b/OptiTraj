"""
This algorithm implements Sparse Astar algorithm for path planning
"""
import numpy as np
import time
from queue import PriorityQueue
from typing import List, Tuple, Dict
from optitraj.planner.position_vector import PositionVector
from optitraj.planner.grid import Grid, FWAgent


def round_to_nearest_even(number: int) -> int:
    rounded_number = round(number)
    if rounded_number % 2 == 1:  # Check if the rounded number is odd
        return rounded_number + 1  # If odd, add 1 to make it even
    else:
        return rounded_number  # If even, return it as is


class Report:
    def __init__(self, path: List[List[float]], time: float) -> None:
        self.path = path
        self.time = time

    def package_path(self) -> Dict:
        """
        Returns the path as a dictionary formatted as follows:
        x: List[float]
        y: List[float]
        z: List[float]
        phi: List[float] (degrees)
        theta: List[float] (degrees)
        psi: List[float] (degrees)
        """
        path_dict = {
            "x": [],
            "y": [],
            "z": [],
            "phi": [],
            "theta": [],
            "psi": []
        }

        for i in range(len(self.path)):
            path_dict["x"].append(self.path[i][0])
            path_dict["y"].append(self.path[i][1])
            path_dict["z"].append(self.path[i][2])
            path_dict["phi"].append(self.path[i][3])
            path_dict["theta"].append(self.path[i][4])
            path_dict["psi"].append(self.path[i][5])
            path_dict["time"] = self.time

        return path_dict


class Node(object):
    """
    parent = parent of current node
    position = position of node right now it will be x,y coordinates
    g = cost from start to current to node
    h = heuristic
    f = is total cost
    """

    def __init__(self, parent: "Node",
                 position: PositionVector,
                 velocity_m: float = 15,
                 prev_psi_dg: float = 0,
                 theta_dg: float = 0,
                 psi_dg: float = 0):

        self.parent: Node = parent
        self.position: PositionVector = position  # x,y,z coordinates
        self.gravity: float = 9.81
        if parent is not None:
            self.direction_vector = np.array([self.position.x - self.parent.position.x,
                                              self.position.y - self.parent.position.y,
                                              self.position.z - self.parent.position.z])
            self.theta_dg = np.arctan2(self.direction_vector[2],
                                       np.linalg.norm(self.direction_vector[0:2]))
            self.theta_dg = np.rad2deg(self.theta_dg)
            self.psi_dg = np.arctan2(
                self.direction_vector[1], self.direction_vector[0])
            self.psi_dg = np.rad2deg(self.psi_dg)
            delta_psi_rad = np.deg2rad(self.psi_dg - prev_psi_dg)
            phi_rad = np.arctan((delta_psi_rad*velocity_m)/self.gravity)
            self.phi_dg = np.rad2deg(phi_rad)

        else:
            self.direction_vector = np.array([self.position.x,
                                              self.position.y,
                                              self.position.z])
            self.theta_dg = theta_dg
            self.psi_dg = psi_dg
            delta_psi_rad = np.deg2rad(self.psi_dg - prev_psi_dg)
            phi_rad = np.arctan((delta_psi_rad*velocity_m)/9.81)
            self.phi_dg = np.rad2deg(phi_rad)

        self.g: float = 0.0
        self.h: float = 0.0
        self.f: float = 0.0
        self.total_distance: float = 0.0
        self.total_time: float = 0.0

    def get_direction_vector(self) -> np.array:
        return self.direction_vector

    def __lt__(self, other):
        return self.f < other.f

    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position
    # Print node

    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))


class SparseAstar():
    def __init__(self, grid: Grid,
                 velocity: float = 15,
                 max_time_search: float = 10.0) -> None:
        self.open_set: PriorityQueue = PriorityQueue()
        self.closed_set: Dict = {}

        self.grid: Grid = grid
        self.agent: FWAgent = grid.agent
        self.start_node: Node = None
        self.goal_node: Node = None
        self.velocity: float = velocity
        self.max_time_search: float = max_time_search

    def clear_sets(self) -> None:
        self.open_set: PriorityQueue = PriorityQueue()
        self.closed_set: Dict = {}

    def init_nodes(self) -> None:
        # Snap the start and end position to the grid
        direction = self.agent.position.vec - self.agent.goal_position.vec
        direction_vector = PositionVector(
            direction[0], direction[1], direction[2])
        direction_vector.update_position(
            direction[0], direction[1], direction[2])
        # rounded_start_position = self.grid.map_position_to_grid(
        #     self.agent.position, direction_vector)

        self.start_node = Node(None, self.agent.position,
                               self.velocity, 0,
                               self.agent.theta_dg, self.agent.psi_dg)
        self.start_node.g = self.start_node.h = self.start_node.f = 0
        self.open_set.put((self.start_node.f, self.start_node))

        # self.goal_node = Node(None, rounded_goal_position)
        self.goal_node = Node(None, self.agent.goal_position,
                              self.velocity, 0,
                              self.agent.theta_dg, self.agent.psi_dg)
        self.goal_node.g = self.goal_node.h = self.goal_node.f = 0

    def is_valid_position(self, position: PositionVector) -> bool:
        """Checks if position is valid based on grid constraints"""
        if self.grid.is_out_bounds(position):
            return False
        if self.grid.is_in_obstacle(position):
            return False

        return True

    def get_legal_moves(self, current_node: Node, psi_dg: float) -> list:
        """Get legal moves based on agent constraints"""
        moves = self.agent.get_moves(current_node.position, psi_dg)
        legal_moves = []

        # need to scale moves based on grid size
        for move in moves:
            scaled_move = [move[0], move[1], move[2]]
            scaled_position = PositionVector(move[0], move[1], move[2])
            scaled_position.update_position(
                scaled_move[0], scaled_move[1], scaled_move[2])
            if self.is_valid_position(scaled_position):
                legal_moves.append(scaled_position)

        return legal_moves

    def compute_cost(self) -> float:
        """
        pass
        """

    def compute_distance(self, node1: Node, node2: Node) -> float:
        """
        Compute distance between two nodes
        """
        cost = np.linalg.norm(node1.position.vec - node2.position.vec)
        return cost

    def get_rcs_key(self, roll: int, pitch: int, yaw: int) -> str:
        """returns the rcs key based on roll pitch yaw"""
        return f"{roll}_{pitch}_{yaw}"

    def get_key(self, azimith_dg: int, elevation_dg: int) -> str:
        """returns the rcs key based on roll pitch yaw"""
        return f"{azimith_dg}_{elevation_dg}"

    def return_path(self, current_node) -> Dict[str, List[float]]:
        path = []
        current = current_node

        while current is not None:
            states = [current.position.x,
                      current.position.y, current.position.z]
            states.append(current.phi_dg)
            states.append(current.theta_dg)
            states.append(current.psi_dg)
            path.append(states)
            current = current.parent
        # Return reversed path as we need to show from start to end path
        path = path[::-1]

        waypoints = []
        for points in path:
            waypoints.append(points)

        report = Report(waypoints, current_node.total_time)

        return report.package_path()

    def search(self) -> Dict[str, List[float]]:

        max_iterations = 10000
        iterations = 0

        start_time = time.time()

        while (not self.open_set.empty() and iterations < max_iterations):

            iterations += 1
            cost, current_node = self.open_set.get()

            self.closed_set[str(
                list(current_node.position.vec))] = current_node

            current_time = time.time() - start_time

            if current_time > self.max_time_search:
                print("reached time limit", current_time)
                return self.return_path(current_node)

            if current_node.position == self.goal_node.position:
                print("time", current_time)
                print("found goal", current_node.position)
                return self.return_path(current_node)

            if iterations >= max_iterations:
                print("iterations", iterations)
                return self.return_path(current_node)
                # break

            if self.compute_distance(current_node, self.goal_node) < self.agent.leg_m:
                print("time", current_time)
                print("found goal", current_node.position)
                return self.return_path(current_node)

            expanded_moves = self.get_legal_moves(
                current_node, current_node.psi_dg)

            if not expanded_moves:
                continue

            for move in expanded_moves:
                if str(list(move.vec)) in self.closed_set:
                    continue

                if move == current_node.position:
                    continue

                neighbor: Node = Node(current_node, move, self.velocity,
                                      current_node.psi_dg)

                # TODO: add method to do cost function
                neighbor.g = current_node.g + 1
                neighbor.h = (self.compute_distance(neighbor, self.goal_node))
                neighbor.f = neighbor.g + neighbor.h
                self.open_set.put((neighbor.f, neighbor))

        return self.return_path(current_node)
