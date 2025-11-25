import numpy as np

class SubgoalMapper:
    """
    Convert high-level action (0–7) into local subgoal coordinates.
    Ensures:
        - direction is correct
        - distance is fixed
        - subgoal lies on navigable surface (checked via pathfinder)
    """

    # 8 directions in X-Z plane
    DIRECTIONS = {
        0: np.array([0, 1]),    # North (Z+)
        1: np.array([1, 1]),    # NE
        2: np.array([1, 0]),    # East (X+)
        3: np.array([1, -1]),   # SE
        4: np.array([0, -1]),   # South
        5: np.array([-1, -1]),  # SW
        6: np.array([-1, 0]),   # West
        7: np.array([-1, 1]),   # NW
    }

    def __init__(self, pathfinder, step_size=5.0):
        """
        Args:
            pathfinder: Habitat pathfinder object to validate navigable points.
            step_size: distance of each subgoal.
        """
        self.pathfinder = pathfinder
        self.step_size = step_size

    def get_subgoal(self, agent_pos, action):
        """
        Convert high-level action → subgoal coordinate.

        Params:
            agent_pos: np.array([x, y, z])
            action: int 0–7

        Returns:
            subgoal_pos: np.array([x, y, z])
        """

        # get direction vector (X, Z)
        vec = self.DIRECTIONS[action].astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # normalize

        # target (X+dx, Z+dz)
        dx, dz = vec[0] * self.step_size, vec[1] * self.step_size

        raw_goal = np.array([
            agent_pos[0] + dx,
            agent_pos[1],     # keep same Y height
            agent_pos[2] + dz,
        ], dtype=np.float32)

        # if point is not navigable, project onto navmesh
        if not self.pathfinder.is_navigable(raw_goal):
            corrected = self.pathfinder.snap_point(raw_goal)
            return corrected

        return raw_goal
