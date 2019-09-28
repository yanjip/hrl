import random

from gym import spaces
from gym_minigrid.envs.fourrooms import FourRoomsEnv
from gym_minigrid.minigrid import Grid, Goal
from enum import IntEnum


def stochastic_step(env, action, prob=0.9):
    env.step_count += 1
    
    reward = 0
    done = False
    
    # Get the position in front of the agent
    fwd_pos = env.front_pos
    
    # Get the contents of the cell in front of the agent
    fwd_cell = env.grid.get(*fwd_pos)
    
    def rotate_left():
        env.agent_dir -= 1
        if env.agent_dir < 0:
            env.agent_dir += 4
        return reward, done
    
    def rotate_right():
        env.agent_dir = (env.agent_dir + 1) % 4
        return reward, done
    
    def forward(reward=0, done=False):
        if fwd_cell is None or fwd_cell.can_overlap():
            env.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            done = True
            reward = env._reward()
        if fwd_cell is not None and fwd_cell.type == 'lava':
            done = True
        
        return reward, done
    
    # Rotate left
    if action == env.actions.left:
        if random.random() > prob:
            reward, done = random.choice((rotate_right, forward))()
        else:
            reward, done = rotate_left()
    
    # Rotate right
    elif action == env.actions.right:
        if random.random() > prob:
            reward, done = random.choice((rotate_left, forward))()
        else:
            reward, done = rotate_right()
    
    # Move forward
    elif action == env.actions.forward:
        if random.random() > prob:
            reward, done = random.choice((rotate_left, rotate_right))()
        else:
            reward, done = forward()
    
    # Done action (not used by default)
    elif action == env.actions.done:
        pass
    
    else:
        assert False, "unknown action"
    
    if env.step_count >= env.max_steps:
        done = True
    
    obs = env.gen_obs()
    
    return obs, reward, done, {}


class FourRooms(FourRoomsEnv):
    """ Overwrites the original generator to make the hallways static """
    
    def __init__(self, agent_pos: tuple = (1, 1), goal_pos: tuple = (15, 15)):
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
    
    def _gen_grid(self, width, height):
        
        # Create the grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        
        room_w = width // 2
        room_h = height // 2
        
        # For each row of rooms
        for j in range(0, 2):
            
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h
                
                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    # pos = (xR, self._rand_int(yT + 1, yB))
                    # self.grid.set(*pos, None)
                
                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    # pos = (self._rand_int(xL + 1, xR), yB)
                    # self.grid.set(*pos, None)
        
        hallways = {
            'top'  : (9, 4),
            'left' : (3, 9),
            'right': (16, 9),
            'bot'  : (9, 14)
        }
        for hallway in hallways.values():
            self.grid.set(*hallway, None)
        
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.start_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.start_dir = self._rand_int(0,
                                            4)  # assuming random start direction
        else:
            self.place_agent()
        
        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())
        
        self.mission = 'Reach the goal'


class FourRoomsSB(FourRoomsEnv):
    """
    Overwrites the original generator to make the hallways static
    Adds compatibility with stable baselines by changing observation space from Dict to Box
    """
    
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        
        # # Pick up an object
        # pickup = 3
        # # Drop an object
        # drop = 4
        # # Toggle/activate an object
        # toggle = 5
        
        # Done completing task
        done = 3
    
    def __init__(self, agent_pos: tuple = (1, 1), goal_pos: tuple = (15, 15)):
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        # Action enumeration for this environment
        self.actions = FourRoomsSB.Actions
        
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
    
    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        
        grid, vis_mask = self.gen_obs_grid()
        
        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        
        assert hasattr(self,
                       'mission'), "environments must define a textual mission string"
        
        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image'    : image,
            'direction': self.agent_dir,
            'mission'  : self.mission
        }
        
        return image
    
    def _gen_grid(self, width, height):
        
        # Create the grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        
        room_w = width // 2
        room_h = height // 2
        
        # For each row of rooms
        for j in range(0, 2):
            
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h
                
                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    # pos = (xR, self._rand_int(yT + 1, yB))
                    # self.grid.set(*pos, None)
                
                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    # pos = (self._rand_int(xL + 1, xR), yB)
                    # self.grid.set(*pos, None)
        
        hallways = {
            'top'  : (9, 4),
            'left' : (3, 9),
            'right': (16, 9),
            'bot'  : (9, 14)
        }
        for hallway in hallways.values():
            self.grid.set(*hallway, None)
        
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.start_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.start_dir = self._rand_int(0,
                                            4)  # assuming random start direction
        else:
            self.place_agent()
        
        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())
        
        self.mission = 'Reach the goal'
