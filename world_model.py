import numpy as np
from utils import get_config

class WorldModel:
    def __init__(self, blueprint, cfg_file, simulated=False):
        self._cfg      = (lambda *args: get_config(cfg_file, *args))
        self._offset_x = self._cfg('arena', 'offset', 'x')
        self._offset_y = self._cfg('arena', 'offset', 'y')
        self._offset_z = self._cfg('arena', 'offset', 'z')
        self._arena_width  = self._cfg('arena', 'width')
        self._arena_height = self._cfg('arena', 'height')
        self._arena_length = self._cfg('arena', 'length')
        self._reward_weight = self._cfg('training', 'reward_weight')
        self._cfg_file = cfg_file
        self._simulated = simulated
        self._bp       = blueprint
        self._str_type = '<U{}'.format(max(len(s) for s in self._cfg('inputs')))
        self._rot_bp   = self._bp
        self._old_num_complete    = 0
        self._old_num_incomplete  = 0
        self._old_num_superfluous = 0
        self._attacked_floor      = False
        if not simulated:
            # Wait for update() from Minecraft
            self._world = None
        else:
            # Build world with random agent start position
            self._world = np.full(
                (self._arena_length,
                 self._arena_height,
                 self._arena_width),
                fill_value='air', dtype=self._str_type)
            self._world[
                np.random.randint(self._arena_length),
                0,
                np.random.randint(self._arena_width)] = 'agent'

    def copy(self):
        '''Makes a deep-copy of the world model.'''
        output = WorldModel(self._bp, self._cfg_file, self._simulated)
        output._cfg_file              = self._cfg_file
        output._simulated             = self._simulated
        output._arena_height          = self._arena_height
        output._arena_length          = self._arena_length
        output._arena_width           = self._arena_width
        output._attacked_floor        = self._attacked_floor
        output._bp                    = self._bp.copy()
        output._cfg                   = self._cfg
        output._offset_x              = self._offset_x
        output._offset_y              = self._offset_y
        output._offset_z              = self._offset_z
        output._old_num_complete      = self._old_num_complete
        output._old_num_incomplete    = self._old_num_incomplete
        output._old_num_superfluous   = self._old_num_superfluous
        output._reward_weight         = self._reward_weight
        output._rot_bp                = self._rot_bp.copy()
        output._str_type              = self._str_type
        output._world                 = self._world.copy()
        return output

    def update(self, raw_obs):
        '''Used when hooked up to Minecraft.'''
        if self._world is not None:
            self._old_num_complete    = self.num_complete()
            self._old_num_incomplete  = self.num_incomplete()
            self._old_num_superfluous = self.num_superfluous()

        raw_world = np.array( raw_obs["world_grid"], dtype=self._str_type )
        extended_world = np.transpose(np.reshape(raw_world, (self._arena_height+1, self._arena_width, self._arena_length)), (2, 0, 1))
        world = extended_world[:,1:,:]
        agent_yaw = raw_obs['Yaw']
        agent_x = int(raw_obs['XPos'] - self._offset_x)
        agent_y = int(raw_obs['YPos'] - self._offset_y)
        agent_z = int(raw_obs['ZPos'] - self._offset_z)
        if (0 <= agent_x < world.shape[0] and 
            0 <= agent_y < world.shape[1] and 
            0 <= agent_z < world.shape[2] ):
            world[agent_x, agent_y, agent_z] = 'agent'
        # Rotate world and blueprint to be agent-facing
        self._attacked_floor = (extended_world[:,0,:] == 'air').any()
        self._world  = np.rot90(world,    k=-int(np.round(agent_yaw/90)), axes=(0,2))
        self._rot_bp = np.rot90(self._bp, k=-int(np.round(agent_yaw/90)), axes=(0,2))

    def simulate(self, action):
        '''Used instead of connecting to Malmo, for efficient training.'''
        self._old_num_complete    = self.num_complete()
        self._old_num_incomplete  = self.num_incomplete()
        self._old_num_superfluous = self.num_superfluous()
        
        if action == "jumpmove 1":
            # Find agent:
            agent_pos = tuple(self.agent_position())
            if agent_pos[2] >= self._world.shape[2] - 1:
                # Agent is at the edge of the world, facing out. Assume this makes agent jump out of world
                self._world[agent_pos] = 'air'
                return
            # Determine non-air blocks in front of agent
            #   agent position + 1 in z dir, column from 2 above agent (for head clearance) down
            in_front = (self._world[agent_pos[0], :agent_pos[1]+3, agent_pos[2]+1] != 'air')
            if (not in_front[-2:].any() or
                (agent_pos[2] == self._world.shape[1] - 2 and not in_front[-1]) or
                (agent_pos[2] == self._world.shape[1] - 1)):
                # Clearance to jump; compute where we land, as 1 above top-most non-air cell:
                new_agent_pos = (agent_pos[0], (np.where(in_front)[0][-1]+1 if in_front.any() else 0), agent_pos[2]+1)
                self._world[agent_pos]     = 'air'
                # If we're about to jump out of the arena, don't write agent cell
                if new_agent_pos[1] < self._world.shape[1]:
                    self._world[new_agent_pos] = 'agent'
            # Else, no room to jump. Action is a no-op
        elif action == "turn 1":
            self._world  = np.rot90(self._world,  k=-1, axes=(0,2))
            self._rot_bp = np.rot90(self._rot_bp, k=-1, axes=(0,2))
        elif action == "turn -1":
            self._world  = np.rot90(self._world,  k= 1, axes=(0,2))
            self._rot_bp = np.rot90(self._rot_bp, k= 1, axes=(0,2))
        elif action == "use":
            # Find agent:
            agent_pos = self.agent_position()
            if ((agent_pos[2] >= self._world.shape[2] - 1) or
                (self._world[agent_pos[0], agent_pos[1], agent_pos[2] + 1] != 'air')):
                # Agent is at the edge of the world, facing out, or is staring at a block.
                return
            # Determine all non-air blocks in agent's line-of-construction, to build on top of
            in_front = (self._world[agent_pos[0], max(0, agent_pos[1]-4):agent_pos[1], agent_pos[2]+1] != 'air')
            if in_front.any():
                # We're looking at a solid block, so we can place a block on top of it!
                new_block_y = agent_pos[1] - np.where(np.flip(in_front))[0][0]
                self._world[agent_pos[0], new_block_y, agent_pos[2]+1] = 'stone'
            elif agent_pos[1] <= 4:
                # There was no block to construct on, but we were low enough to see the floor.
                # Place a block on the floor instead
                self._world[agent_pos[0], 0, agent_pos[2]+1] = 'stone'
            # Else, can only see air in line-of-construction, so cannot construct!
        elif action == "attack":
            # Find agent:
            agent_pos = self.agent_position()
            if agent_pos[2] >= self._world.shape[2] - 1:
                # Agent is at the edge of the world, facing out. Don't bother tracking this action.
                return
            # Determine all air blocks in agents line-of-attack
            in_front = (self._world[agent_pos[0], max(0, agent_pos[1]-4):agent_pos[1]+1, agent_pos[2]+1] != 'air')
            if in_front.any():
                # We're looking at a solid block, so we can attack it
                attacked_block_y = agent_pos[1] - np.where(np.flip(in_front))[0][0] 
                self._world[agent_pos[0], attacked_block_y, agent_pos[2]+1] = 'air'
            elif agent_pos[1] <= 4:
                # There was no block to attack, but we were low enough to see the floor.
                # In the real MC, this would attack the floor, which we don't allow
                self._attacked_floor = True
            # Else, can only see air in line-of-attack, so cannot attack!
        else:
            print('ERROR: Illegal action {} specified.'.format(action))
            exit(1)

    def agent_position(self):
        return np.ravel(np.where(self._world == 'agent'))

    def get_observation(self):
        return np.array([self._rot_bp, self._world])

    def num_complete(self):
        return ((self._world == self._rot_bp) & (self._rot_bp != 'air')).sum()

    def num_incomplete(self):
        return ((self._rot_bp != self._world) & (self._rot_bp != 'air')).sum()

    def num_superfluous(self):
        return ((self._rot_bp != self._world) & (self._rot_bp == 'air') & (self._world != 'agent')).sum()

    def agent_in_arena(self):
        '''Returns true if world is uninitialized or agent is present in world model.'''
        return (self._world is None) or (self._world == 'agent').any()

    def agent_attacked_floor(self):
        return self._attacked_floor

    def mission_complete(self):
        return self.num_incomplete() == 0 and self.num_superfluous() == 0

    def distance_to_incomplete(self, default=None):
        '''Returns the minimum straight-line distance in Minecraft units from agent to a block specified in the blueprint which doesn't match the world'''
        # Tuple of arrays (X, Y, Z) s.t. (xj, yj, zj) is the j-th incomplete block in the blueprint
        incomplete = np.array(np.where( (self._rot_bp != self._world) & (self._rot_bp != 'air') & (self._world != 'agent') ))
        # Compute agent position:
        agent_pos = self.agent_position()
        if incomplete.size == 0 or agent_pos.size == 0:
            return default
        # Compute distance to each block using some numpy magic:
        #   Manually broadcasts agent_pos to line up with incomplete
        #   Then, computes difference of coordinates, and element-wise squares them
        #   Next, sums acroos coords of each sample, to give a 1D output vector
        #   Finally, takes min of these distances as output
        return np.sum( (incomplete - np.repeat(agent_pos.reshape((-1, 1)), incomplete.shape[1]))**2, axis=0 ).min()

    def facing_incomplete(self):
        agent_pos = self.agent_position()
        return ((agent_pos[2] < self._world.shape[2] - 1) and
                ( self._world[agent_pos[0], agent_pos[1], agent_pos[2]+1] == 'air') and
                (self._rot_bp[agent_pos[0], agent_pos[1], agent_pos[2]+1] != 'air') )

    def is_mission_running(self):
        return self.agent_in_arena() and not self.agent_attacked_floor() and not self.mission_complete()

    def reward(self):
        if self.agent_attacked_floor():
            return self._reward_weight['attack_floor']
        if not self.agent_in_arena():
            return self._reward_weight['leave_arena']
        if self.mission_complete():
            return self._reward_weight['mission_complete']
        # Compute the farthest an agent could theoretically be from the nearest blueprint block: opposite world corner
        max_dist = np.sum(np.array(self._world.shape)**2)
        reward = (
            # Use a default=1 on distance_to_incomplete
            #   so that agent optimizes that part all blocks are complete
            #   Avoids possibility of dancing around the last incomplete block to gain reward
            (  self._reward_weight['distance'] * (1 - abs((1/max_dist) - self.distance_to_incomplete(default=1))**0.4) ) +
            (  self._reward_weight['facing_incomplete'] * (self.facing_incomplete()) ) +
            # Reward actually placing necessary blocks, and penalize placing superfluous ones
            #   This also penalizes removing necessary blocks, and rewards removing superfluous ones
            #   That second function avoids being able to place the same block over and over to rack up rewards
            (  self._reward_weight['place_necessary'] * (self.num_complete() - self._old_num_complete) ) +
            (  self._reward_weight['place_superfluous'] * (self.num_superfluous() - self._old_num_superfluous) )
            )
        return reward
