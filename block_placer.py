# This is the most basic version of the building task.
# A "blueprint" with only one square set for building is given to the agent.
# The agent is reward for placing a block on this square, and mildly punished for any other action.
# The idea here is to train an agent how to place a block, as well as to experiment with 

from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer, Flatten, Reshape, Permute, Conv3D, MaxPooling3D
from keras.utils import to_categorical, Sequence
from keras.callbacks import LambdaCallback
import numpy as np
from utils import chance, std_load, ask_options, ask_yn, ask_int
import random
import math

import MalmoPython
import time
import sys
import os
import json

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

MODEL_NAME     = 'block_placer'
VERSION_NUMBER = '2.4'

MAX_RETRIES = 10

NUM_EPISODES     = 100000
SAVE_FREQ        = int(NUM_EPISODES / 100)
MAX_EPISODE_TIME = 20
INITIAL_EPSILON  = 0.5
FINAL_EPSILON    = 0.01
EPSILON_DECAY    = (FINAL_EPSILON / INITIAL_EPSILON)**(1.0/NUM_EPISODES)

# All coordinates are in Minecraft's global coordinate system.
ARENA_WIDTH  = 5 # X-direction
ARENA_HEIGHT = 5 # Y-direction
ARENA_LENGTH = 5 # Z-direction
ANCHOR_X     = 0
ANCHOR_Y     = 5
ANCHOR_Z     = 0
OFFSET_X     = ANCHOR_X + 0.5
OFFSET_Y     = ANCHOR_Y
OFFSET_Z     = ANCHOR_Z + 0.5
START_X      = 2
START_Y      = 1
START_Z      = 1
BLUEPRINT    = None
BLUEPRINT_OH = None
MISSION_XML  = None
ACTION_DELAY = None
AGENT_HOST   = None

INPUTS = ['air', 'stone', 'agent']
INPUTS_CODING = {b:i for i,b in enumerate(INPUTS)}

# Use = Place block
# Attack = Mine block
ACTIONS = ["jumpmove 1", "turn 1", "turn -1", "use", "attack"]
ACTION_CODING = {s:i for i,s in enumerate(ACTIONS)}

OVERCLOCK_FACTOR = 5

def build_world(training=False, randomize_start_xz=False):
    '''Sets global variables that represent the world, and connects AGENT_HOST.
If training=True, sets overclocking and deactivates rendering.'''
    global BLUEPRINT, BLUEPRINT_OH, MISSION_XML, ACTION_DELAY, AGENT_HOST

    BLUEPRINT = [[['air' for _ in range(ARENA_WIDTH)] for _ in range(ARENA_HEIGHT)] for _ in range(ARENA_LENGTH)]
    BLUEPRINT[int(ARENA_LENGTH/2)][0][int(ARENA_WIDTH/2)] = 'stone'
    BLUEPRINT = np.array(BLUEPRINT)

    start_x = (np.random.randint(ARENA_LENGTH) if randomize_start_xz else START_X)
    start_z = (np.random.randint(ARENA_WIDTH) if randomize_start_xz else START_Z)
    start_y = START_Y

    BLUEPRINT_OH = to_categorical( np.vectorize(INPUTS_CODING.get)(BLUEPRINT), len(INPUTS) )

    MISSION_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
            <About>
              <Summary>Place a block!</Summary>
            </About>

              <ModSettings>
                <MsPerTick>{ms_per_tick}</MsPerTick>
                <PrioritiseOffscreenRendering>{offscreen_rendering}</PrioritiseOffscreenRendering>
              </ModSettings>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>12000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;5*minecraft:bedrock;1;" forceReset="1"/>
                  <ServerQuitFromTimeUp timeLimitMs="{server_timeout}"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Blockhead</Name>
                <AgentStart>
                  <Placement x="{start_x}" y="{start_y}" z="{start_z}" yaw="0" pitch="70"/>
                  <Inventory>
                    <InventoryObject slot="0" type="stone" quantity="64"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ObservationFromGrid>
                    <Grid name="world_grid" absoluteCoords="1">
                      <min x="{arena_x1}" y="{arena_y1}" z="{arena_z1}"/>
                      <max x="{arena_x2}" y="{arena_y2}" z="{arena_z2}"/>
                    </Grid>
                  </ObservationFromGrid>
                  <DiscreteMovementCommands/>
                  <MissionQuitCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''.format(
                    ms_per_tick         = int(50/OVERCLOCK_FACTOR if training else 50),
                    offscreen_rendering = ('false' if training else 'true'),
                    start_x = start_x + OFFSET_X,
                    start_y = start_y + OFFSET_Y - 1, # -1 corrects for a mismatch in the way MC positions characters vs. how it reads the position back
                    start_z = start_z + OFFSET_Z,
                    arena_x1 = ANCHOR_X, arena_x2 = ANCHOR_X - 1 + ARENA_WIDTH,
                    arena_y1 = ANCHOR_Y - 1, arena_y2 = ANCHOR_Y - 1 + ARENA_HEIGHT,
                    arena_z1 = ANCHOR_Z, arena_z2 = ANCHOR_Z - 1 + ARENA_LENGTH,
                    border_x1 = ANCHOR_X - 1, border_x2 = ANCHOR_X + ARENA_WIDTH,
                    border_y1 = ANCHOR_Y - 2, border_y2 = ANCHOR_Y - 1,
                    border_z1 = ANCHOR_Z - 1, border_z2 = ANCHOR_Z + ARENA_LENGTH,
                    server_timeout = 1000*MAX_EPISODE_TIME
                )
    ACTION_DELAY = (0.2/OVERCLOCK_FACTOR if training else 0.2)

    AGENT_HOST = MalmoPython.AgentHost()
    try:
        AGENT_HOST.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(AGENT_HOST.getUsage())
        exit(1)
    if AGENT_HOST.receivedArgument("help"):
        print(AGENT_HOST.getUsage())
        exit(0)

class WorldModel:
    def __init__(self, blueprint, simulated=False):
        self._bp       = blueprint
        self._str_type = '<U{}'.format(max(len(s) for s in INPUTS))
        self._rot_bp   = self._bp
        self._old_num_complete    = 0
        self._old_num_incomplete  = 0
        self._old_num_superfluous = 0
        self._attacked_floor = False
        if not simulated:
            # Wait for update() from Minecraft
            self._world = None
        else:
            # Build world with random agent start position
            self._world = np.full((ARENA_LENGTH, ARENA_HEIGHT, ARENA_WIDTH), fill_value='air', dtype=self._str_type)
            self._world[np.random.randint(ARENA_LENGTH),0,np.random.randint(ARENA_WIDTH)] = 'agent'

    def update(self, raw_obs):
        '''Used when hooked up to Minecraft.'''
        if self._world is not None:
            self._old_num_complete    = self.num_complete()
            self._old_num_incomplete  = self.num_incomplete()
            self._old_num_superfluous = self.num_superfluous()

        raw_world = np.array( raw_obs["world_grid"], dtype=self._str_type )
        extended_world = np.transpose(np.reshape(raw_world, (ARENA_HEIGHT+1, ARENA_WIDTH, ARENA_LENGTH)), (2, 0, 1))
        world = extended_world[:,1:,:]
        agent_yaw = raw_obs['Yaw']
        agent_x = int(raw_obs['XPos'] - OFFSET_X)
        agent_y = int(raw_obs['YPos'] - OFFSET_Y)
        agent_z = int(raw_obs['ZPos'] - OFFSET_Z)
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

    def reward(self):
        if self.agent_attacked_floor() or not self.agent_in_arena():
            return -1
        if self.mission_complete():
            return 1
        # Compute the farthest an agent could theoretically be from the nearest blueprint block: opposite world corner
        max_dist = np.sum(np.array(self._world.shape)**2)
        return (
            # Use a default=1 on distance_to_incomplete
            #   so that agent optimizes that part all blocks are complete
            #   Avoids possibility of dancing around the last incomplete block to gain reward
            (  0.1 * (1 - abs((1/max_dist) - self.distance_to_incomplete(default=1))**0.4) ) +
            (  0.1 * (self.facing_incomplete()) ) +
            # Reward actually placing necessary blocks, and penalize placing superfluous ones
            #   This also penalizes removing necessary blocks, and rewards removing superfluous ones
            #   That second function avoids being able to place the same block over and over to rack up rewards
            (  0.5 * (self.num_complete() - self._old_num_complete) ) +
            ( -0.5 * (self.num_superfluous() - self._old_num_superfluous) )
            )

class RLearner:
    '''Implements a target-network Deep Q-Learning architecture.'''

    class HistoryGenerator(Sequence):
        '''Generates training data based on a history file.'''
        # Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        def __init__(self, history_file, batch_size, batches, discount, target_network):
            try:
                self._history = np.load(history_file)
            except IOError:
                print('No history file. Could not train on history.')
                return
            self._batch_size     = batch_size
            self._batches        = batches
            self._discount       = discount
            self._target_network = target_network
            self._history_length = self._history['observation'].shape[0]
            self.on_epoch_end()

        def on_epoch_end(self):
            print('Generating training data...')
            # First, generate all the indices for all the batches
            idx = np.random.randint(self._history_length, size=(self._batches * self._batch_size))
            # Collect inputs:
            self._X = self._history['observation'][idx]
            # Compute basic outputs:
            self._Y = self._target_network.predict(self._history['observation'][idx])
            # Compute best-estimate outputs for the action that was taken on that sample,
            #   using reward for that action and current best-estimate of the next state
            self._Y[np.arange(self._Y.shape[0]), np.array(self._history['action'][idx], dtype='int')] = (
                    self._history['reward'][idx] +
                    self._discount * self._target_network.predict(self._history['next_observation'][idx]).max(axis=1)
                )
            print('Generated.')

        def __len__(self):
            return self._batches

        def __getitem__(self, idx):
            start = idx * self._batch_size
            end   = start + self._batch_size
            return (self._X[start:end], self._Y[start:end])

    def __init__(self, save_history=False):
        self._name = MODEL_NAME + '_v' + VERSION_NUMBER
        self._save_history = save_history
        self._history_file = 'history/{}.npz'.format(self._name)
        self._unsaved_history = {'observation': [], 'action': [], 'reward': [], 'next_observation': []}
        self._prediction_network = Sequential([
            # Take in the blueprint as desired and the state of the world, in the same shape as the blueprint
            InputLayer(input_shape=(2, *BLUEPRINT_OH.shape)),
            # Reduce incoming 6D tensor to 5D by merging channels (dim 1) and one-hot (dim 5):
            Permute((1, 5, 2, 3, 4)),
            Reshape((-1, *BLUEPRINT.shape)),
            # Convolve each input, treating the blueprint and world state as separate channels
            Conv3D(8, (3, 3, 3), padding="same", data_format="channels_first", activation="relu"),
            Conv3D(8, (3, 3, 3), padding="same", data_format="channels_first", activation="relu"),
            # max-pool features together a bit:
            MaxPooling3D(pool_size=BLUEPRINT.shape, data_format="channels_first"),
            # Flatten, ready for fully-connected layers:
            Flatten(),
            # Do some thinking:
            Dense(16, activation='relu'),
            # Output one-hot encoded action
            Dense(len(ACTIONS), activation='softmax')
        ])
        self._prediction_network.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.start_epoch = 0
        self._prediction_network,self.start_epoch = std_load(self._name, self._prediction_network)
        self._target_network = clone_model(self._prediction_network)
        self._target_network.build(self._prediction_network.input_shape)

        self._target_update_frequency = 20
        self._iters_since_target_update = 0
        self._epsilon_decay = EPSILON_DECAY
        self._epsilon = INITIAL_EPSILON * (self._epsilon_decay**(self.start_epoch + 1))
        self._discount = 0.95
        self._last_obs = None
        self._last_action = None

    def _preprocess(self, observation):
        return np.array([
            to_categorical( np.vectorize(INPUTS_CODING.get)(observation), len(INPUTS) )
            ])

    def _maybe_update_pn(self):
        if self._iters_since_target_update >= self._target_update_frequency:
            self._target_network.set_weights(self._prediction_network.get_weights())
            self._iters_since_target_update = 0

    def act(self, last_reward, next_observation):
        # Update model based on last_reward:

        one_hot_obs = self._preprocess(next_observation)

        if self._last_obs is not None:
            target = last_reward + self._discount * self._target_network.predict(one_hot_obs).max()
            target_vec = self._target_network.predict(self._last_obs)
            target_vec[0, self._last_action] = target
            self._prediction_network.train_on_batch(self._last_obs, target_vec)
            self._unsaved_history['observation'].append(self._last_obs[0])
            self._unsaved_history['action'].append(self._last_action)
            self._unsaved_history['reward'].append(last_reward)
            self._unsaved_history['next_observation'].append(one_hot_obs[0])

        # Now, choose next action, and store last_* info for next iteration
        self._last_obs  = one_hot_obs
        if chance(self._epsilon):
            self._last_action = np.random.randint(len(ACTIONS))
        else:
            self._last_action = self._prediction_network.predict(one_hot_obs).argmax()

        # Update epsilon after using it for chance
        # Increment counter for target update
        self._iters_since_target_update += 1
        self._maybe_update_pn()

        return ACTIONS[self._last_action]

    def mission_ended(self):
        self._last_obs = None
        self._last_action = None
        self._epsilon *= self._epsilon_decay

    def save(self, id=None):
        self._prediction_network.save('checkpoint/' + self._name + ('' if id is None else '.' + id) + '.hdf5')
        self.save_history()

    def save_history(self):
        # TODO: consider mem-mapping to deal with large history files
        if self._save_history and len(self._unsaved_history['observation']) > 0:
            new_history = {k:np.array(v) for k,v in self._unsaved_history.items()}
            for v in self._unsaved_history.values():
                v.clear()
            try:
                old_history = np.load(self._history_file)
            except IOError:
                old_history = {k:np.zeros((0, *v.shape[1:])) for k,v in new_history.items()}
            np.savez(self._history_file, **{k:np.concatenate([old_history[k], v]) for k,v in new_history.items()})

    def train_on_history(self, batch_size=1000, batches=100, epochs=10):
        print('Training on history...')

        def update_pn(*args, **kwargs):
            self._iters_since_target_update += batch_size
            self._maybe_update_pn()

        self._prediction_network.fit_generator(RLearner.HistoryGenerator(
                history_file   = self._history_file,
                batch_size     = batch_size,
                batches        = batches,
                discount       = self._discount,
                target_network = self._target_network
                ),
            epochs=epochs,
            callbacks=[LambdaCallback(on_batch_end=update_pn)]
            )
        print('Finished training on history.')

    def predict(self, observation):
        '''Runs the model on observation without saving to history or changing model weights.'''
        one_hot_obs = self._preprocess(observation)
        raw_pred = self._model.predict(one_hot_obs)[0]
        return dict(zip(ACTIONS, raw_pred))

class Display:
    def __init__(self, model):
        self._model = model
        self._scale = 40
        self._block_color = {
            'stone': '#4040D0',
            'agent': '#D04040'
        }

        self._world_alpha = 'D0'
        self._bp_alpha = '40'

        self._fig = plt.figure()
        self._axis = self._fig.add_subplot( 111, projection='3d' )
        self._fig.show()

    def update(self, world_model):
        bp, wd = world_model.get_observation()
        plt_bp = np.flip(bp.transpose( (0,2,1) ), 0)
        plt_wd = np.flip(wd.transpose( (0,2,1) ), 0)
        not_air = (plt_bp != 'air') | (plt_wd != 'air')
        colormap = np.full(plt_bp.shape, '#00000000')
        for block,color in self._block_color.items():
            # Set bp-only blocks to show with bp alpha, world blocks show with real alpha
            colormap[plt_bp == block] = color + self._bp_alpha
            colormap[plt_wd == block] = color + self._world_alpha

        self._axis.clear()
        self._axis.voxels(filled=not_air, facecolors=colormap)

        self._fig.canvas.flush_events()

def run_mission(model, display=None):

    # Create default Malmo objects:
    my_mission = MalmoPython.MissionSpec(MISSION_XML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    world_model = WorldModel(BLUEPRINT)
    # Attempt to start a mission:
    for retry in range(MAX_RETRIES):
        try:
            AGENT_HOST.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == MAX_RETRIES - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2**retry)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = AGENT_HOST.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = AGENT_HOST.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print("\nMission running.")

    total_reward = 0
    current_r = 0

    # Loop until mission ends
    while (world_state.is_mission_running and
           world_model.agent_in_arena() and
           not world_model.mission_complete()):
        world_state = AGENT_HOST.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
        current_r += sum(r.getValue() for r in world_state.rewards)
        if len(world_state.observations) > 0:
            raw_obs = json.loads(world_state.observations[-1].text)
            world_model.update(raw_obs)
            current_r += world_model.reward()
            action = model.act( current_r, world_model.get_observation() )
            if display is not None:
                display.update(world_model)
            total_reward += current_r
            current_r = 0
            if world_model.mission_complete() or not world_model.agent_in_arena():
                AGENT_HOST.sendCommand('quit')
            elif world_state.is_mission_running:
                AGENT_HOST.sendCommand( action )
        time.sleep(ACTION_DELAY)

    model.mission_ended()
    print()
    print("Mission ended")

    return total_reward

def run_simulated_mission(model, display=None, use_delays=False):
    print("Simulated mission running.")

    world_model  = WorldModel(BLUEPRINT, simulated=True)
    ticks_left   = 5*MAX_EPISODE_TIME
    total_reward = 0
    current_r    = 0

    while (ticks_left > 0 and
           world_model.agent_in_arena() and
           not world_model.mission_complete()):
        current_r = world_model.reward()
        action = model.act(current_r, world_model.get_observation())
        if display is not None:
            display.update(world_model)
        total_reward += current_r
        world_model.simulate(action)
        if use_delays:
            time.sleep(ACTION_DELAY)

    # Collect last reward, and give to model, then end the mission
    current_r = world_model.reward()
    model.act(current_r, world_model.get_observation())
    total_reward += current_r
    model.mission_ended()
    print()
    print("Simulated mission ended")

    return total_reward

def train_model(model, epochs, initial_epoch=0, display=None, simulated=False):
    best_reward = None
    for epoch_num in range(initial_epoch, epochs):
        if epoch_num % 10 == 0:
            build_world(training=True, randomize_start_xz=True)
        print('Epoch {}/{}'.format(epoch_num, epochs))
        print('Current Epsilon: {}'.format(model._epsilon))
        reward = (run_simulated_mission(model, display=display) if simulated else run_mission(model, display=display))
        print('Total reward:', reward)
        if best_reward is None or reward > best_reward or epoch_num % SAVE_FREQ == 0:
            model.save('epoch_{:03d}.reward_{:03d}'.format(epoch_num, int(reward)))
        if best_reward is None or reward > best_reward:
            best_reward = reward

if __name__ == '__main__':
    modes = {
        'Training - Simulated, no Display': (True, False, True),
        'Training - Simulated w/ Display': (True, True, True),
        'Training - Real w/ Display': (True, True, False),
        'Demonstration - Simulated': (False, True, True),
        'Demonstration - Real': (False, True, False)
    }
    set_training,set_display,set_simulated = modes[ask_options('Select execution mode:', list(modes.keys()))]

    if set_training:
        save_history  = ask_yn('Save samples to history file?')
        train_history = ask_yn('Train on history file first?')
    else:
        save_history,train_history = False,False

    build_world(training=set_training, randomize_start_xz=True)
    model = RLearner(save_history=save_history)
    if train_history:
        model.train_on_history(
            batch_size = ask_int('Batch size for history training: ', min_val = 1),
            batches    = ask_int('Batches per epoch for history training: ', min_val = 1),
            epochs     = ask_int('Epochs for history training: ', min_val = 1)
        )
    disp  = (Display(model) if set_display else None)
    if set_training:
        train_model(model, NUM_EPISODES, initial_epoch=model.start_epoch, display=disp, simulated=set_simulated)
    elif set_simulated:
        run_simulated_mission(model, display=disp, use_delays=True)
    else:
        run_mission(model, display=disp)


