# This is the most basic version of the building task.
# A "blueprint" with only one square set for building is given to the agent.
# The agent is reward for placing a block on this square, and mildly punished for any other action.
# The idea here is to train an agent how to place a block, as well as to experiment with 

from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer, Flatten, Reshape, Permute, Conv3D, MaxPooling3D
from keras.utils import to_categorical
import numpy as np
from utils import chance, std_load
import random
import math

import MalmoPython
import time
import sys
import os
import json

import matplotlib.pyplot as plt
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

MAX_RETRIES = 10

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

def build_world(training=False):
    '''Sets global variables that represent the world, and connects AGENT_HOST.
If training=True, sets overclocking and deactivates rendering.'''
    global BLUEPRINT, BLUEPRINT_OH, MISSION_XML, ACTION_DELAY, AGENT_HOST

    BLUEPRINT = [[['air' for _ in range(ARENA_WIDTH)] for _ in range(ARENA_HEIGHT)] for _ in range(ARENA_LENGTH)]
    BLUEPRINT[int(ARENA_LENGTH/2)][0][int(ARENA_HEIGHT/2)] = 'stone'
    BLUEPRINT = np.array(BLUEPRINT)

    BLUEPRINT_OH = to_categorical( np.vectorize(INPUTS_CODING.get)(BLUEPRINT), len(INPUTS) )

    MISSION_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
            <About>
              <Summary>Place a block!</Summary>
            </About>

              <ModSettings>
                <MsPerTick>{ms_per_tick}</MsPerTick>
                <!-- <PrioritiseOffscreenRendering>{offscreen_rendering}</PrioritiseOffscreenRendering> -->
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
                  <ServerQuitFromTimeUp timeLimitMs="20000"/>
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
                  <RewardForSendingCommand reward="-1" />
                </AgentHandlers>
              </AgentSection>
            </Mission>'''.format(
                    ms_per_tick         = int(50/OVERCLOCK_FACTOR if training else 50),
                    offscreen_rendering = (1 if training else 0),
                    start_x = START_X + OFFSET_X,
                    start_y = START_Y + OFFSET_Y - 1, # -1 corrects for a mismatch in the way MC positions characters vs. how it reads the position back
                    start_z = START_Z + OFFSET_Z,
                    arena_x1 = ANCHOR_X, arena_x2 = ANCHOR_X - 1 + ARENA_WIDTH,
                    arena_y1 = ANCHOR_Y, arena_y2 = ANCHOR_Y - 1 + ARENA_HEIGHT,
                    arena_z1 = ANCHOR_Z, arena_z2 = ANCHOR_Z - 1 + ARENA_LENGTH,
                    border_x1 = ANCHOR_X - 1, border_x2 = ANCHOR_X + ARENA_WIDTH,
                    border_y1 = ANCHOR_Y - 2, border_y2 = ANCHOR_Y - 1,
                    border_z1 = ANCHOR_Z - 1, border_z2 = ANCHOR_Z + ARENA_LENGTH
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
    def __init__(self, blueprint):
        self._bp       = blueprint
        self._world    = None
        self._str_type = '<U{}'.format(max(len(s) for s in INPUTS))
        self._rot_bp   = self._bp
        self._old_num_complete    = 0
        self._old_num_incomplete  = 0
        self._old_num_superfluous = 0

    def update(self, raw_obs):
        if self._world is not None:
            self._old_num_complete    = self.num_complete()
            self._old_num_incomplete  = self.num_incomplete()
            self._old_num_superfluous = self.num_superfluous()

        raw_world = np.array( raw_obs["world_grid"], dtype=self._str_type )
        world = np.transpose(np.reshape(raw_world, (ARENA_WIDTH, ARENA_LENGTH, ARENA_HEIGHT)), (2, 0, 1))
        agent_yaw = raw_obs['Yaw']
        agent_x = int(raw_obs['XPos'] - OFFSET_X)
        agent_y = int(raw_obs['YPos'] - OFFSET_Y)
        agent_z = int(raw_obs['ZPos'] - OFFSET_Z)
        if (0 <= agent_x < world.shape[0] and 
            0 <= agent_y < world.shape[1] and 
            0 <= agent_z < world.shape[2] ):
            world[agent_x, agent_y, agent_z] = 'agent'
        # Rotate world and blueprint to be agent-facing
        self._world  = np.rot90(world,    k=-int(np.round(agent_yaw/90)), axes=(0,2))
        self._rot_bp = np.rot90(self._bp, k=-int(np.round(agent_yaw/90)), axes=(0,2))

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

    def mission_complete(self):
        return self.num_incomplete() == 0 and self.num_superfluous() == 0

    def reward(self):
        return (
            (  10 * (self.num_complete() - self._old_num_complete) ) +
            ( -10 * (self.num_superfluous() - self._old_num_superfluous) ) +
            (-200 * (not self.agent_in_arena()) ) +
            ( 200 * (self.mission_complete()) )
            )

class RLearner:
    '''Implements a target-network Deep Q-Learning architecture.'''
    def __init__(self):
        self._name = 'block_placer_v1.0'
        self._prediction_network = Sequential([
            # Take in the blueprint as desired and the state of the world, in the same shape as the blueprint
            InputLayer(input_shape=(2, *BLUEPRINT_OH.shape)),
            # Reduce incoming 6D tensor to 5D by merging channels (dim 1) and one-hot (dim 6):
            Permute((1, 5, 2, 3, 4)),
            Reshape((-1, *BLUEPRINT.shape)),
            # Convolve each input, treating the blueprint and world state as separate channels
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


        self._target_update_frequency = 20
        self._iters_since_target_update = 0
        self._epsilon_decay = 0.9995
        self._epsilon = 0.90 * (self._epsilon_decay**(self.start_epoch + 1))
        self._discount = 0.95
        self._last_obs = None
        self._last_action = None

    def _preprocess(self, observation):
        return np.array([
            to_categorical( np.vectorize(INPUTS_CODING.get)(observation), len(INPUTS) )
            ])

    def act(self, last_reward, next_observation):
        # Update model based on last_reward:

        one_hot_obs = self._preprocess(next_observation)

        if self._last_obs is not None:
            target = last_reward + self._discount * self._target_network.predict(self._last_obs).max()
            target_vec = self._target_network.predict(self._last_obs)
            target_vec[0, self._last_action] = target
            self._prediction_network.train_on_batch(self._last_obs, target_vec)

        # Now, choose next action, and store last_* info for next iteration
        self._last_obs  = one_hot_obs
        if chance(self._epsilon):
            self._last_action = np.random.randint(len(ACTIONS))
        else:
            self._last_action = self._prediction_network.predict(one_hot_obs).argmax()

        # Update epsilon after using it for chance
        self._epsilon *= self._epsilon_decay
        # Increment counter for target update
        self._iters_since_target_update += 1
        if self._iters_since_target_update >= self._target_update_frequency:
            self._target_network.set_weights(self._prediction_network.get_weights())
            self._iters_since_target_update = 0

        return ACTIONS[self._last_action]

    def mission_ended(self):
        self._last_obs = None
        self._last_action = None

    def save(self, id=None):
        self._prediction_network.save('checkpoint/' + self._name + ('' if id is None else '.' + id) + '.hdf5')

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

        plt.draw()
        plt.pause(.001)

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

def train_model(model, epochs, initial_epoch=0, display=None):
    best_reward = None
    for epoch_num in range(initial_epoch, epochs):
        print('Epoch {}/{}'.format(epoch_num, epochs))
        print('Current Epsilon: {}'.format(model._epsilon))
        reward = run_mission(model, display=display)
        print('Total reward:', reward)
        if best_reward is None or reward > best_reward or epoch_num % 10 == 0:
            model.save('epoch_{:03d}.reward_{:03d}'.format(epoch_num, int(reward)))
        if best_reward is None or reward > best_reward:
            best_reward = reward

if __name__ == '__main__':
    build_world(training=False)
    model = RLearner()
    disp  = Display(model)
    train_model(model, 1000, initial_epoch=model.start_epoch, display=disp)


