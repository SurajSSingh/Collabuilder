# This is the most basic version of the building task.
# A "blueprint" with only one square set for building is given to the agent.
# The agent is reward for placing a block on this square, and mildly punished for any other action.
# The idea here is to train an agent how to place a block, as well as to experiment with 

from keras.models import Sequential, clone_model
from keras.layers import Dense, InputLayer, Flatten, Reshape, Permute, Conv3D, MaxPooling3D
from keras.utils import to_categorical, Sequence
from keras.callbacks import LambdaCallback
import numpy as np
from utils import chance, std_load, ask_options, ask_yn, ask_int, get_config
import random
import math
import functools

import MalmoPython
import time
import sys
import os
import json

from world_model import WorldModel
from display import Display, RewardsPlot

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    print = functools.partial(print, flush=True)

MODEL_NAME = 'bp_simplified'
VERSION_NUMBER  = '1.1'
CONFIG_FILE = MODEL_NAME + '_v' + VERSION_NUMBER
cfg = lambda *args: get_config(CONFIG_FILE, *args)

MAX_RETRIES = 10

NUM_EPISODES       = cfg('training', 'num_episodes')
SAVE_FREQ          = int(NUM_EPISODES / 100)
MAX_EPISODE_TIME   = cfg('training', 'max_episode_time')
INITIAL_EPSILON    = cfg('training', 'initial_epsilon')
FINAL_EPSILON      = cfg('training', 'final_epsilon')
EPSILON_DECAY      = (FINAL_EPSILON / INITIAL_EPSILON)**(1.0/NUM_EPISODES)
SAVE_HISTORY       = cfg('training', 'save_history')
TRAIN_ON_HISTORY   = cfg('training', 'train_on_history')
HISTORY_BATCH_SIZE = cfg('training', 'history', 'batch_size')
HISTORY_BATCHES    = cfg('training', 'history', 'batches')
HISTORY_EPOCHS     = cfg('training', 'history', 'epochs')

# All coordinates are in Minecraft's global coordinate system.
ARENA_WIDTH  = cfg('arena', 'width')  # X-direction
ARENA_HEIGHT = cfg('arena', 'height') # Y-direction
ARENA_LENGTH = cfg('arena', 'length') # Z-direction
ANCHOR_X     = cfg('arena', 'anchor', 'x')
ANCHOR_Y     = cfg('arena', 'anchor', 'y')
ANCHOR_Z     = cfg('arena', 'anchor', 'z')
OFFSET_X     = ANCHOR_X + cfg('arena', 'offset', 'x')
OFFSET_Y     = ANCHOR_Y + cfg('arena', 'offset', 'y')
OFFSET_Z     = ANCHOR_Z + cfg('arena', 'offset', 'z')
BLUEPRINT    = None
MISSION_XML  = None
ACTION_DELAY = None
AGENT_HOST   = None

INPUTS = cfg('inputs')
INPUTS_CODING = {b:i for i,b in enumerate(INPUTS)}

# Use = Place block
# Attack = Mine block
ACTIONS = cfg('actions')
ACTION_CODING = {s:i for i,s in enumerate(ACTIONS)}

OVERCLOCK_FACTOR = cfg('training', 'overclock_factor')

def build_world(training=False):
    '''Sets global variables that represent the world, and connects AGENT_HOST.
If training=True, sets overclocking and deactivates rendering.'''
    global BLUEPRINT, BLUEPRINT_OH, MISSION_XML, ACTION_DELAY, AGENT_HOST

    BLUEPRINT = [[['air' for _ in range(ARENA_WIDTH)] for _ in range(ARENA_HEIGHT)] for _ in range(ARENA_LENGTH)]
    BLUEPRINT[int(ARENA_LENGTH/2)][0][int(ARENA_WIDTH/2)] = 'stone'
    BLUEPRINT = np.array(BLUEPRINT)

    start_x = np.random.randint(ARENA_LENGTH)
    start_z = np.random.randint(ARENA_WIDTH)
    start_y = 0

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
            InputLayer(input_shape=(2, *BLUEPRINT.shape, len(INPUTS))),
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

def run_mission(model, display=None):

    # Create default Malmo objects:
    my_mission = MalmoPython.MissionSpec(MISSION_XML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    world_model = WorldModel(BLUEPRINT, CONFIG_FILE, simulated=False)
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
           world_model.is_mission_running()):
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

    world_model  = WorldModel(BLUEPRINT, CONFIG_FILE, simulated=True)
    ticks_left   = 5*MAX_EPISODE_TIME
    total_reward = 0
    current_r    = 0

    while (ticks_left > 0 and
           world_model.is_mission_running()):
        current_r = world_model.reward()
        action = model.act(current_r, world_model.get_observation())
        if display is not None:
            display.update(world_model)
        total_reward += current_r
        world_model.simulate(action)
        if use_delays:
            print(action)
            time.sleep(ACTION_DELAY)

    # Collect last reward, and give to model, then end the mission
    current_r = world_model.reward()
    model.act(current_r, world_model.get_observation())
    total_reward += current_r
    model.mission_ended()
    print("Simulated mission ended")

    return total_reward

def train_model(model, epochs, initial_epoch=0, display=None, simulated=False, plot_rewards=False):
    if plot_rewards:
        rp = RewardsPlot()

    best_reward = None
    for epoch_num in range(initial_epoch, epochs):
        if epoch_num % 10 == 0:
            build_world(training=True)
        print('Epoch {}/{}'.format(epoch_num, epochs))
        print('Current Epsilon: {}'.format(model._epsilon))
        reward = (run_simulated_mission(model, display=display) if simulated else run_mission(model, display=display))
        print('Total reward:', reward)
        rp.add(reward)
        if best_reward is None or reward > best_reward or epoch_num % SAVE_FREQ == 0:
            model.save('epoch_{:09d}.reward_{:03d}'.format(epoch_num, int(reward)))
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

    build_world(training=set_training)
    model = RLearner(save_history=SAVE_HISTORY)
    if TRAIN_ON_HISTORY:
        model.train_on_history(
            batch_size = HISTORY_BATCH_SIZE,
            batches    = HISTORY_BATCHES,
            epochs     = HISTORY_EPOCHS
        )
    disp  = (Display(model) if set_display else None)
    if set_training:
        pr = ask_yn('Plot rewards?')
        train_model(model, NUM_EPISODES, initial_epoch=model.start_epoch, display=disp, simulated=set_simulated, plot_rewards=pr)
        print('Training complete.\n\n')

    # Training complete or skipped, just show learned policy:
    model._epsilon = 0
    # Turn on display, regardless of settings, for demo
    if disp is None:
        disp = Display(model)
    print('Demonstration running...')
    if set_simulated:
        reward = run_simulated_mission(model, display=disp, use_delays=True)
    else:
        reward = run_mission(model, display=disp)

    print('Collected reward: {}'.format(reward))
    print('Demonstration complete.')


