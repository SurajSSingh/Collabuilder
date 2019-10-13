# Author: David Legg
# Date: 10/12/19
# 
# This is a test of the Keras system playing with Malmo
# Problem: A blue tile is placed in an otherwise uniformly stone play area,
#   surrounded by a lava boundary. Goal is to reach the blue tile.
# Architecture: 16-node hidden layer fully-connected NN, Q-Learner
# Training strategy: Reinforcement Learning using batches of recent history
#   epsilon-greedy exploration with exponential decay on epsilon

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from keras.utils import to_categorical
import numpy as np
from utils import chance, std_load
import random

import MalmoPython
import time
import sys
import os
import json

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

MISSION_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Do stuff!</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>12000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;1*minecraft:bedrock,5*minecraft:stone;1;"/>
                  <DrawingDecorator>
                    <DrawCuboid type="lava" x1="-4" y1="5" z1="2" x2="4" y2="5" z2="-11"/>
                    <DrawCuboid type="stone" x1="-2" y1="5" z1="0" x2="2" y2="5" z2="-9"/>
                    <DrawBlock type="lapis_block" x="0" y="5" z="-10"/>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="20000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Idiot</Name>
                <AgentStart>
                    <Placement x="0.5" y="6.0" z="0.5" yaw="180"/>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ObservationFromGrid>
                    <Grid name="floor3x3">
                      <min x="-1" y="-1" z="-1"/>
                      <max x="1" y="-1" z="1"/>
                    </Grid>
                  </ObservationFromGrid>
                  <DiscreteMovementCommands/>
                  <RewardForTouchingBlockType>
                    <Block reward="-100.0" type="lava" behaviour="onceOnly"/>
                    <Block reward="100.0" type="lapis_block" behaviour="onceOnly"/>
                  </RewardForTouchingBlockType>
                  <RewardForSendingCommand reward="1" />
                  <AgentQuitFromTouchingBlockType>
                      <Block type="lava" />
                      <Block type="lapis_block" />
                  </AgentQuitFromTouchingBlockType>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# tiles observed on the floor
INPUT_SHAPE = (3,3)

BLOCKS = ['stone', 'lava', 'lapis_block']
BLOCK_CODING = {b:i for i,b in enumerate(BLOCKS)}

ACTIONS = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
ACTION_CODING = {s:i for i,s in enumerate(ACTIONS)}
# TODO: when this parameter is turned down lower, 
#   End-of-mission rewards become unreliable. Diagnose why, and possibly file a bug report
ACTION_DELAY = 0.2

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)


class RLearner:
    def __init__(self):
        self._name = '3x3_grid_q_learner_v2.2'
        self._model = Sequential([
            # Take in a 3x3 grid of one-hot encoded blocks for the floor
            InputLayer(input_shape=(*(INPUT_SHAPE),len(BLOCKS))),
            Flatten(),
            Dense(16,activation='sigmoid'),
            # Output one-hot encoded action
            Dense(len(ACTIONS),activation='softmax')
        ])
        self._model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.start_epoch = 0
        self._model,self.start_epoch = std_load(self._name, self._model)
        self._epsilon_decay = 0.9995
        self._epsilon = 0.80 * (self._epsilon_decay**(self.start_epoch + 1))
        self._discount = 0.95
        self._history = []
        self._sample_size = 32
        self._max_history_size = 10 * self._sample_size
        self._last_obs = None
        self._last_action = None
        self._last_pred = None

    def act(self, last_reward, next_observation):
        # Update model based on last_reward:

        one_hot_obs = to_categorical( np.array(
                [BLOCK_CODING.get(s, BLOCK_CODING['lava']) for s in next_observation]
            ).reshape(1, 3, 3), len(BLOCKS) )

        if self._last_pred is not None:
            # Calculate the best-estimate Q-value for last_obs
            self._last_pred[self._last_action] = last_reward + self._discount * self._last_pred.max()
            # Save last observation and updated Q-values, cutting out batch axis on last_obs
            self._history.insert(0, (self._last_obs[0], self._last_pred))
            # trim the list using pop, since this is usually a small change
            while len(self._history) > self._max_history_size:
                self._history.pop()
            # Use that to update the model
            # self._model.fit(self._last_obs, self._last_pred, epochs=1, verbose=0)
            if len(self._history) >= self._sample_size:
                # Use random sample of history to update model:
                training_set = random.sample(self._history, self._sample_size)
                X_train = np.array([x for x,_ in training_set])
                Y_train = np.array([y for _,y in training_set])
                self._model.train_on_batch(X_train, Y_train)

        # Now, choose next action, and store last_* info for next iteration
        self._last_pred = self._model.predict(one_hot_obs)[0]
        self._last_obs  = one_hot_obs
        if chance(self._epsilon):
            self._last_action = np.random.randint(len(ACTIONS))
        else:
            self._last_action = self._last_pred.argmax()

        # Update epsilon after using it for chance
        self._epsilon *= self._epsilon_decay

        return ACTIONS[self._last_action]

    def save(self, id=None):
        self._model.save('checkpoint/' + self._name + ('' if id is None else '.' + id) + '.hdf5')

    def predict(self, observation):
        '''Runs the model on observation without saving to history or changing model weights.'''
        one_hot_obs = to_categorical( np.array(
                [BLOCK_CODING.get(s, BLOCK_CODING['lava']) for s in observation]
            ).reshape(1, 3, 3), len(BLOCKS) )
        raw_pred = self._model.predict(one_hot_obs)[0]
        return dict(zip(ACTIONS, raw_pred))

class tkDisplay:
    def __init__(self, model):
        self._model = model
        self._scale = 40
        # TODO: pull this from Malmo, don't generate it by hand
        #   Alternatively, generate this and generate Malmo's world from it.
        self._world_model = [
            ['lava','lava','lava','lava','lava','lava','lava'],
            ['lava','lava','lava','lapis_block','lava','lava','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','stone','stone','stone','stone','stone','lava'],
            ['lava','lava','lava','lava','lava','lava','lava']
        ]
        self._block_color = {
            'stone': '#000',
            'lava' : '#620',
            'lapis_block': '#006'
        }
        self._action_translation = {
            'movenorth 1': (0,-1),
            'movesouth 1': (0, 1),
            'moveeast 1' : ( 1,0),
            'movewest 1' : (-1,0),
        }

        self._root = tk.Tk()
        self._root.wm_title("Q Estimates")
        self._canvas = tk.Canvas(self._root,
            width  = self._scale * len(self._world_model[0]),
            height = self._scale * len(self._world_model),
            borderwidth        = 0,
            highlightthickness = 0,
            bg = "black"
            )
        self._canvas.grid()
        self._root.update()

    def update(self):
        self._canvas.delete('all')
        for y,row in enumerate(self._world_model):
            for x,block in enumerate(row):
                self._canvas.create_rectangle( x*self._scale, y*self._scale, (x+1)*self._scale, (y+1)*self._scale, outline="#fff", fill=self._block_color[block])
        # Overlay q-estimates, as different thickness partial lines
        for y in range(len(self._world_model)):
            for x in range(len(self._world_model[y])):
                if self._world_model[y][x] == 'lava':
                    continue
                observation = [self._world_model[j][i] for j in range(y-1, y+2) for i in range(x-1, x+2)]
                q_values    = self._model.predict(observation)
                total_q = sum(q_values.values())
                for action,q in q_values.items():
                    action_x,action_y = self._action_translation[action]
                    self._canvas.create_line(
                        (x + 0.5 + 0.2*action_x)*self._scale,
                        (y + 0.5 + 0.2*action_y)*self._scale,
                        (x + 0.5 + 0.5*action_x)*self._scale,
                        (y + 0.5 + 0.5*action_y)*self._scale,
                        fill="#3A0",
                        width=10*q/total_q
                        )
        self._root.update()


def run_mission(model):

    # Create default Malmo objects:

    my_mission = MalmoPython.MissionSpec(MISSION_XML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    for retry in range(MAX_RETRIES):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == MAX_RETRIES - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2**retry)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print("\nMission running.")

    total_reward = 0

    # Loop until mission ends, plus one more loop to collect final rewards:
    is_last_iter = False
    while world_state.is_mission_running or is_last_iter:
        world_state = agent_host.getWorldState()
        is_last_iter = not (world_state.is_mission_running or is_last_iter)
        for error in world_state.errors:
            print("Error:",error.text)
        if len(world_state.observations) > 0:
            current_r = 0
            for reward in world_state.rewards:
                current_r += reward.getValue()
            total_reward += current_r
            if not is_last_iter:
                floor = json.loads(world_state.observations[-1].text)["floor3x3"]
                agent_host.sendCommand( model.act( current_r, floor ) )
        time.sleep(ACTION_DELAY)

    print()
    print("Mission ended")

    return total_reward

def train_model(model, epochs, initial_epoch=0, display=None):
    best_reward = None
    for epoch_num in range(initial_epoch, epochs):
        print('Epoch {}/{}'.format(epoch_num, epochs))
        print('Current Epsilon: {}'.format(model._epsilon))
        reward = run_mission(model)
        print('Total reward:', reward)
        if display is not None:
            display.update()
        if best_reward is None or reward > best_reward or epoch_num % 10 == 0:
            model.save('epoch_{:03d}.reward_{:03d}'.format(epoch_num, int(reward)))
        if best_reward is None or reward > best_reward:
            best_reward = reward

if __name__ == '__main__':
    model = RLearner()
    disp  = tkDisplay(model)
    disp.update()
    train_model(model, 1000, initial_epoch=model.start_epoch, display=disp)


