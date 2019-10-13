# Author: David Legg
# Date: 10/11/19
# 
# This is a test of the Keras system on a simple reinforcement learning task.
# Setup is a 2-layer fully connected neural net, acting as a Deep Q-Learning system.
# Problem is a version of the NChain Gym environment, inspired by this article:
#   https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/

from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical
import numpy as np
from utils import chance

class ChainProblem:
    def __init__(self):
        self._state = 0
        self._dings = 0

    def act(self, action):
        if action == 0:
            if self._state == 4:
                self._dings += 1
                reward = 10
            else:
                reward = 0
                self._state += 1
        elif action == 1:
            reward = 2
            self._state = 0
        else:
            raise ValueError('Action {} is not allowed. Allowed actions are 0, 1.'.format(action))

        return (reward, self._state)

    def reset(self):
        self._state = 0
        return self._state

CHAIN          = ChainProblem()
NUM_EPISODES   = 1000
NUM_STEPS      = 200
DISCOUNT       = 0.95
EXPLORE_FACTOR = 1.00
EXPLORE_DECAY  = 0.99
NUM_STATES     = 5
NUM_ACTIONS    = 2

model = Sequential([
        InputLayer(input_shape=(5,)),
        Dense(10, activation='sigmoid'),
        Dense(2, activation='linear')
    ])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

def print_q_table():
    # Generates predicted Q values for every state/action pair
    predicted_q_values = model.predict(np.eye(NUM_STATES))
    print(' S\\A |', end='')
    for action in range(NUM_ACTIONS):
        print('{:>3}  |'.format(action), end='')
    print()
    print('-----+' * (NUM_ACTIONS + 1))
    for state in range(NUM_STATES):
        print('{:>4} |'.format(state), end='')
        for action in range(NUM_ACTIONS):
            print('{:>5.2f}|'.format(predicted_q_values[state, action]), end='')
        print()

def main():
    current_explore_factor = EXPLORE_FACTOR
    for episode_number in range(NUM_EPISODES):
        if episode_number % 10 == 0:
            print('Episode {}/{}'.format(episode_number, NUM_EPISODES))
            print_q_table()
            print('Dings:', CHAIN._dings)
            CHAIN._dings = 0
        current_explore_factor *= EXPLORE_DECAY
        new_observation = to_categorical( [CHAIN.reset()], NUM_STATES )
        for step_number in range(NUM_STEPS):
            observation = new_observation
            if chance(EXPLORE_FACTOR):
                action = np.random.randint(NUM_ACTIONS)
            else:
                action = model.predict_classes(observation)[0]
            (reward, new_observation) = CHAIN.act(action)
            new_observation = to_categorical( [new_observation], NUM_STATES )
            # target_value is known reward for the action we took, plus the discounted estimated reward for the state that put us in.
            target_value = reward + DISCOUNT * model.predict(new_observation).max()
            # target_vector is what we want the new output for the observation input to be
            target_vector = model.predict(observation)
            target_vector[0, action] = target_value
            # Now, actually update the model according to target_vector
            model.fit(observation, target_vector, epochs=1, verbose=0)
            observation = new_observation

    print('Training complete.')

    print('Example run:')
    print('State | Action | Reward')
    state = CHAIN.reset()
    for _ in range(NUM_STEPS):
        action = model.predict_classes( to_categorical( [state], NUM_STATES ) )[0]
        (reward, new_state) = CHAIN.act(action)
        print('{:>5} |{:>7} |{:>7} '.format(state, action, reward))
        state = new_state


if __name__ == '__main__':
    main()