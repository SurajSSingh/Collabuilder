import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import multiprocessing

# from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import LambdaCallback

from utils import std_load, chance, ask_int

tf.config.threading.set_intra_op_parallelism_threads(
    ask_int('Number of intra-op threads: ', min_val=1, default=multiprocessing.cpu_count()))
tf.config.threading.set_inter_op_parallelism_threads(
    ask_int('Number of inter-op threads: ', min_val=1, default=2))

class RLearner:
    '''Implements a target-network Deep Q-Learning architecture.'''

    class HistoryGenerator(Sequence):
        '''Generates training data based on a history file.'''
        # Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        def __init__(self, history_file, batch_size, batches, discount, target_network):
            try:
                self._history = np.load(history_file)
            except IOError:
                print('No history file.')
                self._valid = False
                return
            self._valid = True
            self._batch_size     = batch_size
            self._batches        = batches
            self._discount       = discount
            self._target_network = target_network
            self._history_length = self._history['observation'].shape[0]
            self.on_epoch_end()

        def on_epoch_end(self):
            # TODO: sometimes, the history files are too large to pull into memory entirely.
            #   However, it's really slow to leave them on the disk and always be reading from the disk.
            #   Need some way to eat the cost of pulling a chunk of data out of history in, once,
            #   then generate a bunch of samples off of it.
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

        def valid(self):
            return self._valid

        def __len__(self):
            return self._batches

        def __getitem__(self, idx):
            start = idx * self._batch_size
            end   = start + self._batch_size
            return (self._X[start:end], self._Y[start:end])

    def __init__(self, name, cfg):
        self._name = name
        self._save_history = cfg('training', 'save_history')
        self._inputs = cfg('inputs')
        self._input_coding = {b:i for i,b in enumerate(self._inputs)}
        self._actions = cfg('actions')
        self._history_file = 'history/{}.npz'.format(self._name)
        self._unsaved_history = {'observation': [], 'action': [], 'reward': [], 'next_observation': []}
        self._prediction_network = Sequential()
        # Take in the blueprint as desired and the state of the world, in the same shape as the blueprint
        self._prediction_network.add(InputLayer(input_shape=(2,
                cfg('arena', 'width'),
                cfg('arena', 'height'),
                cfg('arena', 'length'),
                len(self._inputs)) ))

        # Now, load layers from config file and build them out:
        for layer_str in cfg('agent', 'layers'):
            # Don't try to process comments
            if layer_str.lstrip()[0] != '#':
                # Dangerous to use eval, but convenient for our purposes.
                new_layer = eval(layer_str.format(
                        arena_width  = cfg('arena', 'width'),
                        arena_height = cfg('arena', 'height'),
                        arena_length = cfg('arena', 'length'),
                        num_inputs   = len(cfg('inputs')),
                        num_actions  = len(cfg('actions'))
                    ))
                self._prediction_network.add(new_layer)
        # Output one-hot encoded action
        self._prediction_network.add(Dense(len(cfg('actions')), activation='softmax'))
        self._prediction_network.compile(loss='mse', optimizer='adam', metrics=[])
        self.start_episode = 0
        self._prediction_network,self.start_episode = std_load(self._name, self._prediction_network)
        self._target_network = clone_model(self._prediction_network)
        self._target_network.build(self._prediction_network.input_shape)

        self._target_update_frequency = 20
        self._iters_since_target_update = 0
        self._initial_epsilon = cfg('training', 'initial_epsilon')
        self._epsilon_decay = (cfg('training', 'final_epsilon') / self._initial_epsilon)**(1.0/cfg('training', 'num_episodes'))
        self._epsilon = self._initial_epsilon * (self._epsilon_decay**(self.start_episode + 1))
        self._discount = 0.95
        self._last_obs = None
        self._last_action = None

    def _preprocess(self, observation):
        return np.array([
            to_categorical( np.vectorize(self._input_coding.get)(observation), len(self._inputs) )
            ])

    def _preprocess_batch(self, observations):
        return to_categorical( np.vectorize(self._input_coding.get)(observations), len(self._inputs) )

    def _maybe_update_pn(self):
        if self._iters_since_target_update >= self._target_update_frequency:
            self._target_network.set_weights(self._prediction_network.get_weights())
            self._iters_since_target_update = 0

    def name(self):
        return self._name

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
            self._last_action = np.random.randint(len(self._actions))
        else:
            self._last_action = self._prediction_network.predict(one_hot_obs).argmax()

        # Update epsilon after using it for chance
        # Increment counter for target update
        self._iters_since_target_update += 1
        self._maybe_update_pn()

        return self._actions[self._last_action]

    def demo_act(self, observation):
        # Act without training model
        # Get predictions:
        preds = self.predict(observation)
        # Take the argmax of those predictions, which are keyed by action
        return max(preds.items(), key=(lambda x: x[1]))[0]

    def mission_ended(self):
        self._last_obs = None
        self._last_action = None
        self._epsilon *= self._epsilon_decay

    def reset_learning_params(self):
        self._epsilon = self._initial_epsilon

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

        def update_pn(*args, **kwargs):
            self._iters_since_target_update += batch_size
            self._maybe_update_pn()

        history_generator = RLearner.HistoryGenerator(
                history_file   = self._history_file,
                batch_size     = batch_size,
                batches        = batches,
                discount       = self._discount,
                target_network = self._target_network
            )

        if history_generator.valid():
            print('Training on history...')
            self._prediction_network.fit_generator(history_generator,
                epochs=epochs,
                callbacks=[LambdaCallback(on_batch_end=update_pn)]
                )
            print('Finished training on history.')
        else:
            print('Skipped history training.')

    def predict(self, observation):
        '''Runs the model on observation without saving to history or changing model weights.'''
        one_hot_obs = self._preprocess(observation)
        raw_pred = self._prediction_network.predict(one_hot_obs)[0]
        return dict(zip(self._actions, raw_pred))

    def predict_batch(self, observations):
        '''Runs the model on all observation without saving to history or changing model weights.
    Returns predictions as numpy matrix, one row per observation. Columns are in the order returned by self.actions().'''
        one_hot_obs = self._preprocess_batch(observations)
        return self._prediction_network.predict(one_hot_obs)

    def actions(self):
        return list(self._actions)