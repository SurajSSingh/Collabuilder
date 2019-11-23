import tensorflow.keras as keras
import numpy as np
import glob
import re
import json

CHECKPOINT_DIR = 'checkpoint/'
CONFIG_DIR     = 'config/'

def std_checkpoint(name):
    return keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR + name + '.epoch_{epoch:09d}.val_loss_{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True
    )

def pick_file(name_pattern, prompt='Choose file:', none_prompt=None, failure_prompt='No matching files.'):
    filepaths = sorted(glob.glob(name_pattern))
    if len(filepaths) <= 0:
        print(failure_prompt)
        return None
    return ask_options(prompt, filepaths, none_prompt=none_prompt)

def std_load(name, model=None, load_file=None):
    '''Returns the model and epoch number saved under given name, with user confirmation/disambiguation. Returns None,None if no model is loaded.'''
    fp = (
        load_file if load_file is not None else
        pick_file(CHECKPOINT_DIR + glob.escape(name) + '.epoch_*.hdf5',
            none_prompt='Do not load model',
            failure_prompt='No models saved under name ' + name)
        )
    if fp is None:
        print('Not loading model')
        return model, 0
    else:
        epoch = int(re.match('.*\\.epoch_([0-9]+)', fp).group(1))
        print("Loading", fp)
        if model is None:
            return (keras.models.load_model(fp), epoch)
        else:
            model.load_weights(fp)
            return (model, epoch)

def persistent_model(name, default_model):
    '''Returns the model, loaded from disk if applicable, and epoch to resume training with.'''
    model,epoch = std_load(name)
    return (model,epoch) if model is not None else (default_model,0)

def chance(p):
    '''Returns True with probability p, and False otherwise.'''
    return np.random.random_sample() < p

def ask_yn(question):
    '''Asks user yes/no question, returns True for yes, False for no.'''
    s = input(question + ' (y/n) ').lower()
    while s not in {'yes', 'y', 'no', 'n'}:
        print('Invalid response.')
        s = input(question + ' (y/n) ').lower()
    return s in {'yes', 'y'}

def ask_int(prompt, min_val=None, max_val=None, default=None):
    '''Asks user for an integer, between min_val and max_val, inclusive.'''
    i = None
    while i is None:
        try:
            s = input(prompt + ('' if default is None else '({}) '.format(default)))
            i = default if (default is not None) and (s == "") else int(s)
            if (min_val is not None and i < min_val) or (max_val is not None and i > max_val):
                raise ValueError()
        except ValueError:
            print("Invalid choice. Please enter an integer", end='')
            if min_val is None:
                if max_val is None:
                    print('.')
                else:
                    print(' no more than {}.'.format(max_val))
            else:
                if max_val is None:
                    print(' at least {}.'.format(min_val))
                else:
                    print(' between {} and {} inclusive.'.format(min_val, max_val))
    return i

def ask_options(prompt, options, none_prompt=None):
    '''Asks user to select from a list of options.
If none_prompt is not None, allows user to select option 0, returning None.'''
    print(prompt)
    if none_prompt is not None:
        print("{:>3}) {}".format(0, none_prompt))
    for i,opt in enumerate(options):
        print("{:>3}) {}".format(i+1, opt))
    i = ask_int('Selection: ', int(none_prompt is None), len(options))
    return None if i == 0 else options[i-1]

def get_config(config_file, *attributes, config_dir=CONFIG_DIR, default=None):
    '''Fetches a value specified by attributes, from config_file in config_dir.'''
    try:
        if config_file[-5:] != '.json':
            config_file += '.json'
        with open(config_dir + config_file) as f:
            json_object = json.load(f)
        for a in attributes:
            json_object = json_object[a]
        return json_object
    except KeyError:
        if default is not None:
            return default
        else:
            raise

