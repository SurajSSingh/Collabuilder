import keras
import numpy as np
import glob
import re

CHECKPOINT_DIR = 'checkpoint/'

def std_checkpoint(name):
    return keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR + name + '.epoch_{epoch:03d}.val_loss_{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True
    )

def std_load(name, model=None):
    '''Returns the model and epoch number saved under given name, with user confirmation/disambiguation. Returns None,None if no model is loaded.'''
    filepaths = sorted(glob.glob(CHECKPOINT_DIR + glob.escape(name) + '.epoch_*.hdf5'))
    if len(filepaths) <= 0:
        print("No models saved under name", name)
        i = 0
    else:
        i = None
        while i == None:
            print("Multiple models saved under name", name)
            print("  0) Do not load model")
            for i,fp in enumerate(filepaths):
                print("{:>3}) {}".format(i+1, fp))
            try:
                i = int(input("Choose model number: "))
                if i < 0 or i > len(filepaths):
                    raise ValueError()
            except ValueError:
                print("Invalid choice. Please enter the number to the left of the desired model file.")
                i = None
    if i == 0:
        print("Not loading model")
        return model, 0
    else:
        fp    = filepaths[i-1]
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
