import numpy as np
import random 

MAX_DEPTH = 5

def generate_blueprint(length, width, height):
    blueprint = np.full((height, length, width), fill_value = False)
    # Draw the floor
    blueprint[0,:,:] = True
    # Draw the outer walls
    blueprint[:,(-1,0),:] = True
    blueprint[:,:,(0,-1)] = True
    set_inner_walls(blueprint)
    return blueprint

def set_inner_walls(blueprint, depth=0):
    if depth > MAX_DEPTH:
        return

    options = ['n']
    if blueprint.shape[1] >= 5:
        options.append('h')
    if blueprint.shape[2] >= 5:
        options.append('v')

    mode = random.choice(options)
    if mode == 'h':
        i = random.randint(2, blueprint.shape[1]-3)
        blueprint[:,i,:] = True
        set_inner_walls(blueprint[:,:i+1,:], depth+1)
        set_inner_walls(blueprint[:,i:,:], depth+1)
    elif mode == 'v':
        i = random.randint(2, blueprint.shape[2]-3)
        blueprint[:,:,i] = True
        set_inner_walls(blueprint[:,:,:i+1], depth+1)
        set_inner_walls(blueprint[:,:,i:], depth+1)
