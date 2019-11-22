# Collection of "archetype" worlds that are useful for evaluating the progress of an agent.
import numpy as np
from collections import namedtuple

# Note that the Archetype.world is really an observation to the agent,
# which for us is the blueprint + real world
Archetype = namedtuple('Archetype', ['world', 'name', 'optimal_action'])

def JustInFront(arena_width, arena_height, arena_length):
    observation = np.full((2, arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    # Add a blueprint block in the middle
    observation[0, arena_width//2, 0, arena_length//2] = 'stone'
    # Add the agent just in front of that block
    observation[1, arena_width//2, 0, (arena_length//2) - 1] = 'agent'
    return Archetype(observation, 'Just in Front', 'use')

def OneStepAway(arena_width, arena_height, arena_length):
    observation = np.full((2, arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    # Add a blueprint block in the middle
    observation[0, arena_width//2, 0, (arena_length//2) + 1] = 'stone'
    # Add the agent two blocks in front of that block
    observation[1, arena_width//2, 0, (arena_length//2) - 1] = 'agent'
    return Archetype(observation, '1 Step Away', 'jumpmove 1')

def FacingLeft(arena_width, arena_height, arena_length):
    observation = np.full((2, arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    # Add a blueprint block in the middle
    observation[0, arena_width//2, 0, arena_length//2] = 'stone'
    # Add the agent beside that block
    observation[1, (arena_width//2) + 1, 0, arena_length//2] = 'agent'
    return Archetype(observation, 'Facing Left', 'turn 1')

def FacingRight(arena_width, arena_height, arena_length):
    observation = np.full((2, arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    # Add a blueprint block in the middle
    observation[0, arena_width//2, 0, arena_length//2] = 'stone'
    # Add the agent beside that block
    observation[1, (arena_width//2) - 1, 0, arena_length//2] = 'agent'
    return Archetype(observation, 'Facing Right', 'turn -1')

def StandardArchetypes(arena_width, arena_height, arena_length):
    return [f(arena_width, arena_height, arena_length) for f in [
        JustInFront,
        OneStepAway,
        FacingLeft,
        FacingRight
    ]]
