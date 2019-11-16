import numpy as np

from collections import namedtuple
from run_mission import Mission
from utils import get_config

Lesson = namedtuple('Lesson', [
        'name',
        'function',
        'params'
    ])

class Curriculum:
    def __init__(self, cfg):
        self._current_level = 0
        self._lessons = [
                Lesson(
                    name     = lsn['name'],
                    function = _get_lesson_function(lsn['name']),
                    params   = lsn['params']
                ) for lsn in cfg('currciulum', 'lessons')
            ]
        self._successes = np.full(cfg('currciulum', 'observation_period'), fill_value=False)
        self._max_lesson_length = cfg('currciulum', 'max_lesson_length')
        self._arena_width = cfg('arena', 'width')
        self._arena_height = cfg('arena', 'height')
        self._arena_length = cfg('arena', 'length')
        self._current_episode = 0
        self._current_target_reward = np.inf

    def is_completed(self):
        '''Returns True if curriculum was completed successfully.'''
        return self._current_level >= len(self._lessons)

    def lesson_num(self):
        return self._current_level

    def get_mission(self, last_reward, model_reset_callback=None):
        # Run this check after finding the mission, so we have a mission to give on the last iteration
        if self._successes.all():
            # Agent has successfully completed the lesson the desired number of times.
            # Advance to the next lesson
            self._current_level += 1
            self._current_episode = 0
            self._successes.fill(False)
            if model_reset_callback is not None:
                model_reset_callback()

        if ((self._current_episode >= self._max_lesson_length) or
            (self._current_level >= len(self._lessons)) ):
            return (None, None)

        self._successes[self._current_episode % self._successes.size] = (
                last_reward >= self._current_target_reward
            )

        bp, start_pos, target_reward = self._lessons[self._current_level].function(
                arena_width  = self._arena_width,
                arena_height = self._arena_height,
                arena_length = self._arena_length,
                **self._lessons[self._current_level].params
            )
        self._current_episode += 1


        self._current_target_reward = target_reward
        return (bp, start_pos)

def _get_lesson_function(name):
    # This is where we can wire together lesson names and functions.
    #   Have something to the effect of:
    #
    # if name == 'my_lesson_function':
    #   from file_for_my_lesson import my_lesson_function
    #   return my_lesson_function
    # elif name == 'my_other_lesson_function':
    #   from file_for_my_other_lesson import my_other_lesson_function
    #   return my_other_lesson_function
    # ...
    #
    # Extend that pattern for each function you write

    if name == 'dummy_1':
        return dummy_1
    elif name == 'dummy_2':
        return dummy_2
    elif name == 'lessonA':
        return lessonA
    elif name == 'lessonB':
        return lessonB
    elif name == 'lessonC' or name == 'lessonD':
        return lessonCD
    # Final case, if nothing matches
    raise ValueError("'{}' is not a recognized function.".format(name))

def dummy_1(arena_width, arena_height, arena_length, p1, p2, **kwargs):
    print('--- dummy_1 ---')
    print('arena shape: ({}, {}, {})'.format(arena_width, arena_height, arena_length))
    print('p1 = {}; p2 = {}'.format(p1, p2))
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    bp[int(arena_width/2)][0][int(arena_length/2)] = 'stone'
    start_x = np.random.randint(arena_width)
    start_z = np.random.randint(arena_length)
    return (bp, (start_x, 0, start_z), 0.85)

def dummy_2(arena_width, arena_height, arena_length, foo, bar, **kwargs):
    print('--- dummy_2 ---')
    print('foo = {}; bar = {}'.format(foo, bar))
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    bp[int(arena_width/3)][0][int(2*arena_length/3)] = 'stone'
    start_x = np.random.randint(arena_width)
    start_z = np.random.randint(arena_length)
    return (bp, (start_x, 0, start_z), 0.85)

def lessonA(arena_width, arena_height, arena_length, **kwargs):
    # 1 block, placed randomly in the arena, with the agent nearby (within a k unit radius)
    MAX_REWARD = 1
    BUFFER = .15
    k = kwargs['k']
    center_x = arena_width//2
    center_z = arena_length//2
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    bp[np.random.randint(low=0+k,high=arena_width-k+1)][0][np.random.randint(low=0+k,high=arena_length-k+1)] = 'stone'
    return (bp, (center_x, 0, center_z), MAX_REWARD-BUFFER)

def lessonB(arena_width, arena_height, arena_length, **kwargs):
    # 1 block, placed randomly in the arena, with the agent placed randomly in the arena as well
    MAX_REWARD = 1
    BUFFER = .15
    start_x = np.random.randint(arena_width)
    start_z = np.random.randint(arena_length)
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')
    bp[np.random.randint(arena_width)][0][np.random.randint(arena_length)] = 'stone'
    return (bp, (start_x, 0, start_z), MAX_REWARD-BUFFER)


def _random_block_placement(arena_width, arena_length, agent_pos_x, agent_pos_z, num_of_block):
    ## Creates an randomly scattered block arrangement
    set_of_blocks = set()
    while len(set_of_blocks) < num_of_block:
        block_x = np.random.randint(0,arena_width)
        block_z = np.random.randint(0,arena_length)
        if block_x == agent_pos_x:
            if block_x + 1 == arena_width:
                block_x-=1
            else:
                block_x+=1
        if block_z == agent_pos_z:
            if block_z + 1 == arena_length:
                block_z-=1
            else:
                block_z+=1
        set_of_blocks.add((block_x,block_z))
    return set_of_blocks

def _organized_block_placement(arena_width, arena_length, agent_pos_x, agent_pos_z, k_val, num_of_block, org_type=None, floor_size=None, debug=False):
    set_of_blocks = set()
    ## Creates an organized arrangement of block
    ## types of arrangement include: lines, corners, floor
    org_array = ["xline","zline","blcorner","brcorner", "tlcorner","trcorner"]
    # Create and add the starting location
    # (make sure there is no conflict with the agent postion)
    if floor_size != None:
        block_x = np.random.randint(0,arena_width-(k_val*2))
        block_z = np.random.randint(0,arena_length-(k_val*2))
    else:
        block_x = np.random.randint(0+k_val,arena_width-k_val)
        block_z = np.random.randint(0+k_val,arena_length-k_val)
    if block_x == agent_pos_x:
        if block_x + 1 == arena_width:
            block_x-=1
        else:
            block_x+=1
    if block_z == agent_pos_z:
        if block_z + 1 == arena_length:
            block_z-=1
        else:
            block_z+=1
    set_of_blocks.add((block_x,block_z))

    curr_step = 1;
    ot = np.random.choice(org_array) if org_type == "random" else org_type
    if debug:
        print("Organization Type: {}".format(ot))
    # Line along x-axis
    if ot == "xline":
        while len(set_of_blocks) < num_of_block:
            set_of_blocks.add((block_x,(block_z+curr_step)%arena_length))
            set_of_blocks.add((block_x,(block_z-curr_step)%arena_length))
            curr_step += 1
    # Line along z-axis
    elif ot == "zline":
        while len(set_of_blocks) < num_of_block:
            set_of_blocks.add(((block_x+curr_step)%arena_width,block_z))
            set_of_blocks.add(((block_x-curr_step)%arena_width,block_z))
            curr_step += 1
    # Bottom Left Corner
    elif ot == "blcorner":
        while len(set_of_blocks) < num_of_block:
            set_of_blocks.add((block_x,(block_z+curr_step)%arena_length))
            set_of_blocks.add(((block_x-curr_step)%arena_width,block_z))
            curr_step += 1
    # Bottom Right Corner
    elif ot == "brcorner":
        while len(set_of_blocks) < num_of_block:
            set_of_blocks.add((block_x,(block_z-curr_step)%arena_length))
            set_of_blocks.add(((block_x-curr_step)%arena_width,block_z))
            curr_step += 1
    # Top Left Corner
    elif ot == "tlcorner":
        while len(set_of_blocks) < num_of_block:
            set_of_blocks.add((block_x,(block_z+curr_step)%arena_length))
            set_of_blocks.add(((block_x+curr_step)%arena_width,block_z))
            curr_step += 1
    # Top Right Corner
    elif ot == "trcorner":
        while len(set_of_blocks) < num_of_block:
            set_of_blocks.add((block_x,(block_z-curr_step)%arena_length))
            set_of_blocks.add(((block_x+curr_step)%arena_width,block_z))
            curr_step += 1
    # MxN floor (starting postion is at the lower left hand corner)
    elif ot == "floor" and floor_size != None:
        for x_val in range(0,floor_size[0]):
            for z_val in range(0,floor_size[1]):
                set_of_blocks.add(((block_x+z_val)%arena_width,(block_z+x_val)%arena_length))
    else:
        # Could not find organization type, so just do random
        return _random_block_placement(arena_width, arena_length, agent_pos_x, agent_pos_z, num_of_block)
    return set_of_blocks

def lessonCD(arena_width, arena_height, arena_length, **kwargs):
    ## Create a multi-block lesson, maybe organized or unorganized

    # Create an empty arena blueprint
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')

    # Randomize start
    start_x = np.random.randint(0,arena_width)
    start_z = np.random.randint(0,arena_length)

    # Get number of blocks and initalize x and z sums
    number_of_block = kwargs['n_blocks'] if 'n_blocks' in kwargs else 5
    x_sum = 0
    z_sum = 0

    # Create and place the blocks
    # Positions = (x,z); no y since single layer
    if 'organized' in kwargs:
        fsx = kwargs['floor_size_x'] if 'floor_size_x' in kwargs else (number_of_block+1)//2
        fsz = kwargs['floor_size_z'] if 'floor_size_z' in kwargs else (number_of_block+1)//2
        k_value = kwargs['k'] if 'k' in kwargs else 0
        positions = _organized_block_placement(arena_width,arena_length, start_x, start_z, k_value, number_of_block,
        org_type=kwargs['organized'], floor_size=(fsx,fsz), debug=False)
        for pos in positions:
            bp[pos[0]][0][pos[1]] = 'stone'
    else:
        positions = _random_block_placement(arena_width,arena_length, start_x, start_z, number_of_block)
        for pos in positions:
            bp[pos[0]][0][pos[1]] = 'stone'
            x_sum += (abs(start_x - pos[0])-1)
            z_sum += (abs(start_z - pos[1])-1)

    # Find most optimal route
    # Currently hardcoded cost
    movement_cost = 0.01
    placement_cost = 1
    optimum = 1#((x_sum*movement_cost) - (x_sum//movement_cost)) + ((z_sum*movement_cost) - (z_sum//movement_cost)) + (number_of_block*placement_cost)

    # Allow near optimal buffer
    buff = 'buff' if 'buff' in kwargs else 0.5
    buff_opt = optimum - buff

    # Debug Print Statements
    if 'debug' in kwargs:
      print('arena shape: ({}, {}, {})'.format(arena_width, arena_height, arena_length))
      print('agent start: ({}, 0, {})'.format(start_x,start_z))
      print('blocks position: {}'.format(positions))
      print('buffered optimal: between {} and {}'.format(buff_opt, optimum))
      print('blueprint:\n{}'.format(bp))

    return (bp, (start_x,0,start_z), buff_opt)

def lessonS(arena_width, arena_height, arena_length, **kwargs):
    ## Create a single tower lesson

    # Create an empty arena blueprint
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')

    # Randomize start
    start_x = np.random.randint(0,arena_width)
    start_z = np.random.randint(0,arena_length)

    # Randomly choose a position in the arena
    # and make it a tower of some height
    # bounded by min_height and max_height
    min_height = kwargs['min_h'] if 'min_h' in kwargs else 1
    max_height = kwargs['max_h'] if 'max_h' in kwargs else arena_height
    block_x = np.random.randint(0,arena_width)
    block_y = np.random.randint(min_height,max_height)
    block_z = np.random.randint(0,arena_length)

    # Offset if tower and agent share the same position
    if block_x == start_x:
        if block_x + 1 == arena_width:
            block_x-=1
        else:
            block_x+=1
    if block_z == start_z:
        if block_z + 1 == arena_length:
            block_z-=1
        else:
            block_z+=1

    # Add the stone section to the blueprint
    current_height = 0
    while current_height <= block_y:
        bp[block_x][current_height][block_z] = 'stone'
        current_height+=1

    # Find most optimal route
    # Currently hardcoded cost
    movement_cost = 1
    placement_cost = 2
    optimum = 1#(abs(block_x-(start_x-1))*movement_cost) + (abs(block_z-(start_z-1))*movement_cost) + (block_y*placement_cost)

    # Allow near optimal buffer
    buff = 'buff' if 'buff' in kwargs else 0.5
    buff_opt = optimum - buff

    # Debug Print Statements
    if 'debug' in kwargs:
      print('arena shape: ({}, {}, {})'.format(arena_width, arena_height, arena_length))
      print('agent start: ({}, 0, {})'.format(start_x,start_z))
      print('height range: ({}, {})'.format(min_height,max_height))
      print('tower postion: ({}, [0:{}], {})'.format(block_x,block_y,block_z))
      print('buffered optimal: between {} and {}'.format(buff_opt, optimum))
      print('blueprint:\n{}'.format(bp))

    return (bp, (start_x,0,start_z), buff_opt)
