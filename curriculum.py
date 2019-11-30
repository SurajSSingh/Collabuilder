import numpy as np
import json

from collections import namedtuple
from run_mission import Mission
from utils import CHECKPOINT_DIR, pick_file

Lesson = namedtuple('Lesson', [
        'name',
        'function',
        'params',
        'max_episodes',
        'max_episode_time',
        'set_learning_schedule'
    ])

class Curriculum:
    def __init__(self, cfg, name, load_file=None):
        '''Create a Curriculum based on given config.
If load_file is a valid file path, read from that save file instead.
If load_file is None, look for save files with correct name, and ask user to load from those.
If load_file is False, do not look for save files.'''
        self._name = 'curriculum.' + name
        self._max_lesson_length = cfg('curriculum', 'max_lesson_length')
        self._default_episode_time = cfg('training', 'num_episodes')
        self._lessons = [
                Lesson(
                    name     = lsn['name'],
                    function = _get_lesson_function(lsn['name']),
                    params   = lsn['params'],
                    max_episodes = lsn.get('max_episodes', self._max_lesson_length),
                    max_episode_time = lsn.get('max_episode_time', self._default_episode_time),
                    set_learning_schedule = lsn.get('set_learning_schedule', False)
                ) for lsn in cfg('curriculum', 'lessons')
            ]
        self._max_lesson_length = cfg('curriculum', 'max_lesson_length')
        self._arena_width = cfg('arena', 'width')
        self._arena_height = cfg('arena', 'height')
        self._arena_length = cfg('arena', 'length')
        self._current_target_reward = np.inf
        #WIP, can use this to calculate some aggregate stats in future
        self._episode_summary = []

        # Load these from save file, if possible
        save_fp = (
            None if load_file is False else
            load_file if load_file is not None else
            pick_file(CHECKPOINT_DIR + self._name + '*.json',
                prompt=f'Choose curriculum checkpoint for {self._name}',
                none_prompt=f'Do not load curriculum checkpoint for {self._name}.',
                failure_prompt=f'No curriculum checkpoint for {self._name}.')
            )
        if save_fp is None:
            # Start off before the first lesson, to correctly trigger model resetting for the first lesson
            self._successes       = np.full(cfg('curriculum', 'observation_period'), fill_value=True)
            self._current_level   = -1
            self._current_episode = 0
        else:
            with open(save_fp) as f:
                saved_data = json.load(f)
                self._successes = np.array(saved_data['successes'])
                self._current_level = saved_data['current_level']
                self._current_episode = saved_data['current_episode']

    def save(self, id):
        filepath = CHECKPOINT_DIR + self._name + id + '.json'
        with open(filepath, 'w') as f:
            json.dump({
                    'current_level': self._current_level,
                    'current_episode': self._current_episode,
                    'successes': self._successes.tolist()
                }, f)
        return filepath

    def is_completed(self):
        '''Returns True if curriculum was completed successfully.'''
        return self._current_level >= len(self._lessons)

    def lesson_num(self):
        return self._current_level

    def episode_num(self):
        '''Episode number for this lesson only.'''
        return self._current_episode

    def pass_rate(self):
        '''Returns the fraction of lessons passed in the most recent observation period.'''
        return self._successes.mean()

    def get_mission(self, last_reward, model_reset_callback=None, max_lesson=None):
        # Run this check after finding the mission, so we have a mission to give on the last iteration
        if self._successes.all():
            # Agent has successfully completed the lesson the desired number of times.
            # Advance to the next lesson
            self._current_level += 1
            self._episode_summary.append(self._current_episode)
            self._current_episode = 0
            self._successes.fill(False)
            if model_reset_callback is not None:
                if (self._current_level < len(self._lessons) and
                    self._lessons[self._current_level].set_learning_schedule):
                    model_reset_callback(num_episodes=self._lessons[self._current_level].max_episodes)
                else:
                    model_reset_callback()

        if ((max_lesson is not None and self._current_level > max_lesson) or
            (self._current_level >= len(self._lessons)) or
            (self._current_episode >= self._lessons[self._current_level].max_episodes)):
            return (None, None, None)

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

        max_episode_time = self._lessons[self._current_level].max_episode_time
        self._current_target_reward = target_reward
        return (bp, start_pos, max_episode_time)

    def get_demo_mission(self):
        # Return a mission without treating this as training, don't ask for or record rewards
        level = max(self._current_level, len(self._lessons) - 1)
        bp, start_pos, _ = self._lessons[level].function(
                arena_width  = self._arena_width,
                arena_height = self._arena_height,
                arena_length = self._arena_length,
                **self._lessons[level].params
            )

        max_episode_time = self._lessons[self._current_level].max_episode_time
        return (bp, start_pos, max_episode_time)

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
    elif name in ['lessonC','lessonD','lessonE','lessonF','lessonG']:
        return lessonMB
    elif name == 'in_front':
        return just_in_front_lesson
    elif name == 'turn':
        return turn_lesson
    elif name == 'approach':
        return approach_lesson
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


def _random_block_placement(arena_width, arena_length, agent_pos_x, agent_pos_z, k_val, num_of_block):
    ## Creates an randomly scattered block arrangement
    set_of_blocks = set()
    while len(set_of_blocks) < num_of_block:
        block_x = np.random.randint(0+k_val,arena_width-k_val)
        block_z = np.random.randint(0+k_val,arena_length-k_val)
        set_of_blocks.add((block_x,block_z))
    return set_of_blocks

def _organized_block_placement(arena_width, arena_length, agent_pos_x, agent_pos_z, k_val, num_of_block, org_type=None, floor_size=None, debug=False):
    set_of_blocks = set()
    ## Creates an organized arrangement of block
    ## types of arrangement include: lines, corners, floor
    org_array = ["xline","zline","blcorner","brcorner", "tlcorner","trcorner"]
    # Create and add the starting location
    # use k value to pad out from the edge
    if org_type == "floor" and floor_size != None:
        block_x = np.random.randint(0,arena_width-(k_val*2))
        block_z = np.random.randint(0,arena_length-(k_val*2))
    else:
        block_x = np.random.randint(0+k_val,arena_width-k_val)
        block_z = np.random.randint(0+k_val,arena_length-k_val)
    # # Check if agent inside starting block
    # if block_x == agent_pos_x:
    #     if block_x + 1 == arena_width:
    #         block_x-=1
    #     else:
    #         block_x+=1
    # if block_z == agent_pos_z:
    #     if block_z + 1 == arena_length:
    #         block_z-=1
    #     else:
    #         block_z+=1
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
        return _random_block_placement(arena_width, arena_length, agent_pos_x, agent_pos_z, k_val, num_of_block)
    return set_of_blocks

def _tower_builder(postions, min_height=2, max_height=2, random_height=False):
    new_pos = list()
    for pos in postions:
        # If random height, make tower n-randomly high
        if random_height:
            for i in range(1,np.random.randint(min_height,max_height)):
                new_pos.append((pos[0],i,pos[1]))
        else:
            for i in range(1,max_height):
                new_pos.append((pos[0],i,pos[1]))
    return new_pos

def lessonMB(arena_width, arena_height, arena_length, **kwargs):
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
    positions = None
    if 'organized' in kwargs:
        fsx = kwargs['floor_size_x'] if 'floor_size_x' in kwargs else (number_of_block+1)//2
        fsz = kwargs['floor_size_z'] if 'floor_size_z' in kwargs else (number_of_block+1)//2
        k_value = kwargs['k'] if 'k' in kwargs else 0
        positions = _organized_block_placement(arena_width,arena_length, start_x, start_z, k_value, number_of_block,
        org_type=kwargs['organized'], floor_size=(fsx,fsz), debug=False)
        for pos in positions:
            bp[pos[0]][0][pos[1]] = 'stone'
    else:
        k_value = kwargs['k'] if 'k' in kwargs else 0
        positions = _random_block_placement(arena_width,arena_length, start_x, start_z, k_value, number_of_block)
        for pos in positions:
            bp[pos[0]][0][pos[1]] = 'stone'
            x_sum += (abs(start_x - pos[0])-1)
            z_sum += (abs(start_z - pos[1])-1)

    # Add height to blueprint if required
    if positions != None and 'tower' in kwargs:
        mx_h = kwargs['max_height'] if 'max_height' in kwargs else arena_height
        mn_h = kwargs['min_height'] if 'min_height' in kwargs else 2
        rand_height = True if 'random_height' in kwargs else False
        tower_positions = _tower_builder(positions, min_height=mn_h, max_height=mx_h, random_height=rand_height)
        for pos in tower_positions:
            bp[pos[0]][pos[1]][pos[2]] = 'stone'

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

def just_in_front_lesson(arena_width, arena_height, arena_length, target_reward=0.95, **kwargs):
    ## Creates a blueprint block right in front of the agent, with nothing to do but place it.

    # Create an empty arena blueprint
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')

    block_x = np.random.randint(0, arena_width)
    block_y = 0
    block_z = np.random.randint(1, arena_length)
    bp[block_x, block_y, block_z] = 'stone'

    return (bp, (block_x, block_y, block_z - 1), target_reward)

def turn_lesson(arena_width, arena_height, arena_length, target_reward=0.95, **kwargs):
    ## Creates a blueprint block to the side of the agent.

    # Create an empty arena blueprint
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')

    block_x = np.random.randint(1, arena_width-1)
    block_y = 0
    block_z = np.random.randint(0, arena_length)
    bp[block_x, block_y, block_z] = 'stone'

    # Choose randomly whether agent is facing left or right of block
    side = (-1)**np.random.randint(0, 2)

    return (bp, (block_x + side, block_y, block_z), target_reward)

def approach_lesson(arena_width, arena_height, arena_length, max_distance=2, target_reward=0.95, **kwargs):
    ## Creates a blueprint block a few blocks in front of the agent, with nothing to do but place it.

    # Create an empty arena blueprint
    bp = np.full((arena_width, arena_height, arena_length), fill_value='air', dtype='<U8')

    block_x = np.random.randint(0, arena_width)
    block_y = 0
    block_z = np.random.randint(max_distance, arena_length)
    bp[block_x, block_y, block_z] = 'stone'

    distance = np.random.randint(1, max_distance+1)

    return (bp, (block_x, block_y, block_z - distance), target_reward)
