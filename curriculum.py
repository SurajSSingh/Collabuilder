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

