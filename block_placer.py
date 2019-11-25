# This is now the core script for the project.
# It contains the logic for training models, running missions, etc.,
# as well as the basic CLI for executing those tasks.

from utils import ask_options, ask_yn, get_config
import functools

import sys
import os

from agent import RLearner
from display import Display
from curriculum import Curriculum

from train_model import train_model
from run_mission import Mission, run_mission

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    print = functools.partial(print, flush=True)

MODEL_BASE_NAME = 'simple_curriculum'
VERSION_NUMBER  = '2.5'
MODEL_NAME      = MODEL_BASE_NAME + '_v' + VERSION_NUMBER
CONFIG_FILE     = MODEL_NAME

if __name__ == '__main__':
    cfg = lambda *args, **kwargs: get_config(CONFIG_FILE, *args, **kwargs)

    modes = {
        'Training - Simulated, no Display': (True, False, True),
        'Training - Simulated w/ Display': (True, True, True),
        'Training - Real w/ Display': (True, True, False),
        'Demonstration - Simulated': (False, True, True),
        'Demonstration - Real': (False, True, False)
    }
    set_training,set_display,set_simulated = modes[ask_options('Select execution mode:', list(modes.keys()))]

    model = RLearner(MODEL_NAME, cfg)
    disp  = (Display(model) if set_display else None)
    curriculum = Curriculum(cfg, model.name())
    if set_training:
        plot_stats = ask_yn('Plot stats?')
        show_qsummary = ask_yn('Show Q-Summary?')
        if cfg('training', 'train_on_history'):
            model.train_on_history(
                batch_size = cfg('training', 'history', 'batch_size'),
                batches    = cfg('training', 'history', 'batches'),
                epochs     = cfg('training', 'history', 'epochs')
            )
        train_model(model, curriculum, cfg, initial_episode=model.start_episode, display=disp, simulated=set_simulated, plot_stats=plot_stats, show_qsummary=show_qsummary)
        print('Training complete.\n\n')


    # Turn on display, regardless of settings, for demo
    if disp is None:
        disp = Display(model)

    bp, start_pos, max_episode_time = curriculum.get_demo_mission()
    mission = Mission(
            blueprint        = bp,
            start_position   = start_pos,
            training         = False,
            action_delay     = 0.2,
            max_episode_time = max_episode_time,
            simulated        = set_simulated,
            display          = disp
        )
    run_mission(model, mission, cfg, demo=True)

    input('Press ENTER to exit...')

