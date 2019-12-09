# Headless version of the block_placer script.
# It contains the logic for training models, running missions, etc.,
# but uses passed arguments instead of an interactive CLI

if __name__ == '__main__':
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description='Execute a single model, for training or demonstration.')
    parser.add_argument('model_name', help='Name of the model, including version number. config/<model_name>.json will be read for settings.')
    parser.add_argument('-e', '--demo', help="Execute in demonstration mode, don't train model.", dest='training', action='store_false')
    parser.add_argument('-r', '--real', '--malmo', help='Connect to "real" (Malmo) environment, rather than running simulation.', dest='simulated', action='store_false')
    parser.add_argument('-d', '--display', help='Show world model display.', dest='display', action='store_true')
    parser.add_argument('-p', '--plot', help='Plot reward & length while training.', action='store_true')
    parser.add_argument('-q', '--qsummary', help='Show QSummary while training.', action='store_true')
    parser.add_argument('-l', '--latest', help='Automatically select latest model to restart from.', action='store_true')
    parser.add_argument('--intra-op-threads', type=int, dest='intra', default=multiprocessing.cpu_count())
    parser.add_argument('--inter-op-threads', type=int, dest='inter', default=2)
    options = parser.parse_args()

    # Delay importing expensive packages until after args are parsed.
    # This makes the "help" message more responsive, and generally avoids unnecessary work.
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(options.intra)
    tf.config.threading.set_inter_op_parallelism_threads(options.inter)

    from utils import get_config
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

    cfg = lambda *args, **kwargs: get_config(options.model_name, *args, **kwargs)

    model = RLearner(options.model_name, cfg, auto_latest=options.latest)
    disp  = (Display(model) if options.display else None)
    curriculum = Curriculum(cfg, model.name(), auto_latest=options.latest)
    if options.training:
        if cfg('training', 'train_on_history'):
            model.train_on_history(
                batch_size = cfg('training', 'history', 'batch_size'),
                batches    = cfg('training', 'history', 'batches'),
                epochs     = cfg('training', 'history', 'epochs')
            )
        train_model(model, curriculum, cfg, initial_episode=model.start_episode, display=disp, simulated=options.simulated, plot_stats=options.plot, show_qsummary=options.qsummary)
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
            simulated        = options.simulated,
            display          = disp
        )
    run_mission(model, mission, cfg, demo=True)

    input('Press ENTER to exit...')

