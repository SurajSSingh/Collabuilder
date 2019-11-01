import numpy as np

from run_mission import Mission, run_mission
from utils import get_config

def train_model(model, curriculum, cfg, initial_episode=0, display=None, simulated=True, plot_stats=False):
    if plot_stats:
        from display import LivePlot
        rp = LivePlot('Episode Reward during Training', '# Episodes', 'Total Reward')
        lp = LivePlot('Episode Length during Training', '# Episodes', 'Length (s)')

    def reset_fn():
        model.reset_learning_params()
        if plot_stats:
            rp.add_sep()
            lp.add_sep()

    last_reward = -np.inf
    episode_num = initial_episode
    action_delay = (0 if simulated else 20 / cfg('training', 'overclock_factor'))
    max_episode_time = cfg('training', 'max_episode_time')
    save_frequency   = cfg('training', 'save_frequency')

    bp, start_pos = curriculum.get_mission(last_reward, reset_fn)
    while bp is not None:
        episode_num += 1
        mission = Mission(
                blueprint        = bp,
                start_position   = start_pos,
                training         = True,
                action_delay     = action_delay,
                max_episode_time = max_episode_time,
                simulated        = simulated,
                display          = display
            )
        print('Lesson {}, Episode {}'.format(curriculum.lesson_num(), episode_num))
        mission_stats = run_mission(model, mission, cfg)
        last_reward = mission_stats.reward
        print('Total reward   :', mission_stats.reward)
        print('Episode length :', mission_stats.length)
        if plot_stats:
            rp.add(mission_stats.reward)
            lp.add(mission_stats.length)
        if episode_num % save_frequency == 0:
            model.save('epoch_{:09d}'.format(episode_num))
        bp, start_pos = curriculum.get_mission(last_reward, reset_fn)
    model.save('epoch_{:09d}'.format(episode_num))

    if curriculum.is_completed():
        print("Agent completed curriculum.")
    else:
        print('Agent was unable to complete curriculum lesson {}.'.format(curriculum.lesson_num()))
