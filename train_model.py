import numpy as np

from run_mission import Mission, run_mission
from utils import get_config

def train_model(model, curriculum, cfg, initial_episode=0, display=None, simulated=True, plot_stats=False, show_qsummary=False, stats_filename = None, max_lesson=None):
    if not stats_filename:
        stats_filename = 'stats/' + model.name() + '.csv'
    stats_header = 'Lesson Number,Episode Number,Episode Reward,Episode Length'

    try:
        stats_data = np.genfromtxt(stats_filename,
            dtype=['int','int','float','float'],
            delimiter=',',
            names=True)[:initial_episode]
        rp_data = stats_data['Episode_Reward']
        lp_data = stats_data['Episode_Length']
        separators = np.where(np.diff(stats_data['Lesson_Number']))[0]
        np.savetxt(stats_filename, stats_data, 
            fmt='%d,%d,%f,%f',
            header=stats_header,
            comments='')
        del stats_data
        new_file = False
    except (OSError, FileNotFoundError): # File not found
        rp_data = []
        lp_data = []
        separators = []
        new_file = True
        
    if plot_stats:
        from display import LivePlot
        rp = LivePlot('Episode Reward during Training', '# Episodes', 'Total Reward', start_data = rp_data, separators=separators)
        lp = LivePlot('Episode Length during Training', '# Episodes', 'Length (s)', start_data = lp_data, separators=separators)
    del rp_data, lp_data, separators

    if show_qsummary:
        from display import QSummary
        from archetypes import StandardArchetypes
        qsummary = QSummary(StandardArchetypes(
                cfg('arena', 'width'),
                cfg('arena', 'height'),
                cfg('arena', 'length'),
            ), model)

    if plot_stats or show_qsummary:
        from display import TextDisplay
        text_display = TextDisplay({
                "Lesson #" : (lambda: '{}'.format(curriculum.lesson_num())),
                "Episode #" : (lambda: '{}'.format(curriculum.episode_num())),
                "Total Episodes": (lambda: '{}'.format(episode_num)),
                "Epsilon"  : (lambda: '{:.1f}%'.format(100*model.epsilon())),
                "Pass Rate": (lambda: '{:.1f}%'.format(100*curriculum.pass_rate()))
            }, title=model.name())

    def reset_fn(*args, **kwargs):
        model.reset_learning_params(*args, **kwargs)
        if plot_stats:
            rp.add_sep()
            lp.add_sep()

    last_reward = -np.inf
    episode_num = initial_episode
    action_delay = (0 if simulated else 0.2 / cfg('training', 'overclock_factor'))
    save_frequency   = cfg('training', 'save_frequency')
    text_display.update()

    with open(stats_filename, 'a') as stats_file:
        if new_file:
            print(stats_header, file=stats_file, flush=True)
        if not max_lesson:
            bp, start_pos, max_episode_time = curriculum.get_mission(last_reward, reset_fn)
        else:
            bp, start_pos, max_episode_time = curriculum.get_mission(last_reward, reset_fn, max_lesson=max_lesson)
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
            if show_qsummary:
                qsummary.update()
            if plot_stats or show_qsummary:
                text_display.update()

            print('{},{},{},{}'.format(
                    curriculum.lesson_num(),
                    episode_num,
                    mission_stats.reward,
                    mission_stats.length),
                file=stats_file, flush=True)
            if episode_num % save_frequency == 0:
                save_id = 'epoch_{:09d}'.format(episode_num)
                model.save(save_id)
                curriculum.save(save_id)
            bp, start_pos, max_episode_time = curriculum.get_mission(last_reward, reset_fn)
        save_id = 'epoch_{:09d}'.format(episode_num)
        model.save(save_id)
        curriculum.save(save_id)

    if curriculum.is_completed():
        print("Agent completed curriculum.")
    else:
        print('Agent was unable to complete curriculum lesson {}.'.format(curriculum.lesson_num()))
