import sys
import os
import numpy as np

from display import LivePlot

def plot_file(stats_filename):
    stats_data = np.genfromtxt(stats_filename,
        dtype=['int','int','float','float'],
        delimiter=',',
        names=True)
    rp_data = stats_data['Episode_Reward']
    lp_data = stats_data['Episode_Length']
    separators = np.where(np.diff(stats_data['Lesson_Number']))[0]

    name = os.path.basename(stats_filename).rsplit('.', 1)[0]

    rp = LivePlot(f'{name} Reward during Training', '# Episodes', 'Total Reward', start_data = rp_data, separators=separators)
    lp = LivePlot(f'{name} Length during Training', '# Episodes', 'Length (s)', start_data = lp_data, separators=separators)

    input('Press ENTER to continue...')
    rp.close()
    lp.close()

def plot_directory(dir_path):
    for stats_filename in os.listdir(dir_path):
        plot_file(os.path.join(dir_path, stats_filename))

if __name__ == '__main__':
    try:
        path = sys.argv[1]
        if os.path.isdir(path):
            plot_directory(path)
        elif os.path.isfile(path):
            plot_file(path)
        else:
            print(f'{path} is neither a file nor a directory.')
    except IndexError:
        print('''Usage: display_stats stats_file_or_directory''')
