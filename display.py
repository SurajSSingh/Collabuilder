import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Cursor

class Display:
    def __init__(self, model):
        self._model = model
        self._block_color = {
            'stone': '#4040D0',
            'agent': '#D04040'
        }

        self._world_alpha = 'D0'
        self._bp_alpha = '40'

        self._fig = plt.figure()
        self._axis = self._fig.add_subplot( 111, projection='3d' )
        self._fig.show()

    def update(self, world_model):
        bp, wd = world_model.get_observation()
        plt_bp = np.flip(bp.transpose( (0,2,1) ), 0)
        plt_wd = np.flip(wd.transpose( (0,2,1) ), 0)
        not_air = (plt_bp != 'air') | (plt_wd != 'air')
        colormap = np.full(plt_bp.shape, '#00000000')
        for block,color in self._block_color.items():
            # Set bp-only blocks to show with bp alpha, world blocks show with real alpha
            colormap[plt_bp == block] = color + self._bp_alpha
            colormap[plt_wd == block] = color + self._world_alpha

        self._axis.clear()
        self._axis.voxels(filled=not_air, facecolors=colormap)

        self._fig.canvas.flush_events()

class RewardsPlot:
    def __init__(self):
        self._data = []
        self._fig = plt.figure()
        self._axis = self._fig.add_subplot(111)
        self._line, = self._axis.plot([], [])

        self._fig.suptitle('Episode Reward during Training')
        self._axis.set_xlabel('Episode #')
        self._axis.set_ylabel('Total Reward')

        self._fig.show()

    def add(self, r):
        self._data.append(r)
        self._line.set_xdata(np.arange(len(self._data)))
        self._line.set_ydata(self._data)

        self._axis.relim()
        self._axis.autoscale_view()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
