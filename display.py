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

class LivePlot:
    def __init__(self, title, xlabel, ylabel, start_data=[], separators=[]):
        self._data = list(start_data)
        self._temp = []
        self._downsample_factor = 1
        self._max_data = 600

        self._fig = plt.figure()
        self._axis = self._fig.add_subplot(111)
        self._fig.canvas.draw()

        self._line, = self._axis.plot([], [])
        self._fig.suptitle(title)
        self._axis.set_xlabel(xlabel)
        self._axis.set_ylabel(ylabel)

        self._bg = self._fig.canvas.copy_from_bbox(self._axis.bbox)

        self._fig.canvas.flush_events()

        if len(self._data) > 0:
            # pop the last data point and add it,
            # to trigger the downsampling as necessary
            # and plot the data as it stands
            last = self._data.pop()
            self.add(last)

        for s in separators:
            self._axis.axvline((s - 1), linestyle='--')


    def add(self, r):
        self._temp.append(r)
        if len(self._temp) >= self._downsample_factor:
            self._add(np.mean(self._temp))
            self._temp.clear()
        while len(self._data) >= self._max_data:
            self._downsample_factor *= 2
            self._data = list(np.reshape(self._data, (-1, 2)).mean(axis=1))

    def _add(self, r):
        self._data.append(r)
        self._line.set_xdata(np.arange(0,len(self._data)*self._downsample_factor,self._downsample_factor))
        self._line.set_ydata(self._data)

        self._axis.relim()
        self._axis.autoscale_view()
        self._fig.canvas.restore_region(self._bg)
        self._axis.draw_artist(self._line)
        self._fig.canvas.blit(self._axis.bbox)
        self._fig.canvas.flush_events()

    def add_sep(self):
        self._axis.axvline(
            (len(self._data) - 1)*self._downsample_factor,
            linestyle='--'
        )
