import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Cursor

import sys
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

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
        bp, wd = world_model.get_full_observation()
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
        while len(self._data) >= self._max_data:
            self._downsample_factor *= 2
            if len(self._data) % 2 == 1:
                # If the data has odd length, take one data point off the end
                #   and stuff it into the temp buffer at the beginning,
                #   since anything on the data list comes before things already in
                #   the temp buffer.
                self._temp.insert(0, self._data.pop())
            self._data = list(np.reshape(self._data, (-1, 2)).mean(axis=1))
        while len(self._temp) >= self._downsample_factor:
            self._data.append(np.mean(self._temp[:self._downsample_factor]))
            self._temp = self._temp[self._downsample_factor:]
        self._draw()

    def _draw(self):
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

    def close(self):
        plt.close(self._fig)

class QSummary:
    """Displays a summary of important Q-values, as estimated by a model."""
    # This class draws heavily from tutorial_6 of the Malmo distribution.
    def __init__(self, archetypes, model, scale=60):
        # Offsets in points:
        self._text_margin   = 10
        self._bar_margin    = 5
        self._left_offset   = 200
        self._right_offset  = 50
        self._top_offset    = 120
        self._bottom_offset = 30

        self._archetypes = archetypes
        self._model = model
        self._worlds = np.array([a.world for a in archetypes])
        self._actions = model.actions()
        self._scale = scale
        self._root = tk.Tk()
        self._root.wm_title("Archetype Q-Values")
        self._canvas = tk.Canvas(self._root,
            width = len(self._actions)*scale + self._left_offset + self._right_offset,
            height = len(archetypes)*scale + self._top_offset + self._bottom_offset,
            borderwidth = 0, highlightthickness = 0, bg = "black")
        self._canvas.grid()
        self._root.update()

        self._max_y_pts = scale * len(self._archetypes) + self._top_offset
        self._max_x_pts = scale * len(self._actions) + self._left_offset

        self.update()

    def update(self):
        raw_q = self._model.predict_batch(self._worlds)
        pos_q = raw_q > 0
        mag_q = np.abs(raw_q)
        cell_heights = (self._scale - 2 * self._bar_margin) * (mag_q / mag_q.sum(axis=1).reshape(-1, 1))

        self._canvas.delete('all')
        for y in range(self._top_offset, self._max_y_pts+1, self._scale):
            self._canvas.create_line(self._left_offset, y, self._max_x_pts, y, fill='white')
        for x in range(self._left_offset, self._max_x_pts+1, self._scale):
            self._canvas.create_line(x, self._top_offset, x, self._max_y_pts, fill='white')

        for y,arch in zip(range(self._top_offset + self._scale//2, self._max_y_pts, self._scale), self._archetypes):
            self._canvas.create_text(self._left_offset - self._text_margin, y, anchor='e', text=arch.name, font=('Arial', 20), fill='white')
        for x,action in zip(range(self._left_offset + self._scale//3, self._max_x_pts, self._scale), self._actions):
            self._canvas.create_text(x, self._top_offset - self._text_margin, anchor='w', text=action.title(), font=('Arial', 20), fill='white', angle=30)

        for y,row,arch,sign_row in zip(range(self._top_offset + self._scale, self._max_y_pts + self._scale + 1, self._scale), cell_heights, self._archetypes, pos_q):
            for x,ch,action,positive in zip(range(self._left_offset, self._max_x_pts + 1, self._scale), row, self._actions, sign_row):
                self._canvas.create_rectangle(
                    x + self._bar_margin,
                    y - self._bar_margin,
                    x + self._scale - self._bar_margin,
                    y - self._bar_margin - ch,
                    fill = {
                        (True , True ): '#22CC22',
                        (True , False): '#114411',
                        (False, True ): '#CC2222',
                        (False, False): '#441111'
                    }[(arch.optimal_action == action, positive)])

    def close(self):
        self._root.destroy()

class TextDisplay:
    def __init__(self, params, title='Display'):
        '''Constructs a text display, where params is a dictionary.
Keys are labels, values are functions that generate an updated value to display.'''
        self._labels = {}

        self._root = tk.Tk()
        self._root.wm_title(title)
        self._root.configure(background='black')

        tk.Label(self._root, text=title, font=('Arial', 24), background='black', foreground='white').grid(row=0, columnspan=2)
        for i,(label,fn) in enumerate(params.items(), 1):
            tk.Label(self._root, text=label, font=('Arial', 20), background='black', foreground='white').grid(row=i, column=0, sticky=tk.W)
            text_var = tk.StringVar(self._root)
            text_var.set('---')
            tk.Label(self._root, textvariable=text_var, font=('Arial', 20), background='black', foreground='white').grid(row=i, column=1, sticky=tk.E)
            self._labels[label] = (fn, text_var)

        self._root.update()

    def update(self):
        for fn, text_var in self._labels.values():
            text_var.set(fn())
        self._root.update()

    def close(self):
        self._root.destroy()
