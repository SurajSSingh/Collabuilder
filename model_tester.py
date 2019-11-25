import os
from utils import get_config

from agent import RLearner
from curriculum import Curriculum

from train_model import train_model
from run_mission import Mission, run_mission

MT_Version = '1.0'

class ModelData:
    def __init__(self, config_file, config_dir, output_dir):
        self.name = config_file[:-5]
        self.activated = True
        self.cfg  = lambda *args: get_config(config_file, *args, config_dir=config_dir)
        self.model = RLearner(self.name, self.cfg)
        self.curriculum = Curriculum(self.cfg, self.model.name())
        self.stats_filename = output_dir+self.name+'.csv'
        self.completed = False
        print(self.name)


class ModelTester:
    #ex mt = ModelTester('Tester_Input/', 'Tester_Output/')
    def __init__(self, input_dir, output_dir):
        input_files = os.listdir(input_dir)
        self.modelList = []
        for file in input_files:
            #glob pattern check here
            if file[-5:] == '.json':
                self.modelList.append(ModelData(file, input_dir, output_dir))
        self.deactivated_count = 0
        self.summary = []
        self.current_lesson = 1

    def train(self):
        

        while self.deactivated_count != len(self.modelList):
            for m in self.modelList:
                if m.activated:
                    #train for one lesson and output somewhere
                    train_model(m.model, m.curriculum, m.cfg, stats_filename=m.stats_filename, max_lesson=self.current_lesson)

            for m in self.modelList:
                if m.curriculum._current_level == self.current_lesson or m.curriculum.is_completed():
                    m.activated = False
                    m.completed = m.curriculum.is_completed()
                    self.summary.append(f'Model {m.name} deactivated during round {current_round}\nCompleted: {m.completed}')
                    self.deactivated_count += 1

            # analyze aggregate population stats and deactivate poor performers
            self.current_lesson+=1

        for s in self.summary:
            print(s)
            


