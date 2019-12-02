import os
import json
from utils import get_config, pick_file, CHECKPOINT_DIR

from agent import RLearner
from curriculum import Curriculum

from train_model import train_model
from run_mission import Mission, run_mission

MT_Version = '1.0'

class ModelData:
    def __init__(self, config_file, config_dir, output_dir, model_file=None, curriculum_file=None):
        self.name = config_file[:-5]
        self.activated = True
        self.cfg  = lambda *args, **kwargs: get_config(config_file, *args, config_dir=config_dir, **kwargs)
        self.model = RLearner(self.name, self.cfg, load_file=model_file)
        self.curriculum = Curriculum(self.cfg, self.model.name(), load_file=curriculum_file)
        self.stats_filename = output_dir+self.name+'.csv'
        self.completed = False
        print(self.name)


class ModelTester:
    #ex mt = ModelTester('Tester_Input/', 'Tester_Output/')
    def __init__(self, input_dir, output_dir, name=None, load_file=None):
        self.name = name if name is not None else input_dir.rstrip('/').split('/')[-1]
        if input_dir[-1] != '/':
            input_dir += '/'
        if output_dir[-1] != '/':
            output_dir += '/'

        save_fp = (
            load_file if load_file is not None else
            pick_file(CHECKPOINT_DIR + self.name + '*.json',
                prompt=f'Choose checkpoint for {self.name}',
                none_prompt=f'Do not load checkpoint for {self.name}.',
                failure_prompt=f'No checkpoint for {self.name}')
            )
        if save_fp is None:
            input_files = os.listdir(input_dir)
            self.modelList = []
            for file in input_files:
                #glob pattern check here
                if file[-5:] == '.json':
                    self.modelList.append(ModelData(
                        file, input_dir, output_dir,
                        model_file = False,
                        curriculum_file = False
                        ))
            self.summary = []
            # Records the next 
            self.current_lesson = 0
            self.current_model  = 0
        else:
            with open(save_fp) as f:
                saved_data = json.load(f)
                self.modelList = []
                for saved_md in saved_data['model_list']:
                    md = ModelData(
                        saved_md['name'] + '.json', 
                        input_dir, output_dir,
                        model_file = saved_md['model_file'],
                        curriculum_file = saved_md['curriculum_file']
                        )
                    md.activated = saved_md['activated']
                    md.completed = saved_md['completed']
                    self.modelList.append(md)
                self.summary = saved_data['summary']
                self.current_lesson = saved_data['current_lesson']
                self.current_model = saved_data['current_model']

    def train(self):
        

        while any(m.activated for m in self.modelList):
            while self.current_model < len(self.modelList):
                self.save(f'lesson_{self.current_lesson:03d}.model_{self.current_model:03d}')
                m = self.modelList[self.current_model]
                if m.activated:
                    #train for one lesson and output somewhere
                    train_model(m.model, m.curriculum, m.cfg, stats_filename=m.stats_filename, max_lesson=self.current_lesson)
                self.current_model += 1

            for m in self.modelList:
                if m.curriculum.lesson_num() <= self.current_lesson or m.curriculum.is_completed():
                    m.activated = False
                    m.completed = m.curriculum.is_completed()
                    self.summary.append(f'Model {m.name} deactivated during round {self.current_lesson}\nCompleted: {m.completed}')

            # analyze aggregate population stats and deactivate poor performers
            self.current_lesson+=1
            self.current_model = 0
            self.save(f'lesson_{self.current_lesson:03d}.model_{self.current_model:03d}')

        for s in self.summary:
            print(s)

    def save(self, id=None):
        # Build the json object we're going to save.
        # In the process, save out all models and curriculums
        print(f'Saving model tester {self.name}...')
        json_obj = {
            'name': self.name,
            'model_list': [
                {
                    'name': md.name,
                    'activated': md.activated,
                    'completed': md.completed,
                    'model_file': md.model.save(id=f'{id}.epoch_{md.curriculum.episode_num():09d}'),
                    'curriculum_file': md.curriculum.save(id=id)
                }
                for md in self.modelList
            ],
            'summary': self.summary,
            'current_lesson': self.current_lesson,
            'current_model': self.current_model
        }
        filepath = CHECKPOINT_DIR + self.name + ('' if id is None else '.' + id) + '.json'
        with open(filepath, 'w+') as f:
            json.dump(json_obj, f)
        print(f'Finished saving model tester {self.name}.')
        return filepath
