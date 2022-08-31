import json
import os

class SimpleLog():

    def __init__(self):
        self.d = {}

    def add(self, variable_name, value):
        if not variable_name in self.d:
            self.d[variable_name] = []
        self.d[variable_name].extend(value)

    def clear(self):
        self.d = {}

class NoLog():

    def __init__(self):
        pass

    def add(self, variable_name, value):
        pass

    def save(self):
        pass

    def clear(self):
        pass

class Log():

    def __init__(self, filename, save_freq=1):
        self.filename = filename
        self.d = {}
        self.save_freq = save_freq
        self.freq_cnt = 0

    def add(self, variable_name, value):
        if not variable_name in self.d:
            self.d[variable_name] = []
        self.d[variable_name].append(value)
        print(self.d)

    def save(self):
        self.freq_cnt += 1
        if self.freq_cnt >= self.save_freq:
            with open(self.filename, 'w') as f:
                json.dump(self.d, f)
            self.freq_cnt = 0

    def clear(self):
        self.d = {}

class ListOfLogs():

    def __init__(self, filename, separate_files=False, max_files=20):
        self.filename = filename
        self.separate_files = separate_files
        self.max_files = max_files
        self.logs = [{}]
        if self.separate_files:
            #assert not os.path.isdir(filename)
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
            self.num_logs = 0

    def add(self, variable_name, value):
        if not variable_name in self.logs[-1]:
            self.logs[-1][variable_name] = []
        self.logs[-1][variable_name].extend(value)

    def finish_log(self):
        if self.separate_files:
            with open(self.filename + '/' + str(self.num_logs).zfill(6) + '.json', 'w') as f:
                json.dump(self.logs[0], f)
            if self.num_logs >= self.max_files:
                os.remove(self.filename + '/' + str(self.num_logs - self.max_files).zfill(6) + '.json')
            self.num_logs += 1
            self.logs = [{}]

        else:
            with open(self.filename, 'w') as f:
                json.dump(self.logs, f)
            self.logs.append({})

class DataCollector():

    def __init__(self, filename, pack_size, packs_to_keep):
        self.filename = filename
        self.pack_size = pack_size
        self.packs_to_keep = packs_to_keep

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.all_states = []
        self.all_actions = []
        self.ep_states = []
        self.ep_actions = []
        self.pack_num = 0
        self.eps_in_curr_pack = 0

    def new_state(self, state):
        self.ep_states.append(state.copy().tolist())

    def new_action(self, action):
        self.ep_actions.append(action.copy().tolist())


    def ep_done(self):
        self.all_states.append(self.ep_states.copy())
        self.all_actions.append(self.ep_actions.copy())
        self.ep_states = []
        self.ep_actions = []

        self.eps_in_curr_pack += 1
        if self.eps_in_curr_pack == self.pack_size:
            data = {'states': self.all_states, 'actions': self.all_actions}
            with open(self.filename + '/pack_' + str(self.pack_num).zfill(6) + '.json', 'w') as f:
                json.dump(data, f)
            self.all_states = []
            self.all_actions = []
            self.eps_in_curr_pack = 0

            self.pack_num += 1
            if self.pack_num > self.packs_to_keep:
                os.remove(self.filename + '/pack_' + str(self.pack_num - self.packs_to_keep - 1).zfill(6) + '.json')
