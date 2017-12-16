from collections import defaultdict
from functools import reduce
import numpy as np
import pandas as pd
import time


class BPM:
    def __init__(self, traincsv_path):
        self._parse_data(traincsv_path)
        self._instance_dict = self._structure_data()
        self._hmm_initial_params()
        self._max_concurrent_items()

    def _parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def predict(self, train_csv_path):
        pass

    def _structure_data(self, verbose=False):
        cnt = 0
        time_arr = [x.split(" ")[2] for x in self.data[' Čas prejema']]
        data_array = np.vstack((self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], time_arr)).T
        data_array = data_array[data_array[:, 1].argsort()]

        previous_instance = data_array[0, 1]
        instance_dict = dict()
        instance_array = []
        for i in data_array:
            # for i in data[' Ime koraka']:
            if i[1] != previous_instance:
                npArr = np.array(instance_array)
                # sort by time, save to dict
                instance_dict[previous_instance] = npArr[npArr[:, 2].argsort()]
                instance_array = []
                previous_instance = i[1]

            instance_array.append(i)

        if verbose:
            # print out instances
            for i in instance_dict:
                for j in instance_dict[i]:
                    print(j)
                print('---------------------')

            # print unique instances
            unique_instances = set()
            task_col = 0
            for i in instance_dict:
                curr_instance = []
                for j in instance_dict[i]:
                    curr_instance.append(j[task_col])
                unique_instances.add(tuple(curr_instance))
                for step in curr_instance:
                    print(step, '->', end='')
                print()

        return instance_dict

    def _hmm_initial_params(self, verbose=False):
        states = set()
        start_probability = defaultdict(float)
        transitions = defaultdict(float)
        data_set_size = len(self._instance_dict)
        for i in self._instance_dict:
            for j in self._instance_dict[i]:
                j[0] = str(j[0]).strip()

                # Get start probabilities
                if int(j[2]) == 1:
                    start_probability[j[0]] += 1.0

                # Get all distinct states
                if j[0] not in states:
                    states.add(j[0])
                print(j)
            print('---------------------')

        start_probability = {key: start_probability[key] / data_set_size for key in start_probability}
        transitions = {key: {k: 0 for k in states} for key in states}

        number_of_transitions = 0
        for i in self._instance_dict:
            prev_state = None
            for j in self._instance_dict[i]:
                if prev_state is not None:
                    transitions[prev_state][j[0]] += 1
                    number_of_transitions += 1
                prev_state = j[0]

        transition_sums = {key: reduce(lambda x, y: transitions[key][y] + x, transitions[key], 0) for key in
                           transitions}
        transitions = {
        key: {k: transitions[key][k] / transition_sums[key] if transitions[key][k] != 0 else 0 for k in transitions} for
        key in transitions}

        print(states)
        print(start_probability)
        print(transition_sums)
        for i in transitions:
            print(str(i) + " " + str(reduce(lambda x, y: x + transitions[i][y], transitions[i], 0)) + " " + str(
                transitions[i]))

        return states, start_probability, transitions

    def _max_concurrent_items(self):
        actions = []
        pattern = '%H:%M:%S'
        for i in self._instance_dict:
            for j in self._instance_dict[i]:
                if j[0] == 'Priprava':
                    actions.append((int(time.mktime(time.strptime(j[3], pattern))), True))
                elif j[0] == 'Dodajanje sestavin':
                    actions.append((int(time.mktime(time.strptime(j[3], pattern))), False))

        # max = reduce(lambda x, y: x + (1 if y else -1, actions), 0)
        # print(max)



if __name__ == '__main__':
    bpm = BPM('data/train.csv')