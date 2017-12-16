from collections import defaultdict
from functools import reduce
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class BPM:
    def __init__(self, traincsv_path):
        self.transitions = None
        self.transitions_counts = None
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
        data_array = np.vstack((self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], time_arr, self.data[' Čas začetka'], self.data[' Čas zaključka'])).T
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
                verbose and print(j)
            verbose and print('---------------------')

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
        self.transitions_counts = transitions
        transitions = {
        key: {k: transitions[key][k] / transition_sums[key] if transitions[key][k] != 0 else 0 for k in transitions} for
        key in transitions}

        verbose and print(states)
        verbose and print(start_probability)
        verbose and print(transition_sums)
        for i in transitions:
            verbose and print(str(i) + " " + str(reduce(lambda x, y: x + transitions[i][y], transitions[i], 0)) + " " + str(
                transitions[i]))
        self.transitions = transitions

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

    def avg(self, verbose=False):
        avg_times = dict()
        for instance in self._instance_dict:
            for task in self._instance_dict[instance]:
                task_name = task[0]
                start_time = datetime.strptime(task[-2], ' %d/%m/%Y %H:%M:%S')
                end_time = datetime.strptime(task[-1], ' %d/%m/%Y %H:%M:%S')
                duration = (end_time - start_time).total_seconds() / 60.0
                if task_name not in avg_times:
                    avg_times[task_name] = {"duration": duration, "cnt": 0}
                else:
                    avg_times[task_name]["duration"] += duration
                    avg_times[task_name]["cnt"] += 1

        for key, val in avg_times.items():
            verbose and print(key, ": ", str(val["duration"]/val["cnt"]))

        return {key: val["duration"]/val["cnt"] for key, val in avg_times.items()}

    def visualize_confusion_mat(self):
        conf_mat = []
        norm_conf = []
        for i in self.transitions_counts.keys():
            tmp = []
            tmp_norm = []
            assert(self.transitions_counts.keys() == self.transitions_counts[i].keys())
            for j in self.transitions_counts[i]:
                tmp.append(self.transitions_counts[i][j])
                tmp_norm.append(self.transitions[i][j])
            conf_mat.append(tmp)
            norm_conf.append(tmp_norm)

        conf_mat = np.array(conf_mat)

        print(conf_mat)
        # norm_conf = []
        # for i in conf_mat:
        #     a = 0
        #     tmp_arr = []
        #     a = sum(i, 0)
        #     for j in i:
        #         tmp_arr.append(float(j) / float(a))
        #     norm_conf.append(tmp_arr)

        fig = plt.figure(figsize=(13, 25))
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = conf_mat.shape

        for x in range(width):
            for y in range(height):
                ax.annotate(str(conf_mat[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        class_labels = self.transitions.keys()
        plt.xticks(range(width), class_labels, rotation=45)
        plt.yticks(range(height), class_labels)

        plt.show()

    def predict_with_avg(self, avg_times):
        train_data = np.vstack((self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], self.data[' Čas začetka'], self.data[' Čas zaključka'])).T

        test_x = train_data[int(2 * train_data.shape[0] / 3):, :-1]
        test_y = train_data[int(2 * train_data.shape[0] / 3):, -1]

        MAE = 0
        MAE_cnt = 0
        for i, val in enumerate(test_x):
            task = val[0].strip()
            if task in avg_times:
                start_time = datetime.strptime(val[3], ' %d/%m/%Y %H:%M:%S')
                end_time = datetime.strptime(test_y[i], ' %d/%m/%Y %H:%M:%S')
                pred_dur = start_time + timedelta(minutes=avg_times[task])

                actual_dur = end_time

                MAE += abs((pred_dur - actual_dur).total_seconds() / 60.0)
                MAE_cnt += 1


        print(MAE / MAE_cnt)


if __name__ == '__main__':
    bpm = BPM('data/train.csv')
    #average_times = bpm.avg(True)
    #bpm.predict_with_avg(average_times)
    bpm.visualize_confusion_mat()
