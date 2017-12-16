from collections import defaultdict
from functools import reduce
import numpy as np
import pandas as pd
import time
from graphviz import Digraph


class BPM:
    def __init__(self, traincsv_path):
        self._parse_data(traincsv_path)
        self._instance_dict = self._structure_data()
        self._states, self._start_probability, self._transitions = self._hmm_initial_params()
        # self._max_concurrent_items()
        self._delivery_correlation()
        pass

    def _predict_delivery(self, steps):
        if steps[0] == 'Pakiranje':
            if steps[4] == ' DA':
                if steps[5] > 0:
                    return {k: 1 if k == 'Hitra dostava' else 0 for k in self._transitions[steps[0]]}
                else:
                    return {k: 1 if k == 'Osebni prevzem' else 0 for k in self._transitions[steps[0]]}
            else:
                if steps[5] > 0:
                    return {k: 1 if k == 'Hitra dostava' else 0 for k in self._transitions[steps[0]]}

        return self._transitions[steps[0]]

    def _delivery_correlation(self):
        deliveries = {'Dostava': 0, 'Hitra dostava': 0, 'Osebni prevzem': 0}
        deliveriesEmrg = deliveries.copy()
        count = 0
        countEmrg = 0
        for i in self._instance_dict:
            for j in self._instance_dict[i]:
                deliv, cnt = (deliveries, count) if j[4] == ' NE' else (deliveriesEmrg, countEmrg)
                if j[0] in deliv:
                    deliv[j[0]] += 1
                    cnt += 1
        return deliveries, deliveriesEmrg

    def transitions(self):
        return self._transitions

    def _parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def predict(self, train_csv_path):
        pass

    def _number_of_orders(self):
        column_names = [' Naročena pizza 1', ' Naročena pizza 2', ' Naročena pizza 3', ' Naročena pizza 4', ' Naročena pizza 5', ' Naročena pizza 6', ' Naročena pizza 7', ' Naročena pizza 8']
        result = []
        cc_a = {}
        for i in range(len(self.data[column_names[0]])):
            # if str(self.data['Številka instance'][i]) == "1131":
            #     print(1131)

            cc = 0
            for c in column_names:
                if str(self.data[c][i]).strip() != "":
                    cc += 1
            result.append(cc)

            # if cc == 0:
            #     pass
            # if self.data['Številka instance'][i] not in cc_a:
            #     cc_a[self.data['Številka instance'][i]] = cc
            # else:
            #     pass

        # print(cc_a)
        return np.array(result)

    def _structure_data(self, verbose=False):
        cnt = 0
        time_arr = [x.split(" ")[2] for x in self.data[' Čas prejema']]
        num_of_orders = self._number_of_orders()
        data_array = np.vstack((self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], time_arr, self.data[' Je nujno'], self.data[' Znesek dostave'], num_of_orders.T)).T
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
        orders = {}
        for i in self._instance_dict:
            priprava_time = None
            pattern = '%Y-%m-%d %H:%M:%S'
            for j in self._instance_dict[i]:
                j[0] = str(j[0]).strip()

                if j[0] == 'Priprava' and priprava_time is None:
                    priprava_time = time.mktime(time.strptime("2017-01-01 "+j[3], pattern))

                # Reduce number of order if current state is
                if j[0] == 'Pečenje':
                    current_time = time.mktime(time.strptime("2017-01-01 "+j[3], pattern))
                    approx_num = (current_time - priprava_time) / 180 if j[6] > 1 else 1
                    priprava_time = None
                    if j[1] not in orders:
                        orders[j[1]] = j[6] - 1
                    else:
                        orders[j[1]] -= 1

                j[6] = j[6] if j[1] not in orders else orders[j[1]]

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
        pattern = '%Y-%m-%d %H:%M:%S'
        for i in self._instance_dict:
            start_time = None
            end_time = None
            for j in self._instance_dict[i]:
                if j[0] == 'Počakaj' and start_time is None or j[0] == 'Prekliči naročilo' and start_time is None:
                    break
                elif j[0] == 'Priprava' and start_time is None:
                    start_time = time.mktime(time.strptime("2017-01-01 "+j[3], pattern))
                elif j[0] == 'Pečenje':
                    end_time = time.mktime(time.strptime("2017-01-01 "+j[3], pattern))

            if start_time is not None and end_time is not None:
                actions.append((start_time, True))
                actions.append((end_time, False))

        actions = sorted(actions, key=lambda x: x[0])
        # max = reduce(lambda x, y: x + (1 if y[1] is True else -1), actions, 0)
        # print(max)

        # bucket = 5 * 60
        # start_time = actions[0][0]
        # end_time = actions[len(actions)][0]

        z = 0
        z_max = 0
        for i in actions:
            z = z + 1 if i[1] else z - 1
            if z_max < z:
                z_max = z
        print(z_max)


def diagram(transitions):
    dot = Digraph(comment='BPM')
    for i in transitions:
        dot.node(i, i)

    for i in transitions:
        for j in transitions[i]:
            max = reduce(lambda x, y: transitions[i][y] if x < transitions[i][y] else x, transitions[i], 0)
            if transitions[i][j] > 0:
                if transitions[i][j] == max:
                    dot.edge(i, j, str(int(transitions[i][j] * 100)) + "%")
                else:
                    dot.edge(i, j, str(int(transitions[i][j] * 100)) + "%", style="dotted")

    dot.render('./bpm', view=True)


if __name__ == '__main__':
    bpm = BPM('data/train.csv')
    bpm_test = BPM('data/test.csv')
    test = bpm_test._instance_dict

    states = bpm._states

    header = ["Številka instance","Številka naloge","Dodajanje sestavin","Dostava","Hitra dostava","Naročilo","Osebni prevzem","Pakiranje","Pečenje","Počakaj","Prekliči naročilo","Priprava","Uredi naročilo","Zaključek","Trajanje_Dodajanje sestavin","Trajanje_Dostava","Trajanje_Hitra dostava","Trajanje_Naročilo","Trajanje_Osebni prevzem","Trajanje_Pakiranje","Trajanje_Pečenje","Trajanje_Počakaj","Trajanje_Prekliči naročilo","Trajanje_Priprava","Trajanje_Uredi naročilo","Trajanje_Zaključek"]
    results = []
    print(len(header))
    prob_h = ["Dodajanje sestavin","Dostava","Hitra dostava","Naročilo","Osebni prevzem","Pakiranje","Pečenje","Počakaj","Prekliči naročilo","Priprava","Uredi naročilo","Zaključek"]
    for k in test.keys():
        steps = test[k]
        last_step = steps[len(steps)-1]
        next_step = bpm._predict_delivery(last_step)
        propbs = [0 if a not in next_step else next_step[a] for a in prob_h]
        times = [0 for _ in prob_h]
        r = [last_step[1], last_step[2] + 1]
        r.extend(propbs)
        r.extend(times)
        results.append(r)
        pass

    a = np.asarray(results)
    a[:, 0].astype(int)
    a[:, 1].astype(int)
    np.savetxt("prediction.csv", a, delimiter=",", fmt='%d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f')
    diagram(bpm.transitions())

