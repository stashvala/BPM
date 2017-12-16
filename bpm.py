import numpy as np
import pandas as pd


class BPM:

    def __init__(self, traincsv_path):
        self.parse_data(traincsv_path)

    def parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def predict(self, train_csv_path):
        pass

    def structure_data(self, print_data=True):
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

        if print_data:
            # print out instances
            for i in instance_dict:
                for j in instance_dict[i]:
                    print(j)
                print('---------------------')

        self.transition_matrix(instance_dict)

    def transition_matrix(self, instance_dict):
        transitions = dict()

        task_col = 0

        for instance in instance_dict:
            prev_state = None
            for task in instance_dict[instance]:
                task_name = task[task_col]
                if not prev_state:
                    prev_state = task_name
                else:
                    if prev_state not in transitions:
                        transitions[prev_state] = {"next": task_name, "cnt": 1}
                    else:
                        transitions[prev_state]["cnt"] += 1 
                    prev_state = task_name

        print(transitions)




if __name__ == '__main__':
    bpm = BPM('data/train.csv')
    bpm.structure_data(False)

