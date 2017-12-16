import numpy as np
import pandas as pd


class BPM:

    def __init__(self, traincsv_path):
        self.parse_data(traincsv_path)

    def parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def predict(self, train_csv_path):
        pass

    def structure_data(self):
        cnt = 0
        time_arr = [x.split(" ")[2] for x in self.data[' Čas prejema']]
        data_array = np.vstack((self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], time_arr)).T
        data_array = data_array[data_array[:, 1].argsort()]

        previous_instance = data_array[0, 1]
        instance_dict = dict()
        instance_array = [data_array[0]]
        for i in data_array:
            # for i in data[' Ime koraka']:
            if i[1] != previous_instance:
                npArr = np.array(instance_array)
                # sort by time, save to dict
                instance_dict[previous_instance] = npArr[npArr[:, 3].argsort()]
                instance_array = []
                previous_instance = i[1]

            instance_array.append(i)

        # print out instances
        for i in instance_dict:
            for j in instance_dict[i]:
                print(j)
            print('---------------------')

if __name__ == '__main__':
    bpm = BPM('data/train.csv')
    bpm.structure_data()

