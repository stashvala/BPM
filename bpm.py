import pandas as pd
import numpy as np


class BPM:

    def __init__(self, traincsv_path):
        self.parse_data(traincsv_path)

    def parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def structure_data(self):
        cnt = 0
        print(self.data.keys())
        time_arr = [x.split(" ")[2] for x in self.data[' Čas prejema']]
        tmp = np.vstack((self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], time_arr)).T
        tmp = tmp[tmp[:, 3].argsort()]
        tmp = tmp[tmp[:, 1].argsort()]

        print(tmp.shape)
        iprev = 0
        d = dict()
        tArr = [["a", "a", "a", "a"]]
        for i in tmp:
            # for i in data[' Ime koraka']:
            if (i[1] != iprev):
                npArr = np.array(tArr)
                print(npArr.shape)
                d[iprev] = npArr[npArr[:, 3].argsort()]
                tArr = []
                print("-----------------------")
                iprev = i[1]
            tArr.append(i)
            # print(i)
            cnt += 1
            if cnt > 500:
                break
        for i in d:
            for j in d[i]:
                print(j)
            print('---------------------')

    def predict(self, train_csv_path):
        pass

if __name__ == '__main__':
    bpm = BPM('data/train.csv')
    bpm.structure_data()


