import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def one_hot_encoding(arr, classes=-1):
    arr = np.array(arr).reshape(-1)
    if classes == -1:
        classes = np.max(arr) + 1
    one_hot = np.eye(classes)[arr]
    return one_hot

class BPM:

    def __init__(self, traincsv_path):
        self.parse_data(traincsv_path)
        self.regr = None
        self.train_random_forest()

    def parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def train_random_forest(self, modelname = 'models/rf_model1'):
        if not os.path.exists(modelname):
            cnt = 0
            time_arr = [x.split(" ")[2] for x in self.data[' Čas prejema']]
            time_end_arr = [x.split(" ")[2] for x in self.data[' Čas zaključka']]

            print(self.data.keys())
            pizzas = filter(lambda x: ("Naročena" in x), self.data.keys())
            pizza_array = self.data[list(pizzas)]
            print(pizza_array.values[0])
            pizza_count = np.array([sum([1 if len(y) > 3 else 0 for y in x]) for x in pizza_array.values])
            pizza_count = pizza_count.reshape((pizza_count.shape[0], 1))
            data_array = np.vstack(
                (self.data[' Ime koraka'], self.data['Številka instance'], self.data[' Številka naloge'], time_arr, time_end_arr)).T

            pizza_set = set()
            pizza_dict = {}
            pizza_cnt = 0
            for i in pizza_array:
                for j in i:
                    pizza_set.add(j)
                    if j not in pizza_dict:
                        pizza_dict[j] = pizza_cnt
                        pizza_cnt += 1

            # train_set = pd.get_dummies(data[' Ime koraka'])
            ime_korak = pd.get_dummies(self.data[' Ime koraka'])
            prejemnik = pd.get_dummies(self.data[' Prejemnik'])
            nujno = pd.get_dummies(self.data[' Je nujno'])
            # print(data[' Razdalja']==" ")
            razdalja = np.array([float(x) if x != " " else 0.0 for x in self.data[' Razdalja']]).reshape((nujno.shape[0], 1))
            znesek = self.data[' Znesek skupaj'].reshape((nujno.shape[0], 1))
            znesek_dostava = pd.get_dummies(self.data[' Znesek dostave'])
            duration = self.data[' Čas začetka']
            zacetek = [datetime.strptime(x, ' %d/%m/%Y %H:%M:%S') for x in self.data[' Čas začetka']]
            konec = [datetime.strptime(x, ' %d/%m/%Y %H:%M:%S') for x in self.data[' Čas zaključka']]
            ura = np.array([x.hour for x in zacetek])
            ura_dum = pd.get_dummies(ura)
            ura = ura.reshape((nujno.shape[0], 1))
            duration = np.array([(x - y).total_seconds() / 60.0 for y, x in zip(zacetek, konec)]).reshape(
                (duration.shape[0], 1))

            print(ura_dum.values[0])
            print(nujno.shape)
            print(prejemnik.shape)
            print(duration.shape)
            # print(konec.shape)
            # print(zacetek.shape)
            print(znesek.shape)
            print(znesek_dostava.shape)
            print(razdalja.shape)

            train_data = np.hstack(
                (ime_korak, prejemnik, pizza_count, nujno, razdalja, znesek, znesek_dostava, ura, duration))
            print(train_data.shape)

            self.regr = RandomForestRegressor(n_estimators=10000, n_jobs=8, verbose=True, min_samples_leaf=5,
                                              oob_score=True)
            x = train_data[:, :-1]
            y = train_data[:, -1]
            self.regr.fit(x, y)

            with open(modelname, 'wb') as f:
                pickle.dump(self.regr, f)
        else:
            with open(modelname, 'rb') as f:
                self.regr = pickle.load(f)

    def predict(self, x):
        # TODO: check input format

        self.regr.predict(x)

if __name__ == '__main__':
    bpm = BPM('data/train.csv')

