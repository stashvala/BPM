import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

def one_hot_encoding(arr, classes=-1):
    arr = np.array(arr).reshape(-1)
    if classes == -1:
        classes = np.max(arr) + 1
    one_hot = np.eye(classes)[arr]
    return one_hot

class BPM:

    def __init__(self, traincsv_path):
        self.parse_data(traincsv_path)

    def parse_data(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def predict(self, train_csv_path):
        pass

if __name__ == '__main__':
    bpm = BPM('data/train.csv')
    data = bpm.data
    cnt = 0
    time_arr = [x.split(" ")[2] for x in data[' Čas prejema']]
    time_end_arr = [x.split(" ")[2] for x in data[' Čas zaključka']]
    

    print(data.keys())
    pizzas = filter(lambda x: ("Naročena" in x), data.keys())
    pizza_array = data[list(pizzas)]
    print(pizza_array.values[0])
    pizza_count = np.array([sum([1 if len(y)>3 else 0 for y in x]) for x in pizza_array.values])
    pizza_count = pizza_count.reshape((pizza_count.shape[0],1))
    data_array = np.vstack((data[' Ime koraka'],data['Številka instance'], data[' Številka naloge'], time_arr, time_end_arr)).T
    

    pizza_set = set()
    pizza_dict = {}
    pizza_cnt = 0
    for i in pizza_array:
        for j in i:
            pizza_set.add(j)
            if j not in pizza_dict:
                pizza_dict[j] = pizza_cnt
                pizza_cnt+=1




    #train_set = pd.get_dummies(data[' Ime koraka'])
    ime_korak = pd.get_dummies(data[' Ime koraka'])
    prejemnik = pd.get_dummies(data[' Prejemnik'])
    nujno = pd.get_dummies(data[' Je nujno'])
    #print(data[' Razdalja']==" ")
    razdalja = np.array([float(x) if x!=" " else 0.0 for x in data[' Razdalja']]).reshape((nujno.shape[0],1))
    znesek = data[' Znesek skupaj'].reshape((nujno.shape[0],1))
    znesek_dostava = pd.get_dummies(data[' Znesek dostave'])
    duration = data[' Čas začetka']
    zacetek = [datetime.strptime(x, ' %d/%m/%Y %H:%M:%S') for x in data[' Čas začetka']]
    konec = [datetime.strptime(x, ' %d/%m/%Y %H:%M:%S') for x in data[' Čas zaključka']]
    ura = np.array([x.hour for x in zacetek])
    ura_dum = pd.get_dummies(ura)
    ura = ura.reshape((nujno.shape[0],1))
    duration = np.array([(x - y).total_seconds()/60.0 for y,x in zip(zacetek,konec)]).reshape((duration.shape[0],1))

    print(ura_dum.values[0])
    print(nujno.shape)
    print(prejemnik.shape)
    print(duration.shape)
    #print(konec.shape)
    #print(zacetek.shape)
    print(znesek.shape)
    print(znesek_dostava.shape)
    print(razdalja.shape)
    

    train_data = np.hstack((ime_korak, prejemnik, pizza_count,nujno,razdalja,znesek,znesek_dostava,ura,duration))
    print(train_data.shape)

    train_x = train_data[:int(2*train_data.shape[0]/3),:-1]
    train_y = train_data[:int(2*train_data.shape[0]/3),-1]

    test_x = train_data[int(2*train_data.shape[0]/3):,:-1]
    test_y = train_data[int(2*train_data.shape[0]/3):,-1]
    print(train_x[0,:])
    print(test_y)

    #regr = linear_model.LinearRegression()

    regr = RandomForestRegressor(n_estimators=10000, n_jobs=8, verbose=True, min_samples_leaf=5, oob_score=True)
    regr.fit(train_x, train_y)
    # Train the model using the training sets
    regr.fit(train_x, train_y)
    pred = regr.predict(test_x)

    f = sum(abs(pred-test_y))/len(test_y)
    print(f)
