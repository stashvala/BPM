import pandas as pd
import numpy as np


class BPM:

    def __init__(self, traincsv_path):
        self.parse_data(traincsv_path)

    def parse_data(self, csv_path):
        pd.read_csv(csv_path)

    def predict(self, train_csv_path):
        pass

if __name__ == '__main__':
    bpm = BPM('data/train.csv')

