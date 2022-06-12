import csv
import numpy as np
import pandas as pd


class GroundTruth:
    def __init__(self, file_path):
        self.file_path = file_path
        data = self._read_file()
        self.dataframe = pd.DataFrame(data, columns=['timestamp', 'x', 'y', 'theta'])
        self.dataframe['timestamp'] = self.dataframe['timestamp'].astype('int64')
        self.dataframe['x'] = self.dataframe['x'].astype('float64')
        self.dataframe['y'] = self.dataframe['y'].astype('float64')
        self.dataframe['theta'] = self.dataframe['theta'].astype('float64')

    def _read_file(self):
        data = []
        with open(self.file_path, newline='') as csvfile:
            gt_reader = csv.reader(csvfile, delimiter=',')
            for row in gt_reader:
                data.append(row)
        
        return data

    def get_times(self):
        start = self.dataframe.iloc[0]['timestamp']
        stop = self.dataframe.iloc[-1]['timestamp']

        return start, stop

    def get(self, t):
        xval = np.interp(t, self.dataframe['timestamp'], self.dataframe['x'])
        yval = np.interp(t, self.dataframe['timestamp'], self.dataframe['y'])
        thetaval = np.interp(t, self.dataframe['timestamp'], self.dataframe['theta'])

        return np.array([xval, yval, thetaval])
