import os
import glob

from pathlib import Path

import numpy as np
import cv2
from torch.utils.data import Dataset
from .ground_truth_reader import GroundTruth
from .mit_stata_center_info import mit_stata_center_get_sensor_info
from .mit_stata_center_file_manager import MitStataCenterFileManager


class MITStataCenterDataset(Dataset):
    def __init__(self, path, sequence, sensors, data_transform, ground_truth_file, download=False):
        self._path = path
        self._sensors = sensors
        self._dataTransform = data_transform
        self._sequence = sequence
        self.file_manager = MitStataCenterFileManager(path, sequence, sensors, ground_truth_file)
        self._last_frame_time = 0
        self._data = {}

        if download:
            self.file_manager.download()

        if not self.file_manager.check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        self._ground_truth = GroundTruth(os.path.join(self._path, "ground_truth", ground_truth_file + ".poses"))
        (start, stop) = self._ground_truth.get_times()

        # Find all the data files
        self._load_data(sensors, int(start), int(stop))
        self._last_frame_time = int(self._timestamps[0])

    def __len__(self):
        size = len(self._timestamps) - 1

        if size % 2 != 0:
            size = size - 1
        return size

    def __getitem__(self, idx):
        frame_time1 = float(Path(self._data['rgb'][idx][1]).stem)
        frame_time2 = float(Path(self._data['rgb'][idx + 1][1]).stem)
        value = {}

        #value['rgb'] = data
        #value['imu'] = self._data['imu'][idx + 1]
        target = np.array(self._ground_truth.get(frame_time2))
        #frame_time = int(self._timestamps[idx + 1])
        #value = {}
        #target = self._ground_truth.get(frame_time)
        for sensor in self._sensors:
            data = np.array(self._data[sensor])
            if sensor in ["left", "right", "rgb", "depth", "lidar"]:
                temp = data[np.where(data[:, 0].astype(np.uint64) >= frame_time1)]
                frame_image1 = cv2.imread(temp[0][1])
                frame_image2 = cv2.imread(temp[1][1])
                if self._dataTransform:
                    value[sensor] = [self._dataTransform(frame_image1), self._dataTransform(frame_image2)]
                else:
                    value[sensor] = [frame_image1, frame_image2]
            else:
                data = np.delete(data, np.where(data[:, 0].astype(np.uint64) > frame_time1), axis=0)
                value[sensor] = np.delete(data, 0, axis=1)[-2:]
        self._last_frame_time = frame_time2

        return value, target

    def _load_data(self, sensors, start, stop):
        for sensor in sensors:
            if sensor in ["left", "right", "rgb", "depth", "lidar"]:
                if sensor in ["left", "right", "rgb"]:
                    extension = "jpg"
                elif sensor == "depth":
                    extension = "tif"
                elif sensor == "lidar":
                    extension = "bin"
                else:
                    extension = "txt"

                """Find and list data files for the sensor."""
                temp = sorted(glob.glob(os.path.join(self.file_manager.sequence_path(), sensor, '*.{}'.format(extension))))

                if sensor in ["left", "right", "rgb", "depth"]:
                    files = []
                    self._timestamps = []
                    for i in range(len(temp)):
                        frame_time = int(float(Path(temp[i]).stem))
                        if start <= frame_time <= stop:
                            files.append([frame_time, temp[i]])
                            self._timestamps.append(frame_time)

                    self._timestamps = np.array(self._timestamps, dtype=np.uint64)

                    self._data[sensor] = files
            elif sensor == "imu":
                file = os.path.join(self.file_manager.sequence_path(), sensor, "imu.txt")
                data = np.loadtxt(file)
                data = np.delete(data, np.where((data[:, 0].astype(np.uint64) <= start) | (data[:, 0].astype(np.uint64) >= stop)), axis=0)
                data = np.delete(data, (2, 3, 4), axis=1)
                self._data[sensor] = data
