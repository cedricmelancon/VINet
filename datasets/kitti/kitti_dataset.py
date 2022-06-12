""" Inspired from https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py and adapted to PyTorch dataset """

from torchvision.datasets import VisionDataset
import torchvision.datasets.utils as utils
import os
import glob
from collections import namedtuple
import numpy as np
import datetime as dt
from .kitti_utils import *
import math
from scipy.spatial.transform import Rotation as R

class KittiDataset(VisionDataset):
    """Load and parse tracking benchmark data into a usable format."""
    def __init__(
        self, 
        base_path, 
        sequence, 
        sensor,
        frames = None,
        transform = None, 
        download = False
    ):
        self.sequence = sequence
        self.sequence_path = os.path.join(base_path, 'sequences', str(sequence).zfill(2))
        self.pose_path = os.path.join(base_path, 'poses')
        self.transform = transform

        super().__init__(self.sequence_path, transform=transform)

        self.frames = frames
        self.sensor = sensor
        self.imtype = 'png'

        if download:
            self.download()
        if not self._check_gt_exists() or not self._check_data_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        #self._load_calib()
        self._load_timestamps()
        self._load_poses()

    def __len__(self):
        """Return the number of frames loaded."""
        size = len(self.timestamps)

        if size % 2 != 0:
            size = size - 1
        return size

    def __getitem__(self, idx):
        if self.sensor == 'cam0':
            data1 = self.get_cam0(idx)
            data2 = self.get_cam0(idx + 1)
        elif self.sensor == 'cam1':
            data1 = self.get_cam1(idx)
            data2 = self.get_cam1(idx + 1)
        elif self.sensor == 'cam2':
            data1 = self.get_cam2(idx)
            data2 = self.get_cam2(idx + 1)
        elif self.sensor == 'cam3':
            data1 = self.get_cam3(idx)
            data2 = self.get_cam3(idx + 1)
        elif self.sensor == 'gray':
            data1 = self.get_gray(idx)
            data2 = self.get_gray(idx + 1)
        elif self.sensor == 'velo':
            data1 = self.get_velo(idx)
            data2 = self.get_velo(idx + 1)
        else:
            print("ERROR: sensor does not exist!")

        if self.transform:
            data = [self.transform(data1), self.transform(data2)]

        timestamp = self.timestamps[idx]
        item_value = {'timestamp': timestamp.microseconds, 'data': data}
        r = R.from_matrix(self.poses[idx][:3, :3])
        r = r.as_euler('zxy', degrees=False)
        item_label = np.array([self.poses[idx][0][3], self.poses[idx][1][3], self.poses[idx][2][3], r[0], r[1], r[2]])

        return item_value, item_label

    def get_cam0(self, idx):
        """Read image file for cam0 (monochrome left) at the specified index."""
        return load_image(self.cam0_files[idx], False)

    def get_cam1(self, idx):
        """Read image file for cam1 (monochrome right) at the specified index."""
        return load_image(self.cam1_files[idx], False)

    def get_cam2(self, idx):
        """Read image file for cam2 (RGB left) at the specified index."""
        return load_image(self.cam2_files[idx], True)

    def get_cam3(self, idx):
        """Read image file for cam3 (RGB right) at the specified index."""
        return load_image(self.cam3_files[idx], True)

    def get_gray(self, idx):
        """Read monochrome stereo pair at the specified index."""
        return (self.get_cam0(idx), self.get_cam1(idx))

    def get_rgb(self, idx):
        """Read RGB stereo pair at the specified index."""
        return (self.get_cam2(idx), self.get_cam3(idx))

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return load_velo_scan(self.velo_files[idx])

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam0_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_0',
                         '*.{}'.format(self.imtype))))
        self.cam1_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_1',
                         '*.{}'.format(self.imtype))))
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_2',
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_3',
                         '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'velodyne',
                         '*.bin')))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.cam0_files = subselect_files(
                self.cam0_files, self.frames)
            self.cam1_files = subselect_files(
                self.cam1_files, self.frames)
            self.cam2_files = subselect_files(
                self.cam2_files, self.frames)
            self.cam3_files = subselect_files(
                self.cam3_files, self.frames)
            self.velo_files = subselect_files(
                self.velo_files, self.frames)


    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.sequence_path, 'calib.txt')
        filedata = read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(self.sequence_path, 'times.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = dt.timedelta(seconds=float(line))
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]

    def _load_poses(self):
        """Load ground truth poses (T_w_cam0) from file."""
        pose_file = os.path.join(self.pose_path, str(self.sequence).zfill(2) + '.txt')

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                if self.frames is not None:
                    lines = [lines[i] for i in self.frames]

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' +
                  str(self.sequence).zfill(2) + '.')

        self.poses = poses

    def _check_data_exists(self) -> bool:
        """Check if the data directory exists."""
        if self.sensor == 'cam0':
            folder = 'image_0'
        elif self.sensor == 'cam1':
            folder = 'image_1'
        elif self.sensor == 'cam2':
            folder = 'image_2'
        elif self.sensor == 'cam3':
            folder = 'image_3'
        elif self.sensor == 'velo':
            folder = 'velodyne'
        elif self.sensor == 'oxts':
            folder = 'oxts'

        sensorFolder = os.path.join(self.sequence_path, folder)
        return os.path.isdir(sensorFolder)

    def _check_gt_exists(self) -> bool:
        return os.path.isdir(self.pose_path)

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if not self._check_data_exists():
            if self.sensor == 'cam0':
                filename = 'data_odometry_gray.zip'
            elif self.sensor == 'cam1':
                filename = 'data_odometry_gray.zip'
            elif self.sensor == 'cam2':
                filename = 'data_odometry_color.zip'
            elif self.sensor == 'cam3':
                filename = 'data_odometry_color.zip'
            elif self.sensor == 'velo':
                filename = 'data_odometry_velodyne.zip'

            # download files
            utils.download_and_extract_archive(
                url=f"https://s3.eu-central-1.amazonaws.com/avg-kitti/{filename}",
                download_root=os.path.dirname(__file__),
                filename=filename
            )

            os.remove(os.path.join(os.path.dirname(__file__), filename))

        if not self._check_gt_exists():
            filename = 'data_odometry_poses.zip'

            # download files
            utils.download_and_extract_archive(
                url=f"https://s3.eu-central-1.amazonaws.com/avg-kitti/{filename}",
                download_root=os.path.dirname(__file__),
                filename=filename
            )

            os.remove(os.path.join(os.path.dirname(__file__), filename))