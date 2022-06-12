import os
from smb.SMBConnection import SMBConnection
import zipfile


class MitStataCenterFileManager:
    server = "10.180.97.24"
    share = "data"

    def __init__(self, path, sequence, sensor, ground_truth):
        self._path = path
        self._sensor = sensor
        self._sequence = sequence
        self._ground_truth = ground_truth

    def _check_sequence_exists(self):
        """Check if the data directory exists."""
        folders = [self._sequence]
        return all(os.path.isdir(os.path.join(self._path, fname)) for fname in folders)

    def _check_ground_trugh_exists(self):
        destination_path = os.path.join(self._path, "ground_truth", self._ground_truth + ".poses")
        return os.path.exists(destination_path)

    def check_exists(self):
        return self._check_sequence_exists() and self._check_ground_trugh_exists()

    def ground_truth_path(self):
        return os.path.join(self._path, "ground_truth", self._ground_truth + ".poses")

    def sequence_path(self):
        return os.path.join(self._path, self._sequence)

    def sensor_path(self):
        return os.path.join(self._path, self._sequence, self._sensor)

    def download(self):
        """Download the MIT Stata Center data if it doesn't exist already."""

        if not self._check_sequence_exists():
            # download files
            self.download_and_extract_archive()

        if not self._check_ground_trugh_exists():
            self.download_ground_truth()

    def download_ground_truth(self):
        conn = SMBConnection("", "", "local", "gitlab-ms", use_ntlm_v2=True)
        assert conn.connect(self.server, 139)

        os.makedirs(os.path.join(self._path, "ground_truth"), exist_ok=True)

        fileobj = open(self.ground_truth_path(), 'wb')

        print("Downloading file")
        conn.retrieveFile(self.share, "Datasets/MITStataCenter/ground_truth/" + self._ground_truth + ".poses", fileobj)
        fileobj.close()

    def download_and_extract_archive(self):
        conn = SMBConnection("", "", "local", "gitlab-ms", use_ntlm_v2=True)
        assert conn.connect(self.server, 139)

        os.makedirs(self._path, exist_ok=True)

        destination_path = os.path.join(self._path, self._sequence + ".zip")
        fileobj = open(destination_path, 'wb')

        print("Downloading file")
        conn.retrieveFile(self.share, "Datasets/MITStataCenter/" + self._sequence + ".zip", fileobj)
        fileobj.close()

        print("Extracting file")
        with zipfile.ZipFile(destination_path, "r", compression=zipfile.ZIP_STORED) as my_zip:
            my_zip.extractall(destination_path)
            my_zip.close()

        os.remove(self._path + "/" + self._sequence + ".zip")
