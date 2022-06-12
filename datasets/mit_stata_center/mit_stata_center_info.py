
def mit_stata_center_get_sensor_info(sensor):
    if sensor == "right":
        encoding = 88
        channel_nb = 1
        image = True
        extension = ".jpg"
    elif sensor == "left":
        encoding = 88
        channel_nb = 1
        image = True
        extension = ".jpg"
    elif sensor == "lidar":
        image = False
        encoding = None
        channel_nb = None
        extension = '.bin'
    elif sensor == "odometer":
        image = False
        encoding = None
        channel_nb = None
        extension = '.txt'
    elif sensor == "imu":
        image = False
        encoding = None
        channel_nb = None
        extension = '.txt'
    elif sensor == "depth":
        encoding = None
        channel_nb = 1
        extension = ".tif"
        image = True
    elif sensor == "rgb":
        channel_nb = 1
        encoding = 47
        image = True
        extension = ".jpg"

    return {'image': image, 'encoding': encoding, 'channel_nb': channel_nb, 'extension': extension}

def mit_stata_center_get_topic_info(topic):
    extension = ".jpg"
    if topic == "/wide_stereo/right/image_raw":
        sensor = "right"
        encoding = 88
        channel_nb = 1
    elif topic == "/wide_stereo/left/image_raw":
        sensor = "left"
        encoding = 88
        channel_nb = 1
    elif topic == "/base_scan":
        sensor = "lidar"
        encoding = None
        channel_nb = None
    elif topic == "/base_odometry/odom":
        sensor = "odometer"
        encoding = None
        channel_nb = None
    elif topic == "/torso_lift_imu/data":
        sensor = "imu"
        encoding = None
        channel_nb = None
    elif topic == "/camera/depth/image_raw":
        sensor = "depth"
        channel_nb = 1
        encoding = None
        extension = ".tif"
    elif topic == "/camera/rgb/image_raw":
        sensor = "rgb"
        channel_nb = 1
        encoding = 47

    return {'sensor': sensor, 'encoding': encoding, 'channel_nb': channel_nb, 'extension': extension}