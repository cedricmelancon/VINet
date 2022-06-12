import rosbag
import argparse
import cv2
import numpy as np
import os
import struct
from mit_stata_center_info import mit_stata_center_get_topic_info

parser = argparse.ArgumentParser(description='MIT Stata Center Rosbag converter',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')

args = parser.parse_args()

sourceName = "./source/" + args.data + ".bag"
bag = rosbag.Bag(sourceName)

bag_topics = bag.get_type_and_topic_info()[1].keys()

topics = ["/torso_lift_imu/data"]
    #["/camera/rgb/image_raw",
    #      "/camera/depth/image_raw",
    #      "/wide_stereo/right/image_raw",
    #      "/wide_stereo/left/image_raw",
    #      "/base_scan",
    #      "/base_odometry/odom",
    #      "/torso_lift_imu/data"]
for topic in topics:
    if topic in bag_topics:
        bagData = bag.read_messages([topic])
        path = os.path.dirname(__file__)

        info = mit_stata_center_get_topic_info(topic)

        sensorPath = os.path.join(path, "sequences", args.data, info['sensor'])
        os.makedirs(sensorPath, exist_ok=True)

        if topic in ["/wide_stereo/right/image_raw", "/wide_stereo/left/image_raw",
                     "/camera/rgb/image_raw", "/camera/depth/image_raw"]:
            for item in bagData:
                data = np.array(list(item.message.data), dtype=np.uint8)
                if topic == "/camera/depth/image_raw":
                    data = data.view(np.uint16).reshape(item.message.height, item.message.width)
                data = data.reshape(info['channel_nb'], item.message.height, item.message.width).transpose(1, 2, 0)
                timestamp = item.timestamp.to_nsec() / 1000
                filename = os.path.join(sensorPath, str(timestamp) + info['extension'])

                if topic == "/camera/depth/image_raw":
                    cv2.imwrite(filename, data)
                else:
                    cv2.imwrite(filename, cv2.cvtColor(data, info['encoding']))

        elif topic == "/base_scan":
            for item in bagData:
                lastMessage = item.message
                timestamp = item.timestamp.to_nsec() / 1000
                filename = os.path.join(sensorPath, str(timestamp) + ".bin")
                newFile = open(filename, "wb")
                data = struct.pack("!I" + "d" * len(item.message.ranges), len(item.message.ranges), *item.message.ranges)
                newFile.write(data)
                newFile.close()

            filename = os.path.join(sensorPath, "config.txt")
            newFile = open(filename, "w")
            newFile.write("Angle increment: " + str(lastMessage.angle_increment))
            newFile.close()
        elif topic == "/base_odometry/odom":
            data = []
            for item in bagData:
                timestamp = item.timestamp.to_nsec() / 1000
                dataRow = [timestamp, item.message.twist.twist.linear.x, item.message.twist.twist.linear.y, item.message.twist.twist.linear.z, item.message.twist.twist.angular.x, item.message.twist.twist.angular.y, item.message.twist.twist.angular.z ]
                data.append(dataRow)

            data = np.array(data, dtype=float)
            filename = os.path.join(sensorPath, "odometer.txt")
            np.savetxt(filename, data, delimiter=' ')
        elif topic == "/torso_lift_imu/data":
            data = []
            for item in bagData:
                timestamp = item.timestamp.to_nsec() / 1000
                dataRow = [timestamp, item.message.orientation.x, item.message.orientation.y, item.message.orientation.z, item.message.orientation.w, item.message.angular_velocity.x, item.message.angular_velocity.y, item.message.angular_velocity.z, item.message.linear_acceleration.x, item.message.linear_acceleration.y, item.message.linear_acceleration.z]
                data.append(dataRow)

            data = np.array(data, dtype=float)
            filename = os.path.join(sensorPath, "imu.txt")
            np.savetxt(filename, data, delimiter=' ')

#zipFilename = os.path.join("/home/data", "sequences", args.data)
#zipDirectory = os.path.join(".", path, "sequences", args.data)
#shutil.make_archive(zipFilename, 'zip', zipDirectory)
