from argparse import ArgumentParser
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from PIL import Image

CAM_TOPIC = "/camera_driver_gv/vis/image_raw"
REF_TOPIC = "/novatel/oem7/inspva"
IMU_TOPIC = "/xsens/imu/data"


def main():
    parser = ArgumentParser("read_bag")
    parser.add_argument("path")
    args = parser.parse_args()

    print("Hello from read-bag!")

    bagpath = Path(args.path)
    # print_topics(bagpath)
    extract_images(bagpath)


def print_topics(bagpath):
    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        for topic_name in reader.topics.keys():
            print(topic_name)


def extract_images(bagpath):
    """

    CSV Dump:
    - Reference : lat, lon, utc_time, pitch, roll, yaw
    - Low Cost Imu: pitch, roll, yaw
    - Pol Cam: image_path

    Name the polarization camera images using their timestamp
    """
    out_dir = Path("images")
    out_dir.mkdir()

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == CAM_TOPIC]
        count = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            filename = Path(f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}.png")
            Image.frombuffer("L", (2448, 2048), msg.data).save(out_dir / filename)

            count += 1
            print(f"[{count}/?] saved: {filename}")


if __name__ == "__main__":
    main()
