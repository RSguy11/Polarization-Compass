import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Topics
CAM_TOPIC = "/camera_driver_gv/vis/image_raw"
REF_TOPIC = "/novatel/oem7/inspva"

class DatabaseLoader():
    def __init__( self, data_path:Path, start_deg = 0.0, step_deg = 1.0 ):

        self.bag_path = Path(data_path)

        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

        self.step_deg = step_deg
        self.start_deg = start_deg

    def dataReader(self):
         with AnyReader([self.bag_path], default_typestore=self.typestore) as reader:
                connections = [x for x in reader.connections if x.topic == CAM_TOPIC]

                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    yield msg, timestamp

                # msg.data = raw pixels

    def extract_features(self, msg):

        H, W = 2048, 2448

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(H, W)

        I0   = img[0::2, 0::2].astype(np.float32)
        I45  = img[0::2, 1::2].astype(np.float32)
        I90  = img[1::2, 0::2].astype(np.float32)
        I135 = img[1::2, 1::2].astype(np.float32)

        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        eps = 1e-6
        denom = np.clip(S0, eps, None)

        DoLP = np.sqrt(S1**2 + S2**2) / denom         # (H, W)
        # Angle of Linear Polarization (radians, in ~(-pi/2, pi/2))
        AoLP = 0.5 * np.arctan2(S2, S1)               # (H, W)

        # Mask low-intensity pixels
        valid = S0 > 3  # threshold may be tuned

        if not np.any(valid):
            return 0.0, 0.0

        DoLP = DoLP[valid]
        AoLP = AoLP[valid]

        # Reduce to scalar features 
        dolp_mean = float(np.mean(DoLP))

        # Circular mean for AoLP (important!)
        aolp_mean = 0.5 * np.arctan2(
            np.mean(np.sin(2 * AoLP)),
            np.mean(np.cos(2 * AoLP))
        )

        return dolp_mean, float(aolp_mean)

    def extract_label(self,index):

        # Every image the camera was rotated by ix1  so we collect the azmuth degree 
        azimuth_deg = self.start_deg + index * self.step_deg

        return np.deg2rad(azimuth_deg)

    def get_item(self, max_samples: int | None = None ):
        X = []
        y = []
        timestamps = []

        #Loop through all the images produced by the data-reader
        for i, (msg, timestamp) in enumerate(self.dataReader()):
            dolp, aolp = self.extract_features(msg)
            
            #Creating the anglle vector to go back as the features
            X.append([
                dolp,
                np.sin(aolp),
                np.cos(aolp),
            ])

            #Getting the azmuth degree to go back as label
            y.append(self.extract_label(i))

            #Adding timestamp as metadata
            timestamps.append(timestamp)

            #Adding Ability to stop early for testing
            if max_samples is not None and i + 1 >= max_samples:
                break

        return (
            np.asarray(X, dtype=np.float32), # X = [DoLP, sin(AoLP), cos(AoLP)]
            np.asarray(y, dtype=np.float32),  # y = azimuth (rad)
            np.asarray(timestamps), # timestamp
        )
