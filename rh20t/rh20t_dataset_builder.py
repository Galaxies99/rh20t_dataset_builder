from typing import Iterator, Tuple, Any

import os
import cv2
import json
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from utils import transform_timestamp_key, convert_tcp, unify_joint, fetch_tactile


"""
The following 2 lines are added according to https://github.com/tensorflow/datasets/issues/2761, in order to fix the curl failure issue; they can be deleted if no error is reported.
"""
tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'


class rh20t(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        # To avoid setting up proxies:
        self._embed = hub.load("https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-large/5.tar.gz")
        with open('task_description.json', 'r') as f:
           self._task_description = json.load(f)
        self.gripper_threshold = 3

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Camera RGB observation.'
                        ),
                        'depth': tfds.features.Image(
                            shape=(720, 1280, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='Camera depth observation (in mm).'
                        ),
                        'tcp': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot tcp pose [3x xyz + 3x rpy] in the camera coordinate.'
                        ),
                        'tcp_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot tcp pose [3x xyz + 3x rpy] in the robot base coordinate.'
                        ),
                        'joint': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint pose [7x position for 7-DoF settings, and 6x for 6-DoF settings].'
                        ),
                        'joint_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint pose [7x position for 7-DoF settings, and 6x for 6-DoF settings];'
                                'all 0 means the configuration does not provide joint velocity data.'
                        ),
                        'ft_robot': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] value received from the robot in the camera coordinate.'
                        ),
                        'ft_raw': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] raw value received from the sensor in the camera coordinate.',
                        ),
                        'ft_zeroed': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] zeroed value (after taring) received from the sensor in the camera coordinate.'
                        ),
                        'ft_robot_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] value received from the robot in the robot base coordinate.'
                        ),
                        'ft_raw_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] raw value received from the sensor in the robot base coordinate.'
                        ),
                        'ft_zeroed_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Force/torque [3x force + 3x torque] zeroed value (after taring) received from the sensor in the robot base coordinate.'
                        ),
                        'gripper_width': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Gripper width in mm.'
                        ),
                        'tactile': tfds.features.Tensor(
                            shape=(96,),
                            dtype=np.int64,
                            doc='Tactile information (only available in RH20T cfg7).'
                        ),
                        'timestamp': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='Timestamp of this data record.'
                        )
                    }),
                    'action': tfds.features.FeaturesDict({
                        'tcp': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Tcp action (in camera coordinate).'
                        ),
                        'tcp_base': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Tcp action (in robot base coordinate).'
                        ),
                        'gripper': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='Gripper action.'
                        )
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'camera_id': tfds.features.Text(
                        doc='Camera serial number.'
                    ),
                    'task_id': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Task ID.'
                    ),
                    'user_id': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='User ID.'
                    ),
                    'scene_id': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Scene ID.'
                    ),
                    'cfg_id': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Configuration ID.'
                    ),
                    'audio': tfds.features.Audio(
                        file_format='wav',
                        doc='Audio.'
                    ),
                    'finish_timestamp': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Finish timestamp.'
                    ),
                    'rating': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='User rating (0: robot entered the emergency state; 1: task failed; 2-9: user evaluation of manipulation quality).'
                    )
                }),
            }))

    def _generate_paths(self, path, depth_path=None, joint_path=None):
        """Generator of paths: RH20T format configuration/scene/camera"""
        paths = []
        for conf in sorted(os.listdir(path)):
            for scene in sorted(os.listdir(os.path.join(path, conf))):
                if '_human' in scene or 'calib' in scene:
                    continue
                for camera in sorted(os.listdir(os.path.join(path, conf, scene))):
                    if 'cam_' not in camera:
                        continue
                    paths.append(
                        scene + "[SPLIT]" +
                        camera + "[SPLIT]" + 
                        os.path.join(path, conf, scene)  + "[SPLIT]" + 
                        os.path.join(path, conf, scene, camera)  + "[SPLIT]" + 
                        ("" if depth_path is None or not os.path.exists(os.path.join(depth_path, conf, scene, camera)) else os.path.join(depth_path, conf, scene, camera)) + "[SPLIT]" + 
                        ("" if joint_path is None or not os.path.exists(os.path.join(joint_path, conf, scene)) else os.path.join(joint_path, conf, scene))
                    )
        return paths

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=self._generate_paths(path='data/RH20T', depth_path='data/RH20T_depth', joint_path='data/RH20T')),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            scene_name, camera_name, base_path, color_path, depth_path, joint_path = episode_path.split("[SPLIT]")
            try:
                cam_sn = camera_name[4:]
                task_id = int(scene_name[5:9])
                user_id = int(scene_name[15:19])
                scene_id = int(scene_name[26:30])
                cfg_id = int(scene_name[35:39])
            except Exception:
                return None
            # Basic integrity check
            if not os.path.exists(color_path) or \
               not os.path.exists(os.path.join(color_path, 'color.mp4')) or \
               not os.path.exists(os.path.join(color_path, 'timestamps.npy')):
                return None
            if not os.path.exists(base_path) or \
               not os.path.exists(os.path.join(base_path, 'metadata.json')) or \
               not os.path.exists(os.path.join(base_path, 'transformed')) or \
               not os.path.exists(os.path.join(base_path, 'transformed', 'tcp.npy')) or \
               not os.path.exists(os.path.join(base_path, 'transformed', 'gripper.npy')) or \
               not os.path.exists(os.path.join(base_path, 'transformed', 'force_torque.npy')):
                return None
            # 0. load metadata and audio
            with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
                meta = json.load(f)
            finish_timestamp = meta['finish_time']
            rating = meta['rating']
            audio_file = os.path.join(base_path, 'audio_mixed')
            audio_file = os.path.join(audio_file, os.listdir(audio_file)[0])
            # 1. load color data and timestamps
            timestamps = np.load(os.path.join(color_path, 'timestamps.npy'), allow_pickle=True).item()
            cap = cv2.VideoCapture(os.path.join(color_path, 'color.mp4'))
            colors = {}
            cnt = 0
            height, width = (720, 1280)    # if not, will be updated soon.
            while True:
                ret, frame = cap.read()
                if ret:
                    height, width, _ = frame.shape
                    colors[timestamps['color'][cnt]] = np.array(frame).astype(np.uint8)
                    cnt += 1
                else:
                    break
            cap.release()
            # 2. load depth data (if any)
            if os.path.exists(os.path.join(depth_path, 'depth.mp4')):
                cap = cv2.VideoCapture(os.path.join(depth_path, 'depth.mp4'))
                depths = {}
                cnt = 0
                while True:
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray1 = np.array(gray[:height, :]).astype(np.uint16)
                        gray2 = np.array(gray[height:, :]).astype(np.uint16)
                        depths[timestamps['depth'][cnt]] = np.array(gray2 * 256 + gray1).astype(np.uint16)
                        cnt += 1
                    else:
                        break
            else:
                depths = None
            # 3. load transformed data 
            # 3.1 tcp w.r.t. camera
            tcps = np.load(os.path.join(base_path, 'transformed', 'tcp.npy'), allow_pickle=True).item()
            tcps = transform_timestamp_key(tcps)
            # 3.2 tcp w.r.t. base
            if os.path.exists(os.path.join(base_path, 'transformed', 'tcp_base.npy')):
                tcps_base = transform_timestamp_key(np.load(os.path.join(base_path, 'transformed', 'tcp_base.npy'), allow_pickle=True).item())
            else:
                tcps_base = None
            # 3.3 force/torque w.r.t. camera
            fts = np.load(os.path.join(base_path, 'transformed', 'force_torque.npy'), allow_pickle=True).item()
            fts = transform_timestamp_key(fts)
            # 3.4 force/torque w.r.t. base
            if os.path.exists(os.path.join(base_path, 'transformed', 'force_torque_base.npy')):
                fts_base = transform_timestamp_key(np.load(os.path.join(base_path, 'transformed', 'force_torque_base.npy'), allow_pickle=True).item())
            else:
                fts_base = None
            # 3.5 gripper
            grippers = np.load(os.path.join(base_path, 'transformed', 'gripper.npy'), allow_pickle=True).item()
            # 3.6 tactile (if any)
            if os.path.exists(os.path.join(base_path, 'transformed', 'tactile.npy')):
                tactiles = np.load(os.path.join(base_path, 'transformed', 'tactile.npy'))
            else:
                tactiles = None
            # 4. load joint data (if any)
            if os.path.exists(os.path.join(joint_path, 'transformed', 'joint.npy')):
                joints = np.load(os.path.join(joint_path, 'transformed', 'joint.npy'), allow_pickle=True).item()
            else:
                joints = None
            # 5. language instruction
            language_instruction = self._task_description[scene_name[:9]]["task_description_english"]
            # compute Kona language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()
            # 6. timestamps: based on color
            timestamp_base = timestamps['color']
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            last_gripper_action = 1
            for i, ts in enumerate(timestamp_base):
                if i == len(timestamp_base) - 1:
                    break
                next_ts = timestamp_base[i + 1]
                is_last = (i == len(timestamp_base) - 2) or (ts < finish_timestamp and next_ts >= finish_timestamp)
                try:
                    depth = np.expand_dims(depths[ts], -1).astype(np.uint16)
                except Exception:
                    depth = np.zeros((height, width, 1), dtype=np.uint16)
                tcp = convert_tcp(tcps[cam_sn][ts]['tcp'])
                tcp_base = convert_tcp(tcps_base[cam_sn][ts]['tcp'])
                tcp_action = convert_tcp(tcps[cam_sn][next_ts]['tcp'])
                tcp_base_action = convert_tcp(tcps_base[cam_sn][next_ts]['tcp'])
                try:
                    joint, joint_vel = unify_joint(joints[cam_sn][ts])
                except Exception:
                    joint = np.zeros(7).astype(np.float32)
                    joint_vel = np.zeros(7).astype(np.float32)
                try:
                    ft_robot = np.array(tcps[cam_sn][ts]['robot_ft']).astype(np.float32)
                except Exception:
                    ft_robot = np.zeros(6).astype(np.float32)
                try:
                    ft_robot_base = np.array(tcps_base[cam_sn][ts]['robot_ft']).astype(np.float32)
                except Exception:
                    ft_robot_base = np.zeros(6).astype(np.float32)
                try:
                    ft_raw = np.array(fts[cam_sn][ts]['raw']).astype(np.float32)
                except Exception:
                    ft_raw = np.zeros(6).astype(np.float32)
                try:
                    ft_zeroed = np.array(fts[cam_sn][ts]['zeroe']).astype(np.float32)
                except Exception:
                    ft_zeroed = np.zeros(6).astype(np.float32)
                try:
                    ft_raw_base = np.array(fts_base[cam_sn][ts]['raw']).astype(np.float32)
                except Exception:
                    ft_raw_base = np.zeros(6).astype(np.float32)
                try:
                    ft_zeroed_base = np.array(fts_base[cam_sn][ts]['zeroed']).astype(np.float32)
                except Exception:
                    ft_zeroed_base = np.zeros(6).astype(np.float32)
                gripper_width = grippers[cam_sn][ts]['gripper_info'][0]
                gripper_next_width = grippers[cam_sn][next_ts]['gripper_info'][0]
                if np.abs(gripper_width - gripper_next_width) >= self.gripper_threshold:
                    if gripper_next_width < gripper_width:
                        gripper_action = 0
                    else:
                        gripper_action = 1
                else:
                    gripper_action = last_gripper_action
                last_gripper_action = gripper_action
                try:
                    tactile = fetch_tactile(tactiles, ts)
                except Exception:
                    tactile = np.zeros(96).astype(np.int64)
                episode.append({
                    'observation': {
                        'image': colors[ts],
                        'depth': depth,
                        'tcp': tcp,
                        'tcp_base': tcp_base,
                        'joint': joint,
                        'joint_vel': joint_vel,
                        'ft_robot': ft_robot,
                        'ft_robot_base': ft_robot_base,
                        'ft_raw': ft_raw,
                        'ft_zeroed': ft_zeroed,
                        'ft_raw_base': ft_raw_base,
                        'ft_zeroed_base':ft_zeroed_base,
                        'gripper_width': gripper_width,
                        'tactile': tactile,
                        'timestamp': ts
                    },
                    'action': {
                        'tcp': tcp_action,
                        'tcp_base': tcp_base_action,
                        'gripper': gripper_action
                    },
                    'discount': 1.0,
                    'reward': float(is_last),
                    'is_first': i == 0,
                    'is_last': is_last,
                    'is_terminal': is_last,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })
            
            # invalid episode check
            if episode == []:
                return None

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': base_path,
                    'camera_id': cam_sn,
                    'task_id': task_id,
                    'user_id': user_id,
                    'scene_id': scene_id,
                    'cfg_id': cfg_id,
                    'audio': audio_file,
                    'finish_timestamp': finish_timestamp,
                    'rating': rating
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = path

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

