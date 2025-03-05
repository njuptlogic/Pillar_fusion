"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box
#########################################################################
import os
from .pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.eval.detection.utils import category_to_detection_name
from .utils_kitti import KittiDB
import copy
#########################################################################
map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}
##################################################
def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha
####################################################

def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10, with_cam=True):
    import concurrent.futures

    def load_radar_data(nusc, sample, radar_channel, cam, nsweeps=4):
        # 使用雷达通道读取数据
        radar_pcs, _ = RadarPointCloud.from_file_multisweep(nusc, sample, radar_channel, cam, nsweeps=nsweeps)
        return radar_pcs.points  # 直接返回点云数据

    def optimized_radar_points(nusc, sample, RADARS_FOR_CAMERA, cam, nsweeps=4):
        all_points = []
    
        # 使用多线程并行读取不同雷达通道的数据
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交所有雷达通道的任务
            futures = [executor.submit(load_radar_data, nusc, sample, radar_channel, cam, nsweeps) 
                       for radar_channel in RADARS_FOR_CAMERA[cam]]
        
            # 获取所有雷达通道的数据并将它们合并
            for future in concurrent.futures.as_completed(futures):
                all_points.append(future.result())

        # 合并所有雷达通道的点云数据（避免使用np.hstack，可以通过list拼接，然后转换成numpy数组）
        all_points = np.concatenate(all_points, axis=1)  # 合并点云数据，确保axis=1是拼接列方向
    
        return all_points




    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The reference channel for LiDAR data
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    # 雷达与摄像头的映射关系
    RADARS_FOR_CAMERA = {
        'CAM_FRONT_LEFT':  ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
        'CAM_FRONT_RIGHT': ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
        'CAM_FRONT':       ["RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT"],
        'CAM_BACK_LEFT':   ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
        'CAM_BACK_RIGHT':  ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
        'CAM_BACK':        ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"]
    }
    CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
    CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ ,hei, wid= get_sample_data(nusc, ref_sd_token)
        #print("hei,wid",hei,wid)
        ref_cam_front_token = sample['data']['CAM_FRONT']
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

        # Location information
        location = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        
        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
            'cam_intrinsic': ref_cam_intrinsic,
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
            "location": location,
        }
        camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        
        # If camera information is needed
        if with_cam:
            info['cams'] = dict()
            l2e_r = ref_cs_rec["rotation"]
            l2e_t = ref_cs_rec["translation"],
            e2g_r = ref_pose_rec["rotation"]
            e2g_t = ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
            for cam in camera_types:
                cam_token = sample["data"][cam]

                # Use nusc.get_sample_data to get the camera intrinsics
                cam_path, boxes, camera_intrinsics = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
                
                #cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                ##
                cam_info['camera_intrinsic_2'] = camera_intrinsics.tolist()
                cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                cam_info.update(camera_intrinsics=camera_intrinsics)
                
                #info['camera_intrinsic_2'] = camera_intrinsics.tolist()
                '''
                # 使用优化后的函数
                radar_points = optimized_radar_points(nusc, sample, RADARS_FOR_CAMERA, cam, nsweeps=4)
                # 如果需要，保存到info字典
                info['radar_pc'] = radar_points.tolist()
                '''
                '''
                _, boxes, camera_intrinsic_2 = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
                info['camera_intrinsic_2'] = camera_intrinsic_2.tolist()
                '''
                calib = np.eye(4, dtype=np.float32)
                calib[:3, :3] = camera_intrinsics
                calib = calib[:3]
                cam_info['calib'] = calib.tolist()
                sd_record_2 = nusc.get('sample_data', cam_token)
                cam_info['width'] = sd_record_2['width']
                cam_info['height'] = sd_record_2['height']
                # Initialize an empty list to collect all radar points
                all_points = RadarPointCloud(np.zeros((18, 0)))
                for radar_channel in RADARS_FOR_CAMERA[cam]:
                    # Preload radar data (assuming from_file_multisweep can return the points directly)
                    radar_pcs, _ = RadarPointCloud.from_file_multisweep(nusc, sample, radar_channel, cam, nsweeps=4)
                    all_points.points = np.hstack((all_points.points, radar_pcs.points))
                    #all_points.points = np.hstack((all_points.points, all_points.points))
                # Store the radar point cloud data as a list for further use
                cam_info['radar_pc'] = all_points.points.tolist()
                
                anns = []
                for box in boxes:
                    det_name = category_to_detection_name(box.name)
                    if det_name is None:
                        continue
                    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                    yaw = -np.arctan2(v[2], v[0])
                    box.translate(np.array([0, box.wlh[2] / 2, 0]))
                    category_id = CAT_IDS[det_name]
             
                    ann = {
                        'category_id': category_id,
                        'dim': [box.wlh[2], box.wlh[0], box.wlh[1]],
                        'location': [box.center[0], box.center[1], box.center[2]],
                        'depth': box.center[2],
                        'rotation_y': yaw,
                        'iscrowd': 0
                        }
                    bbox = KittiDB.project_kitti_box_to_image(copy.deepcopy(box), camera_intrinsics, imsize=(1600, 900))
                    alpha = _rot_y2alpha(yaw, (bbox[0] + bbox[2]) / 2, camera_intrinsics[0, 2], camera_intrinsics[0, 0])
                    ann['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    ann['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    ann['alpha'] = alpha
                    anns.append(ann)
                cam_info['annotations'] = anns
                info["cams"].update({cam: cam_info})
                '''
                # 通过变换矩阵映射雷达点云
                all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))  # 初始化空点云
                #for cam_name in info["cams"].keys():
                #for cam in camera_types:
                # 获取与当前摄像头相关的雷达传感器
                #radar_channels = RADARS_FOR_CAMERA.get(cam, [])
                #radar_channels = RADARS_FOR_CAMERA[cam]
                for radar_channel in RADARS_FOR_CAMERA[cam]:
                    radar_pcs, _ = RadarPointCloud.from_file_multisweep(
                        nusc, sample, radar_channel, cam, nsweeps=6)
                    #radar_pcs.transform(tm)  # 应用变换矩阵到每个雷达点云
                    all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))  # 拼接点云

                info['radar_pc'] = all_radar_pcs.points.tolist()  # 将雷达点云转换为列表并存储
                '''
        # Process sweeps (for LiDAR)
        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get('calibrated_sensor', curr_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        ###########################################################
        '''
        # 雷达处理和映射
        radar_sweeps = []
        for cam_name in info["cams"].keys():
            # 获取与当前摄像头相关的雷达传感器
            radar_channels = RADARS_FOR_CAMERA.get(cam_name, [])
            
            for radar_chan in radar_channels:
                curr_sd_rec = nusc.get('sample_data', sample['data'][radar_chan])
                radar_channel_sweeps = []
                data_path = os.path.normpath(data_path)
                filename = os.path.normpath(curr_sd_rec['filename'])
                while len(radar_channel_sweeps) < 5:
                    if curr_sd_rec['prev'] == '':
                        radar_channel_sweeps.append({
                            'radar_path': Path(data_path) / Path(filename),
                            'sample_data_token': curr_sd_rec['token'], 
                            'time_lag': curr_sd_rec['timestamp'] * 0,
                        })
                    else:
                        curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])
                        radar_path = nusc.get_sample_data_path(curr_sd_rec['token'])
                        radar_channel_sweeps.append({
                            'radar_path': Path(radar_path).relative_to(data_path).__str__(),
                            'sample_data_token': curr_sd_rec['token'],
                            'time_lag': ref_time - 1e-6 * curr_sd_rec['timestamp'],
                        })
                radar_sweeps.append({
                    'channel': radar_chan,
                    'sweeps': radar_channel_sweeps
                })
        info['radar_sweeps'] = radar_sweeps
        '''
        ###################################################################
        '''
        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"
        '''
        '''
        # 通过变换矩阵映射雷达点云
        all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))  # 初始化空点云
        #for cam_name in info["cams"].keys():
        for cam in camera_types:
            # 获取与当前摄像头相关的雷达传感器
            #radar_channels = RADARS_FOR_CAMERA.get(cam, [])
            #radar_channels = RADARS_FOR_CAMERA[cam]
            for radar_channel in RADARS_FOR_CAMERA[cam]:
                radar_pcs, _ = RadarPointCloud.from_file_multisweep(
                    nusc, sample, radar_channel, cam, nsweeps=6
                )
                #radar_pcs.transform(tm)  # 应用变换矩阵到每个雷达点云
                all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))  # 拼接点云

        info['radar_pc'] = all_radar_pcs.points.tolist()  # 将雷达点云转换为列表并存储
        '''
        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)
            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_radar_pts[mask]
            if with_cam:
                info['empty_mask'] = mask
                # add 2d box
                gt_boxes_2d = [] # one 3d box to one 2d box
                all_2d_boxes = [] # all 2d boxes (one 3d box may project to more than one 2d box)
                # NOTE generate projected one-to-one 2d box
                for anno in annotations:
                    box = nusc.get_box(anno['token'])
                    has_proj = False
                    for index, cam in enumerate(camera_types):
                        tmp_box = box.copy()
                        cam_info = info['cams'][cam]
                        tmp_box.translate(-np.array(cam_info['ego2global_translation']))
                        tmp_box.rotate(Quaternion(cam_info['ego2global_rotation']).inverse)
                        # Move them to the calibrated sensor frame.
                        tmp_box.translate(-np.array(cam_info['sensor2ego_translation']))
                        tmp_box.rotate(Quaternion(cam_info['sensor2ego_rotation']).inverse)

                        # Filter out the corners that are not in front of the calibrated sensor.
                        corners_3d = tmp_box.corners()
                        # print(corners_3d)
                        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                        corners_3d = corners_3d[:, in_front]

                        # Project 3d box to 2d.
                        corner_coords = view_points(corners_3d, cam_info['camera_intrinsics'], True).T[:, :2].tolist()
                        # Keep only corners that fall within the image.
                        final_coords = post_process_coords(corner_coords)
                        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                        if final_coords is None:
                            continue
                        else:
                            min_x, min_y, max_x, max_y = final_coords
                            if has_proj == False:
                                gt_boxes_2d.append([min_x, min_y, max_x, max_y, index])
                                has_proj = True
                            all_2d_boxes.append([min_x, min_y, max_x, max_y, index])
                    if has_proj == False:
                        gt_boxes_2d.append([0, 0, 1, 1, 5])
                info['gt_boxes_2d'] = np.array(gt_boxes_2d)
                info['all_2d_boxes'] = np.array(all_2d_boxes)
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos



def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
        print("size",imsize)
    else:
        cam_intrinsic = imsize = None

    #########################################################
    '''
    if sensor_record['modality'] == 'radar':
        radar_intrinsic = None  # RADAR 没有相机内参
        data_path = nusc.get_sample_data_path(sd_record['token'])
        # 获取 RADAR 数据路径
        return data_path, boxes, radar_intrinsic
    '''
    ########################################################

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic, sd_record['height'], sd_record['width']


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw
    

def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
'''
def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10, with_cam=False):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_baval_nusc_infosr = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample['data']['CAM_FRONT']
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
        )

        location = nusc.get(
            "log", nusc.get("scene", sample["scene_token"])["log_token"]
        )["location"]
        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
            'cam_intrinsic': ref_cam_intrinsic,
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
            "location": location,
        }
        if with_cam:
            info['cams'] = dict()
            l2e_r = ref_cs_rec["rotation"]
            l2e_t = ref_cs_rec["translation"],
            e2g_r = ref_pose_rec["rotation"]
            e2g_t = ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info["cams"].update({cam: cam_info})
        

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                )
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps
        ###########################################################
         # RADAR sweeps提取
        import os
        radar_sweeps = []
        radar_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        for radar_chan in radar_channels:
            curr_sd_rec = nusc.get('sample_data', sample['data'][radar_chan])
            radar_channel_sweeps = []
            data_path = os.path.normpath(data_path)
            filename = os.path.normpath(curr_sd_rec['filename'])
            while len(radar_channel_sweeps) < max_sweeps - 1:
                if curr_sd_rec['prev'] == '':
                    radar_channel_sweeps.append({
                        #'radar_path': Path(filename).relative_to(data_path).__str__(),
                        'radar_path': Path(data_path) / Path(filename) ,
                        'sample_data_token': curr_sd_rec['token'], 
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                     })
                else:
                    curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])
                    radar_path = nusc.get_sample_data_path(curr_sd_rec['token'])
                    radar_channel_sweeps.append({
                        'radar_path': Path(radar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': tm,  # 同 LIDAR 的 TM 变换计算
                        'time_lag': ref_time - 1e-6 * curr_sd_rec['timestamp'],
                    })
            radar_sweeps.append({
                'channel': radar_chan,
                'sweeps': radar_channel_sweeps
            })
        info['radar_sweeps'] = radar_sweeps

        ###################################################################
        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)
            ###############################################
            # 添加 RADAR 数据到目标标注中
            radar_sweep_info = []
            for anno in annotations:
                radar_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
                channel_info = {}
                for radar_chan in radar_channels:
                    radar_token = sample['data'][radar_chan]
                    radar_data_path, radar_boxes, _ = get_sample_data(nusc, radar_token, selected_anntokens=[anno['token']])
                    channel_info[radar_chan] = {
                        'radar_path': Path(radar_data_path).relative_to(data_path).__str__(),
                        'radar_boxes': [box.center.tolist() for box in radar_boxes]
                    }
                radar_sweep_info.append(channel_info)
            info['radar_sweep_info'] = radar_sweep_info
            ###########################################################
            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_radar_pts[mask]
            if with_cam:
                info['empty_mask'] = mask
                # add 2d box
                gt_boxes_2d = [] # one 3d box to one 2d box
                all_2d_boxes = [] # all 2d boxes (one 3d box may project to more than one 2d box)
                # NOTE generate projected one-to-one 2d box
                for anno in annotations:
                    box = nusc.get_box(anno['token'])
                    has_proj = False
                    for index, cam in enumerate(camera_types):
                        tmp_box = box.copy()
                        cam_info = info['cams'][cam]
                        tmp_box.translate(-np.array(cam_info['ego2global_translation']))
                        tmp_box.rotate(Quaternion(cam_info['ego2global_rotation']).inverse)
                        # Move them to the calibrated sensor frame.
                        tmp_box.translate(-np.array(cam_info['sensor2ego_translation']))
                        tmp_box.rotate(Quaternion(cam_info['sensor2ego_rotation']).inverse)

                        # Filter out the corners that are not in front of the calibrated sensor.
                        corners_3d = tmp_box.corners()
                        # print(corners_3d)
                        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                        corners_3d = corners_3d[:, in_front]

                        # Project 3d box to 2d.
                        corner_coords = view_points(corners_3d, cam_info['camera_intrinsics'], True).T[:, :2].tolist()
                        # Keep only corners that fall within the image.
                        final_coords = post_process_coords(corner_coords)
                        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                        if final_coords is None:
                            continue
                        else:
                            min_x, min_y, max_x, max_y = final_coords
                            if has_proj == False:
                                gt_boxes_2d.append([min_x, min_y, max_x, max_y, index])
                                has_proj = True
                            all_2d_boxes.append([min_x, min_y, max_x, max_y, index])
                    if has_proj == False:
                        gt_boxes_2d.append([0, 0, 1, 1, 5])
                info['gt_boxes_2d'] = np.array(gt_boxes_2d)
                info['all_2d_boxes'] = np.array(all_2d_boxes)
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos
'''

def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details
