import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image
import cv2
import os

################################################################################
from .image import flip, color_aug
from .image import get_affine_transform, affine_transform
from .image import gaussian_radius, draw_umich_gaussian, gaussian2D
from .pointcloud import map_pointcloud_to_image, pc_dep_to_hm
from . import nuscenes_utils
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
#import pycocotools.coco as coco
###################################################################################
from .pointcloud import RadarPointCloudWithVelocity
from .ddd_utils import comput_corners_3d, alpha2rot_y, get_pc_hm
###########################################################################################
debug=0
input_w = 800
input_h = 448
down_ratio=4
output_w = input_w // down_ratio
output_h = input_h // down_ratio

exp_id='centerfusion'
# log dirs
task='ttt'
root_dir = os.path.join(os.path.dirname(__file__), '..', '..','..')
data_dir = os.path.join(root_dir, 'data')
exp_dir = os.path.join(root_dir, 'exp', task)
save_dir = os.path.join(exp_dir, exp_id)
debug_dir = os.path.join(save_dir, 'debug')
img_format='jpg'
img_ind = 0
no_color_aug = False
pc_feat_lvl = ['pc_dep','pc_vx','pc_vz',] 
pc_roi_method = "pillars"
r_a = float(250)
r_b = float(5)
pc_feat_channels = {feat: i for i,feat in enumerate(pc_feat_lvl)}
max_pc_dist = 60.0
pc_z_offset = 0.0
max_pc = int(1000)
not_rand_crop = True
scale = 0
shift = 0.1
aug_rot = float(0)
rotate = float(0)
dataset_version=None
radar_sweeps = 6
not_max_crop = False
flip = 0.5
split = 'train'
##############

def get_dist_thresh(calib, ct, dim, alpha):
    rotation_y = alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0])
    corners_3d = comput_corners_3d(dim, rotation_y)
    dist_thresh = max(corners_3d[:,2]) - min(corners_3d[:,2]) / 2.0
    return dist_thresh

class NuScenesDataset(DatasetTemplate):
    ########################################
    num_categories = 10
    class_name = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle','traffic_cone', 'barrier']
    #cat_ids = {i + 1: i + 1 for i in range(num_categories)}
    ##############################################
    rest_focal_length = 1200
    num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    edges = [[0, 1], [0, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [6, 12], [5, 11], [11, 12], [12, 14], [14, 16], [11, 13], [13, 15]]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],dtype=np.float32).reshape(1, 1, 3)
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
    _eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],[-0.5832747, 0.00994535, -0.81221408],[-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
    ignore_val = 1
    nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}
  
    ## change these vectors to actual mean and std to normalize
    pc_mean = np.zeros((18,1))
    pc_std = np.ones((18,1))
    num_categories = 10
    class_name = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle','traffic_cone', 'barrier']
    cat_ids = {i + 1: i + 1 for i in range(num_categories)}
    focal_length = 1200
    max_objs = 128
    _tracking_ignored_class = ['construction_vehicle', 'traffic_cone', 'barrier']
    _vehicles = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']
    _cycles = ['motorcycle', 'bicycle']
    _pedestrians = ['pedestrian']
    attribute_to_id = {'': 0, 'cycle.with_rider' : 1, 'cycle.without_rider' : 2,'pedestrian.moving': 3, 'pedestrian.standing': 4, 'pedestrian.sitting_lying_down': 5,'vehicle.moving': 6, 'vehicle.parked': 7, 'vehicle.stopped': 8}
    id_to_attribute = {v: k for k, v in attribute_to_id.items()}
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        ############################################
        self._data_rng = np.random.RandomState(123)
        ##############################################
        self.infos = []
        ################################################
        self.load_interval = self.dataset_cfg.get('INTERVAL',1)
        ################################################
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False
        
        self.map_config = self.dataset_cfg.get('MAP_CONFIG',None)
        if self.map_config is not None:
            self.use_map = self.map_config.get('USE_MAP',True)
            self.map_classes = self.map_config.CLASS_NAMES
        else:
            self.use_map = False

        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
        #########################
    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                ###############################
                if mode == 'train':
                    nuscenes_infos.extend(infos[::self.load_interval])
                else:
                    nuscenes_infos.extend(infos)
                ##############################
                #nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T
    
    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        input_dict['ori_imgs'] = [np.array(img) for img in imgs]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)
            #input_dict["camera_intrinsics_2"].append(camera_info["camera_intrinsics"])

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        #print(input_dict["image_paths"])
        images = []
        for name in filename:
            # images.append(Image.open(str(self.root_path / name))) 
            images.append(Image.fromarray(np.array(Image.open(str(self.root_path / name)))[:,:,::-1])) # bgr
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

####################################################################################################
    def create_pc_pillars(self, img, info, pc_2d, pc_3d, inp_trans, out_trans):
        pillar_wh = np.zeros((2, pc_3d.shape[1]))
        boxes_2d = np.zeros((0,8,2))
        ##################################
        pillar_dim = [1.5,0.2,0.2]
        #################################
        v = np.dot(np.eye(3), np.array([1,0,0]))
        ry = -np.arctan2(v[2], v[0])

        for i, center in enumerate(pc_3d[:3,:].T):
          # Create a 3D pillar at pc location for the full-size image
          box_3d = compute_box_3d(dim=pillar_dim, location=center, rotation_y=ry)
          box_2d = project_to_image(box_3d, info['calib']).T  # [2x8]        
      
          ## save the box for debug plots
          
          if debug:
            box_2d_img, m = self._transform_pc(box_2d, inp_trans, input_w, input_h, filter_out=False)
            boxes_2d = np.concatenate((boxes_2d, np.expand_dims(box_2d_img.T,0)),0)

          # transform points
          box_2d_t, m = self._transform_pc(box_2d, out_trans, output_w, output_h)
      
          if box_2d_t.shape[1] <= 1:
            continue

          # get the bounding box in [xyxy] format
          bbox = [np.min(box_2d_t[0,:]), 
                  np.min(box_2d_t[1,:]), 
                  np.max(box_2d_t[0,:]), 
                  np.max(box_2d_t[1,:])] # format: xyxy

          # store height and width of the 2D box
          pillar_wh[0,i] = bbox[2] - bbox[0]
          pillar_wh[1,i] = bbox[3] - bbox[1]

        ## DEBUG #################################################################
        if debug:
          img_2d = copy.deepcopy(img)
          # img_3d = copy.deepcopy(img)
          img_2d_inp = cv2.warpAffine(img, inp_trans, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
          img_2d_out = cv2.warpAffine(img, out_trans, 
                            (output_w, output_h),
                            flags=cv2.INTER_LINEAR)
          img_3d = cv2.warpAffine(img, inp_trans, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
          blank_image = 255*np.ones((input_h,input_w,3), np.uint8)
          overlay = img_2d_inp.copy()
          output = img_2d_inp.copy()

          pc_inp, _= self._transform_pc(pc_2d, inp_trans, input_w, input_h)
          pc_out, _= self._transform_pc(pc_2d, out_trans, output_w, output_h)

          pill_wh_inp = pillar_wh * (input_w/output_w)
          pill_wh_out = pillar_wh
          pill_wh_ori = pill_wh_inp * 2
      
          for i, p in enumerate(pc_inp[:3,:].T):
            color = int((p[2].tolist()/60.0)*255)
            color = (0,color,0)
        
            rect_tl = (np.min(int(p[0]-pill_wh_inp[0,i]/2), 0), np.min(int(p[1]-pill_wh_inp[1,i]),0))
            rect_br = (np.min(int(p[0]+pill_wh_inp[0,i]/2), 0), int(p[1]))
            cv2.rectangle(img_2d_inp, rect_tl, rect_br, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            img_2d_inp = cv2.circle(img_2d_inp, (int(p[0]), int(p[1])), 3, color, -1)

            ## On original-sized image
            rect_tl_ori = (np.min(int(pc_2d[0,i]-pill_wh_ori[0,i]/2), 0), np.min(int(pc_2d[1,i]-pill_wh_ori[1,i]),0))
            rect_br_ori = (np.min(int(pc_2d[0,i]+pill_wh_ori[0,i]/2), 0), int(pc_2d[1,i]))
            cv2.rectangle(img_2d, rect_tl_ori, rect_br_ori, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            img_2d = cv2.circle(img_2d, (int(pc_2d[0,i]), int(pc_2d[1,i])), 6, color, -1)
        
            p2 = pc_out[:3,i].T
            rect_tl2 = (np.min(int(p2[0]-pill_wh_out[0,i]/2), 0), np.min(int(p2[1]-pill_wh_out[1,i]),0))
            rect_br2 = (np.min(int(p2[0]+pill_wh_out[0,i]/2), 0), int(p2[1]))
            cv2.rectangle(img_2d_out, rect_tl2, rect_br2, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            img_2d_out = cv2.circle(img_2d_out, (int(p[0]), int(p[1])), 3, (255,0,0), -1)
        
            # on blank image
            cv2.rectangle(blank_image, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA)
        
            # overlay
            alpha = 0.1
            cv2.rectangle(overlay, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            # plot 3d pillars
            img_3d = draw_box_3d(img_3d, boxes_2d[i].astype(np.int32), [114, 159, 207], 
                        same_color=False)
            
            
          cv2.imwrite((debug_dir+ '/{}pc_pillar_2d_inp.' + img_format)\
            .format(img_ind), img_2d_inp)
          cv2.imwrite((debug_dir+ '/{}pc_pillar_2d_ori.' + img_format)\
            .format(img_ind), img_2d)
          cv2.imwrite((debug_dir+ '/{}pc_pillar_2d_out.' + img_format)\
            .format(img_ind), img_2d_out)
          cv2.imwrite((debug_dir+'/{}pc_pillar_2d_blank.'+ img_format)\
            .format(img_ind), blank_image)
          cv2.imwrite((debug_dir+'/{}pc_pillar_2d_overlay.'+ img_format)\
            .format(img_ind), output)
          cv2.imwrite((debug_dir+'/{}pc_pillar_3d.'+ img_format)\
            .format(img_ind), img_3d)
          img_ind += 1
        ## DEBUG #################################################################
        return pillar_wh


    def _transform_pc(self, pc_2d, trans, img_width, img_height, filter_out=True):

        if pc_2d.shape[1] == 0:
          return pc_2d, []

        pc_t = np.expand_dims(pc_2d[:2,:].T, 0)   # [3,N] -> [1,N,2]
        t_points = cv2.transform(pc_t, trans)
        t_points = np.squeeze(t_points,0).T       # [1,N,2] -> [2,N]
    
        # remove points outside image
        if filter_out:
          mask = (t_points[0,:]<img_width) \
                  & (t_points[1,:]<img_height) \
                  & (0<t_points[0,:]) \
                  & (0<t_points[1,:])
          out = np.concatenate((t_points[:,mask], pc_2d[2:,mask]), axis=0)
        else:
          mask = None
          out = np.concatenate((t_points, pc_2d[2:,:]), axis=0)

        return out, mask


    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        img = cv2.imread(img_path)
        return img, anns, img_info, img_path

    def _flip_pc(self, pc_2d, width):
        pc_2d[0,:] = width - 1 - pc_2d[0,:]
        return pc_2d

    def _load_data(self,index,input_dict):
        #coco = self.coco
        #img_dir = self.img_dir
        #img_id = self.images[index]
        #img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)
        #img = info['radar_path'][]
        prefix = "/root/lanyun-tmp/Pillar_fusion/data/nuscenes/v1.0-trainval"
        img_path = input_dict["image_paths"]
        img_path = [f"{prefix}/{path}" for path in img_path]

        #print(img_path)
        for path in img_path:
            img = cv2.imread(path)  # Try to load the image
        #img = cv2.imread(img_path)
        #img = input_dict["camera_imgs"]
        #img_info = camera_info[]
        #img_info = self.infos[index]
        #return img, anns, img_info, img_path
        return img, img_path

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
    
        inp = (inp.astype(np.float32) / 255.)
        #if 'train' in self.split and not no_color_aug:
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def _process_pc(self, pc_2d, pc_3d, img, inp_trans, out_trans, info):    
        img_height, img_width = img.shape[0], img.shape[1]

        # transform points
        mask = None
        if len(pc_feat_lvl) > 0:
          pc_feat, mask = self._transform_pc(pc_2d, out_trans, output_w, output_h)
          pc_hm_feat = np.zeros((len(pc_feat_lvl), output_h, output_w), np.float32)
    
        if mask is not None:
          pc_N = np.array(sum(mask))
          pc_2d = pc_2d[:,mask]
          pc_3d = pc_3d[:,mask]
        else:
          pc_N = pc_2d.shape[1]

        # create point cloud pillars
        if pc_roi_method == "pillars":
          pillar_wh = self.create_pc_pillars(img, info, pc_2d, pc_3d, inp_trans, out_trans)    

        # generate point cloud channels
        for i in range(pc_N-1, -1, -1):
          for feat in pc_feat_lvl:
            point = pc_feat[:,i]
            depth = point[2]
            ct = np.array([point[0], point[1]])
            ct_int = ct.astype(np.int32)

            if pc_roi_method == "pillars":
              wh = pillar_wh[:,i]
              b = [max(ct[1]-wh[1], 0), 
                  ct[1], 
                  max(ct[0]-wh[0]/2, 0), 
                  min(ct[0]+wh[0]/2, output_w)]
              b = np.round(b).astype(np.int32)
        
            elif pc_roi_method == "hm":
              radius = (1.0 / depth) * r_a + r_b
              radius = gaussian_radius((radius, radius))
              radius = max(0, int(radius))
              x, y = ct_int[0], ct_int[1]
              height, width = pc_hm_feat.shape[1:3]
              left, right = min(x, radius), min(width - x, radius + 1)
              top, bottom = min(y, radius), min(height - y, radius + 1)
              b = np.array([y - top, y + bottom, x - left, x + right])
              b = np.round(b).astype(np.int32)
        
            if feat == 'pc_dep':
              channel = pc_feat_channels['pc_dep']
              pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = depth
        
            if feat == 'pc_vx':
              vx = pc_3d[8,i]
              channel = pc_feat_channels['pc_vx']
              pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = vx
        
            if feat == 'pc_vz':
              vz = pc_3d[9,i]
              channel = pc_feat_channels['pc_vz']
              pc_hm_feat[channel, b[0]:b[1], b[2]:b[3]] = vz
        '''
        print("####################################################")
        print("successfully loading....2222222222222222222")
        print("####################################################")
        '''

        return pc_2d, pc_3d, pc_hm_feat

    def _load_pc_data(self, img, info, inp_trans, out_trans, input_dict ,flipped=0):
        img_height, img_width = img.shape[0], img.shape[1]




        #radar_pc = get_radar_with_sweeps(index, max_sweeps=max_sweeps)
        #radar_pc = np.array(img_info.get('radar_pc', None))
        #radar_pc = extract_radar_point_clouds(self, index, max_sweeps=6)
        radar_pc = np.array(info.get('radar_pc', None))
        #print(radar_pc.shape)
        if radar_pc is None:
          return None, None, None, None

        # calculate distance to points
        depth = radar_pc[2,:]
    
        # filter points by distance
        if max_pc_dist > 0:
          mask = (depth <= max_pc_dist)
          radar_pc = radar_pc[:,mask]
          depth = depth[mask]

        # add z offset to radar points
        if pc_z_offset != 0:
          radar_pc[1,:] -= pc_z_offset
        
        W, H = input_dict["ori_shape"]
        #cam_info = info["cams"].get(cam)
        # map points to the image and filter ones outside
        pc_2d, mask = map_pointcloud_to_image(radar_pc, np.array(info['camera_intrinsic_2']), img_shape=(W,H))
        pc_3d = radar_pc[:,mask] 

        # sort points by distance
        ind = np.argsort(pc_2d[2,:])
        pc_2d = pc_2d[:,ind]
        pc_3d = pc_3d[:,ind]

        # flip points if image is flipped
        if flipped:
          pc_2d = self._flip_pc(pc_2d,  img_width)
          pc_3d[0,:] *= -1  # flipping the x dimension
          pc_3d[8,:] *= -1  # flipping x velocity (x is right, z is front)

        pc_2d, pc_3d, pc_dep = self._process_pc(pc_2d, pc_3d, img, inp_trans, out_trans, info)
        pc_N = np.array(pc_2d.shape[1])

        # pad point clouds with zero to avoid size mismatch error in dataloader
        n_points = min(max_pc, pc_2d.shape[1])
        pc_z = np.zeros((pc_2d.shape[0], max_pc))
        pc_z[:, :n_points] = pc_2d[:, :n_points]
        pc_3dz = np.zeros((pc_3d.shape[0], max_pc))
        pc_3dz[:, :n_points] = pc_3d[:, :n_points]
        
        return pc_z, pc_N, pc_dep, pc_3dz

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not not_rand_crop) and not disturb:
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = scale
            cf = shift
            # if type(s) == float:
            #   s = [s, s]
            temp = np.random.randn()*cf
            c[0] += s * np.clip(temp, -2*cf, 2*cf)
            c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
        if np.random.random() < aug_rot:
            rf = rotate
            rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
        else:
            rot = 0
        
        return c, aug_s, rot
 ###########################################################################   
    def _get_calib(self, info, width, height):
        if 'calib' in info:
            calib = np.array(info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
        return calib

#################################################################################################################
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
       
        info = copy.deepcopy(self.infos[index])
        #print("type(info)",type(info))
        #print("type(self.infos)",type(self.infos))
        #print("info################",len(self.infos))
        #print("info['lidar_path']",info['lidar_path'])
        #print("self.infos",self.infos)
        #print("info",info) 
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }
        if self.use_map:
            input_dict['ref_from_car'] = info['ref_from_car']
            input_dict['car_from_global'] = info['car_from_global']
            input_dict['location'] = info['location']

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
            if 'gt_boxes_2d' in info:
                info['gt_boxes_2d'] = info['gt_boxes_2d'][info['empty_mask']]
                input_dict.update({
                    'gt_boxes2d': info['gt_boxes_2d'] if mask is None else info['gt_boxes_2d'][mask]
                })

        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        #############################################
        
        pointcloud=True
        tracking=False
        same_aug_pre=True

        #img, anns, img_info, img_path = self._load_data(index)
        img, img_path = self._load_data(index,input_dict)
        height, width = img.shape[0], img.shape[1]

        ## sort annotations based on depth form far to near
        #new_anns = sorted(anns, key=lambda k: k['depth'], reverse=True)

        ## Get center and scale from image
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0 if not not_max_crop \
          else np.array([img.shape[1], img.shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0
        
        
        '''
        ## data augmentation for training set
        if 'train' in self.split:
          c, aug_s, rot = self._get_aug_param(c, s, width, height)
          s = s * aug_s
          if np.random.random() < flip:
            flipped = 1
            img = img[:, ::-1, :]
            anns = self._flip_anns(anns, width)
        '''
        
        c, aug_s, rot = self._get_aug_param(c, s, width, height)
        s = s * aug_s
        trans_input = get_affine_transform(c, s, rot, [input_w, input_h])
        trans_output = get_affine_transform(c, s, rot, [output_w, output_h])
        inp = self._get_input(img, trans_input)
        data_dict.update({'image':inp})
        #print('image shape:',np.shape(data_dict['image']))
        #print('input_dict[camera_imgs]',np.shape(input_dict['camera_imgs']))
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}
        #  load point cloud data
        
        
        if pointcloud:
          pc_2d, pc_N, pc_dep, pc_3d = self._load_pc_data(img,info, trans_input, trans_output, input_dict, flipped=0)
          data_dict.update({ 'pc_2d': pc_2d,
                       'pc_3d': pc_3d,
                       'pc_N': pc_N, 
                       'pc_dep': pc_dep })
        pre_cts, track_ids = None, None
        
        num_classes = self.num_categories
        data_dict['pc_hm'] = np.zeros((len(pc_feat_lvl), output_h, output_w), np.float32)
        calib = self._get_calib(info, width, height)
        num_objs = min(len(info['annotations']), self.max_objs)
        for k in range(num_objs):
            ann = info['annotations'][k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > num_classes or cls_id <= -999:
                continue
            bbox, bbox_amodal = self._get_bbox_output(ann['bbox'], trans_output, height, width)
            if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(data_dict, cls_id, bbox)
                continue
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            dist_thresh = get_dist_thresh(calib, ct, ann['dim'], ann['alpha'])
            pc_dep_to_hm(data_dict['pc_hm'], data_dict['pc_dep'], ann['depth'], bbox, dist_thresh)
        data_dict['calib'] = calib
        
        '''
        if tracking:
          pre_image, pre_anns, frame_dist, pre_img_info = self._load_pre_data(
            img_info['video_id'], img_info['frame_id'], 
            img_info['sensor_id'] if 'sensor_id' in img_info else 1)
          if flipped:
            pre_image = pre_image[:, ::-1, :].copy()
            pre_anns = self._flip_anns(pre_anns, width)
            if pc_2d is not None:
              pc_2d = self._flip_pc(pc_2d,  width)
          if same_aug_pre and frame_dist != 0:
            trans_input_pre = trans_input 
            trans_output_pre = trans_output
          else:
            c_pre, aug_s_pre, _ = self._get_aug_param(
              c, s, width, height, disturb=True)
            s_pre = s * aug_s_pre
            trans_input_pre = get_affine_transform(
              c_pre, s_pre, rot, [input_w, input_h])
            trans_output_pre = get_affine_transform(
              c_pre, s_pre, rot, [output_w, output_h])
          pre_img = self._get_input(pre_image, trans_input_pre)
          pre_hm, pre_cts, track_ids = self._get_pre_dets(
            pre_anns, trans_input_pre, trans_output_pre)
          data_dict['pre_img'] = pre_img
          if pre_hm:
            data_dict['pre_hm'] = pre_hm
          if pointcloud:
            pre_pc_2d, pre_pc_N, pre_pc_hm, pre_pc_3d = self._load_pc_data(pre_img, pre_img_info, 
                trans_input_pre, trans_output_pre, flipped)
        '''
        return data_dict
    
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],dtype=np.float32)
        return bbox
    
    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0: # ignore all classes
            self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1])
        else:
            # mask out one specific class
            self._ignore_region(ret['hm'][abs(cls_id) - 1, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1])
        if ('hm_hp' in ret) and cls_id <= 1:
            self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, int(bbox[0]): int(bbox[2]) + 1])
    
    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],[bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
        for t in range(4):
            rect[t] =  affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal
    
    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict
    
    def evaluation_map_segmentation(self, results):
        import torch

        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"]
            label = result["gt_masks_bev"]
            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None].cpu() >= thresholds
            label = label[:, :, None].cpu()

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics
    
    def create_groundtruth_database(self, used_classes=None, max_sweeps=10, with_cam_gt=False, share_memory=False):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'
        
        #####################
        radar_database_save_path = self.root_path / f'radar_gt_database_{max_sweeps}sweeps_withvelo'
        radar_database_save_path.mkdir(parents=True, exist_ok=True)

        if share_memory:
            db_data_save_path_radar = self.root_path / f'nuscenes_{max_sweeps}sweeps_withvelo_radar.npy'
            radar_offset_cnt = 0
            stacked_gt_radar = []
        #####################

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        if share_memory:
            db_data_save_path_lidar = self.root_path / f'nuscenes_{max_sweeps}sweeps_withvelo_lidar.npy'
            lidar_offset_cnt = 0
            stacked_gt_lidar = []
        if with_cam_gt:
            img_database_save_path = self.root_path / f'img_gt_database_{max_sweeps}sweeps_withvelo'
            img_database_save_path.mkdir(parents=True, exist_ok=True)
            if share_memory:
                db_data_save_path_img = self.root_path / f'nuscenes_{max_sweeps}sweeps_withvelo_img.npy'
                img_offset_cnt = 0
                stacked_gt_img = []

        for idx in tqdm(range(len(self.infos))):
            
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            #######################################################
            #radar_points = self.get_radar_with_sweeps(idx, max_sweeps=max_sweeps)
            radar_points = np.array(info['radar_pc'])  # 从 info 中获取雷达点云数据
            #######################################################
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            if with_cam_gt:
                if gt_boxes.shape[0] == 0:
                    continue
                gt_boxes_2d = info['gt_boxes_2d']
                gt_boxes_2d = gt_boxes_2d[info['empty_mask']]
                image_paths = []
                for _, camera_info in info["cams"].items():
                    image_paths.append(camera_info["data_path"])
                images = []
                for name in image_paths:
                    images.append(cv2.imread(str(self.root_path / name)))
                object_img_patches = common_utils.crop_img_list(images,gt_boxes_2d)

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                ######################################################
                '''
                # 处理 RADAR 数据
                radar_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(torch.from_numpy(radar_points[:, 0:3]).unsqueeze(dim=0).float().cuda(),torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()).long().squeeze(dim=0).cpu().numpy()

                radar_filename = '%s_%s_%d_radar.bin' % (sample_idx, gt_names[i], i)
                radar_filepath = radar_database_save_path / radar_filename
                radar_gt_points = radar_points[radar_box_idxs_of_pts == i]
                radar_gt_points[:, :3] -= gt_boxes[i, :3]

                if not share_memory:
                    with open(radar_filepath, 'w') as f:
                        radar_gt_points.tofile(f)
                '''
                ####################################################

                gt_points[:, :3] -= gt_boxes[i, :3]
                if not share_memory:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)
                if with_cam_gt:
                    img_filename = '%s_%s_%d.png' % (sample_idx, gt_names[i], i)
                    img_filepath = img_database_save_path / img_filename
                    if not share_memory:
                        cv2.imwrite(str(img_filepath),object_img_patches[i])


                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    #############################################
                    '''
                    db_info['radar_path'] = str(radar_filepath.relative_to(self.root_path))
                    db_info['num_points_in_radar'] = radar_gt_points.shape[0]
                    '''
                    ##################################################
                    if share_memory:
                        stacked_gt_lidar.append(gt_points)
                        db_info['global_data_offset'] = [lidar_offset_cnt, lidar_offset_cnt + gt_points.shape[0]]
                        lidar_offset_cnt += gt_points.shape[0]

                    ##########################################
                    '''
                    if share_memory:
                        stacked_gt_radar.append(radar_gt_points)
                        db_info['global_data_offset_radar'] = [radar_offset_cnt, radar_offset_cnt + radar_gt_points.shape[0]]
                        radar_offset_cnt += radar_gt_points.shape[0]
                    '''
                    ##########################################


                    if with_cam_gt:
                        img_db_path = str(img_filepath.relative_to(self.root_path))  # gt_database/xxxxx.png
                        db_info.update({'box2d_camera':gt_boxes_2d[i],'img_path':img_db_path,'img_shape':object_img_patches[i].shape})
                        if share_memory:
                            flatten_img_patches = object_img_patches[i].reshape(-1,3)
                            stacked_gt_img.append(flatten_img_patches)
                            db_info['global_data_offset_img'] = [img_offset_cnt, img_offset_cnt + flatten_img_patches.shape[0]]
                            img_offset_cnt += flatten_img_patches.shape[0]
                   
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        if share_memory:
            stacked_gt_lidar = np.concatenate(stacked_gt_lidar, axis=0)
            np.save(db_data_save_path_lidar, stacked_gt_lidar)
            if with_cam_gt:
                stacked_gt_img = np.concatenate(stacked_gt_img, axis=0)
                np.save(db_data_save_path_img, stacked_gt_img)
        #########################################
        '''
        if share_memory:
            stacked_gt_radar = np.concatenate(stacked_gt_radar, axis=0)
            np.save(db_data_save_path_radar, stacked_gt_radar)
        '''
        ###########################################
    '''
    def create_groundtruth_database(self, used_classes=None, max_sweeps=10, with_cam_gt=False, share_memory=False):
        import torch

        # 设置保存路径
        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'
    
        radar_database_save_path = self.root_path / f'radar_gt_database_{max_sweeps}sweeps_withvelo'
        radar_database_save_path.mkdir(parents=True, exist_ok=True)

        # 内存共享相关配置
        if share_memory:
            db_data_save_path_radar = self.root_path / f'nuscenes_{max_sweeps}sweeps_withvelo_radar.npy'
            radar_offset_cnt = 0
            stacked_gt_radar = []

        # 创建主数据库文件夹
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        if share_memory:
            db_data_save_path_lidar = self.root_path / f'nuscenes_{max_sweeps}sweeps_withvelo_lidar.npy'
            lidar_offset_cnt = 0
            stacked_gt_lidar = []

        if with_cam_gt:
            img_database_save_path = self.root_path / f'img_gt_database_{max_sweeps}sweeps_withvelo'
            img_database_save_path.mkdir(parents=True, exist_ok=True)
            if share_memory:
                db_data_save_path_img = self.root_path / f'nuscenes_{max_sweeps}sweeps_withvelo_img.npy'
                img_offset_cnt = 0
                stacked_gt_img = []

        # 遍历所有样本
        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)

            # 获取已经处理好的雷达点云数据
            radar_points = np.array(info['radar_pc'])  # 从 info 中获取雷达点云数据

            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            # 获取与目标框相关的点云索引
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            # 如果需要摄像头 GT
            if with_cam_gt:
                if gt_boxes.shape[0] == 0:
                    continue
                gt_boxes_2d = info['gt_boxes_2d']
                gt_boxes_2d = gt_boxes_2d[info['empty_mask']]
                image_paths = []
                for _, camera_info in info["cams"].items():
                    image_paths.append(camera_info["data_path"])
                images = []
                for name in image_paths:
                    images.append(cv2.imread(str(self.root_path / name)))
                object_img_patches = common_utils.crop_img_list(images, gt_boxes_2d)

            # 遍历每个目标框
            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                # 处理 RADAR 数据
                radar_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(radar_points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

                radar_filename = '%s_%s_%d_radar.bin' % (sample_idx, gt_names[i], i)
                radar_filepath = radar_database_save_path / radar_filename
                radar_gt_points = radar_points[radar_box_idxs_of_pts == i]
                radar_gt_points[:, :3] -= gt_boxes[i, :3]  # 相对于目标框的偏移

                # 如果不使用内存共享，则保存到文件
                if not share_memory:
                    with open(radar_filepath, 'w') as f:
                        radar_gt_points.tofile(f)

                # 对于 LiDAR 数据的处理
                gt_points[:, :3] -= gt_boxes[i, :3]
                if not share_memory:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                # 如果需要摄像头图像的处理
                if with_cam_gt:
                    img_filename = '%s_%s_%d.png' % (sample_idx, gt_names[i], i)
                    img_filepath = img_database_save_path / img_filename
                    if not share_memory:
                        cv2.imwrite(str(img_filepath), object_img_patches[i])

                # 保存数据库信息
                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                           'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}

                    # 添加雷达路径和点云数目
                    db_info['radar_path'] = str(radar_filepath.relative_to(self.root_path))
                    db_info['num_points_in_radar'] = radar_gt_points.shape[0]

                    # 处理内存共享
                    if share_memory:
                        stacked_gt_lidar.append(gt_points)
                        db_info['global_data_offset'] = [lidar_offset_cnt, lidar_offset_cnt + gt_points.shape[0]]
                        lidar_offset_cnt += gt_points.shape[0]

                        stacked_gt_radar.append(radar_gt_points)
                        db_info['global_data_offset_radar'] = [radar_offset_cnt, radar_offset_cnt + radar_gt_points.shape[0]]
                        radar_offset_cnt += radar_gt_points.shape[0]

                    if with_cam_gt:
                        img_db_path = str(img_filepath.relative_to(self.root_path))  # gt_database/xxxxx.png
                        db_info.update({'box2d_camera': gt_boxes_2d[i], 'img_path': img_db_path, 'img_shape': object_img_patches[i].shape})
                        if share_memory:
                            flatten_img_patches = object_img_patches[i].reshape(-1, 3)
                            stacked_gt_img.append(flatten_img_patches)
                            db_info['global_data_offset_img'] = [img_offset_cnt, img_offset_cnt + flatten_img_patches.shape[0]]
                            img_offset_cnt += flatten_img_patches.shape[0]

                    # 更新数据库
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]

            # 输出数据库信息
            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))

            # 保存数据库信息到文件
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)

            # 保存内存共享数据
            if share_memory:
                stacked_gt_lidar = np.concatenate(stacked_gt_lidar, axis=0)
                np.save(db_data_save_path_lidar, stacked_gt_lidar)
                if with_cam_gt:
                    stacked_gt_img = np.concatenate(stacked_gt_img, axis=0)
                    np.save(db_data_save_path_img, stacked_gt_img)

            # 保存雷达数据
            if share_memory:
                stacked_gt_radar = np.concatenate(stacked_gt_radar, axis=0)
                np.save(db_data_save_path_radar, stacked_gt_radar)
    '''

def create_nuscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))
    ############################################
    # 填充信息：增加 with_radar 参数
    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam, #with_radar=with_radar
    )


    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)




if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    parser.add_argument('--with_cam_gt', action='store_true', default=False, help='use camera gt database or not')
    parser.add_argument('--share_memory', action='store_true', default=False, help='use share memory or not')

    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            with_cam=args.with_cam
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(
            max_sweeps=dataset_cfg.MAX_SWEEPS, 
            with_cam_gt=args.with_cam_gt, 
            share_memory=args.share_memory
        )
