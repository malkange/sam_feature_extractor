'''
using virtual environment named 'pytorch3d'
codes for extracting multiple masks from SAM
SemanticKITTI

save minimal information
n_anchors, 3, 258 (feature, scores, position)
'''
import os, fnmatch
import sys
import cv2
import torch
import time
import glob

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pytorch3d.ops.sample_farthest_points as sample_farthest_points
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from utils import *
# from iostream import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


ckpt = '/data2/SAMSeg3D/samckpt/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=ckpt)
sam.to(device=device)

predictor = SamPredictor(sam)
mask_gen = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.92)
img1 = Image.open('target.png').convert('RGB')

starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation

# load intrinsic
P_dict = {}
Tr_dict = {}

n_anchors = 200

cmap = plt.get_cmap()

# seq_root = '/home/poscoict/Desktop/c3d_semKITTI_refined/dataset/sequences'  # todo
root = '/data2/SAMSeg3D/SemKITTI/dataset/sequences'

save_root = '/data2/SAMSeg3D/SemKITTI_processed/dataset/sequences/'
# save_root = 'D:/Dataset/semKITTI-processed/dataset/sequences/'
# save_root = 'Z:/SAMSeg3D/semKITTI-processed2/dataset/sequences/'
res_save_root = 'res'
# seqs = [os.path.join(root, x) for x in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]]
seqs = [os.path.join(root, x) for x in [ "04"]]
# seqs = ["10"]
# seqs = [os.path.join(root, x) for x in ["11", "12", "13"]]  #  previous sequences are already done in local
for seq in seqs:
    # for seq_dir in seq_dir_root:
    seq_root = os.path.join(root, seq)
    # seq_dir = seq_dir_root[seq]

    # load camera parameters
    with open(os.path.join(seq_root, 'calib.txt'), 'r') as calib:
        P = []
        for idx in range(4):
            line = calib.readline().rstrip('\n')[4:]
            data = line.split(" ")
            P.append(np.array(data, dtype=np.float32).reshape(3, -1))
        P_dict[seq + "_left"] = P[2]
        # P_dict[seq + "_right"] = P[3]
        line = calib.readline().rstrip('\n')[4:]
        data = line.split(" ")
        Tr_dict[seq] = np.array(data, dtype=np.float32).reshape((3, -1))

    # get ids in the sequence
    ids = [os.path.basename(x).split('.')[0] for x in glob.glob(seq_root + '/velodyne/*.bin')]

    destination_root = os.path.join(save_root, seq[-2:])
    os.makedirs(destination_root, exist_ok=True)
    os.makedirs(os.path.join(destination_root, 'seg_fea'), exist_ok=True)
    os.makedirs(os.path.join(destination_root, 'img_fea'), exist_ok=True)


    print("\n\ncurrent seq : ", seq_root)

    # for idx in trange(len(image_dirs)):
    # ids = ['002396']
    for ii in trange(len(ids)):
        # breakpoint()
        # starter1.record()
        id = ids[ii]
        img1 = os.path.join(seq_root, 'image_2', id + '.png')
        img1 = Image.open(img1).convert('RGB')

        max_len = max(img1.size)
        # load pcd and label
        pts = np.fromfile(os.path.join(seq_root, 'velodyne', id + '.bin'), dtype=np.float32).reshape((-1, 4))
        points_ids = np.linspace(0, len(pts) - 1, len(pts)).astype(int)

        # point to img projection
        pcoord1, mask1 = mappcd2img(P_dict, Tr_dict, seqs[0], pts[:, :3], img1.size, "left")

        # pcoord1_cp = pcoord1.copy()
        sub_pts = pts[mask1]
        uvs = pcoord1[mask1].astype(np.int32)  # uv coordinate check
        float_uvs = pcoord1[mask1]
        # pt_ids = points_ids[mask1]
        # breakpoint()

        uvs_t = torch.Tensor(uvs).to(torch.int32) #.to(device)
        sub_pts_t = torch.Tensor(sub_pts)[:,:3] #.to(device)
        predictor.set_image(np.array(img1))


        #######################
        # processing for generating a single level id map
        #######################

        scoremap = np.zeros((img1.size[1], img1.size[0]), dtype=int)
        masks = mask_gen.generate(np.array(img1))

        id_list = []
        sc_list = []
        iou_list = []
        for mid, mask in enumerate(masks):
            tmp_sc = np.ones((uvs_t.shape[0],)) * -1
            tmp_id = np.ones((uvs_t.shape[0],)) * -1
            tmp_ms = mask['segmentation'][uvs_t[:, 1], uvs_t[:, 0]]
            tmp_id[tmp_ms] = mid
            tmp_sc[tmp_ms] = mask['predicted_iou']
            id_list.append(tmp_id)
            sc_list.append(tmp_sc)
            iou_list.append(mask['predicted_iou'])
        id_map = np.stack(id_list, 0)
        sc_map = np.stack(sc_list, 0)
        iou_arr = np.array(iou_list)

        sc_argsort = np.argsort(sc_map, axis=0)
        sc_arg_top3 = sc_argsort[-3:, :]

        id_top3 = id_map[sc_arg_top3, np.repeat(np.arange(sc_map.shape[-1])[None, :], 3, 0)]
        valid_ids = np.unique(id_top3)
        if len(valid_ids) > 255:
            VALID_MAX = 20
            ## Sub-parts of a single object cannot exceeds over 20 parts/ --> Texture details

            id_v, id_c = np.unique(id_top3[2], return_counts=True)
            for id_t in id_v[1:]:
                sec_lev, sec_rev = np.unique(id_top3[1, id_top3[2] == id_t], return_inverse=True)
                if sec_lev.shape[0] < VALID_MAX + 1:
                    continue

                sec_lev[VALID_MAX + 1:] = -1
                id_top3[1, id_top3[2] == id_t] = sec_lev[sec_rev]

            valid_ids = np.unique(id_top3)
            if len(valid_ids) > 255:
                breakpoint()

            # breakpoint()

        valid_ids_seq = np.zeros_like(id_top3) - 1
        valid_ids_map = np.arange(valid_ids.shape[0]) - 1
        for t_id in range(valid_ids.shape[0]):
            valid_ids_seq[id_top3 == valid_ids[t_id]] = valid_ids_map[t_id]

        valid_ids_seq = valid_ids_seq.astype(np.uint8)

        id_mask = torch.Tensor(
            np.stack([masks[int(t_id)]['segmentation'] for t_id in valid_ids if t_id > -1], 0)).cuda()

        img_fea = predictor.get_image_embedding()

        fea_resam = extract_seg_fea_sj(img_fea, img1.size[-2:])
        tem = []
        for lgs in id_mask.chunk(id_mask.shape[0]):
            tem.append((fea_resam[None, ...] * lgs[:, None, ...]).sum((3, 4))[0])

        final_feature = torch.cat((tem), 0)

        pix_mask = np.zeros(img_fea.shape[-2:]).astype(np.bool_)

        float_uvs[:, 0] = float_uvs[:, 0] / (img1.size[0] - 1) * (img_fea.shape[-1] - 1)
        float_uvs[:, 1] = float_uvs[:, 1] / (img1.size[1] - 1) * (img_fea.shape[-1] - 1)
        floor_uvs = np.floor(float_uvs).astype(np.uint16)
        ceil_uvs = np.ceil(float_uvs).astype(np.uint16)
        # (ff)
        pix_mask[floor_uvs[:, 0], floor_uvs[:, 1]] = True
        # (fc)
        pix_mask[floor_uvs[:, 0], ceil_uvs[:, 1]] = True
        # (cf)
        pix_mask[ceil_uvs[:, 0], floor_uvs[:, 1]] = True
        # (cc)
        pix_mask[ceil_uvs[:, 0], ceil_uvs[:, 1]] = True
        pix_loc = np.where(pix_mask)

        feat_mask = img_fea[0][:, pix_loc[0], pix_loc[1]]
        # breakpoint()
        # if len(valid_ids_seq.T)!=len(uvs):
        #     breakpoint()

        # continue


        torch.save(feat_mask.squeeze().half(),  os.path.join(destination_root, 'img_fea', str(id).zfill(6) + ".pt"))
        torch.save(torch.Tensor(np.stack((pix_loc[0], pix_loc[1]), 0)),os.path.join(destination_root, 'img_fea', str(id).zfill(6) + "_idx.pt"))
        torch.save(final_feature,os.path.join(destination_root, 'seg_fea', str(id).zfill(6) + ".pt"))
        torch.save(torch.Tensor(valid_ids_seq),os.path.join(destination_root, 'seg_fea', str(id).zfill(6) + "_idx.pt"))
        # exit(0)
        # torch.cuda.empty_cache()

