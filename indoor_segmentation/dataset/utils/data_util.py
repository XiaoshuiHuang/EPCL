import os
import random

import numpy as np
import SharedArray as SA

import torch
from torch_scatter import scatter_mean
from .voxelize import voxelize, semantickitti_voxelize

seed=0
# pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

def collate_fn_limit(batch, max_batch_points, logger):
    coord, xyz, feat, label = list(zip(*batch))
    offset, count = [], 0
    
    new_coord, new_xyz, new_feat, new_label = [], [], [], []
    k = 0
    for i, item in enumerate(xyz):

        count += item.shape[0]
        if count > max_batch_points:
            break

        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_label.append(label[i])

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in xyz])
        s_now = sum([x.shape[0] for x in new_xyz[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))
    
    data_dict = \
        {
            'coord': torch.cat(new_xyz[:k]),
            'feat': torch.cat(new_feat[:k]),
            'target': torch.cat(new_label[:k]),
            'offset': torch.IntTensor(offset[:k]),
        }
    return data_dict
    # return torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(new_label[:k]), torch.IntTensor(offset[:k])

def collation_fn_voxelmean(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
    coords, xyz, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)

    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)

    data_dict = \
        {
            'coord': xyz,
            'feat': feats,
            'target': labels,
            'offset': offset,
            'inds_recons': inds_recons,
        }
    return data_dict
    
    # return coords, xyz, feats, labels, offset, inds_recons

def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def area_crop(coord, area_rate, split='train'):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min; coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == 'train' or split == 'trainval':
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
    return crop_idx


def load_kitti_data(data_path):
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((-1, 4))  # xyz+remission
    return data


def load_kitti_label(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def data_prepare_v101(
    coord, feat, label, 
    split='train', voxel_size=0.04, voxel_max=None, 
    transform=None, shuffle_index=False):

    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max:
        if (label.shape[0] > voxel_max): 
            init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    
    return coord, feat, label

def data_prepare_scannet(
    coord, feat, label, 
    split='train', voxel_size=0.04, voxel_max=None, 
    transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def data_prepare_semantickitti(coord, feat, label, split='train', voxel_size=np.array([0.1, 0.1, 0.1]), voxel_max=None, transform=None, xyz_norm=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord_norm = coord - coord_min
    if split == 'train':
        uniq_idx = semantickitti_voxelize(coord_norm, voxel_size)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        if voxel_max and label.shape[0] > voxel_max:
            init_idx = np.random.randint(label.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
            coord_voxel = coord_voxel[crop_idx]
    else:
        idx_recon = semantickitti_voxelize(coord_norm, voxel_size, mode=1)

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    if split == 'train':
        coord_voxel = torch.LongTensor(coord_voxel)
        return coord_voxel, coord, feat, label
    else:
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coords_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        coord = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        return coords_voxel, coord, feat, label, idx_recon