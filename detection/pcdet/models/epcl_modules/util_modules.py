import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
try:
    from pointnet2_ops import pointnet2_utils
except:
    import pointnet2_utils

# @torch.no_grad()
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

@torch.no_grad()
def knn(xyz0, xyz1, k):
    """
    Given xyz0 with shape [B, N, C], xyz1 with shape [B, M, C], 
    going to find k nearest points for xyz1 in xyz0
    """
    cdist = torch.cdist(xyz1, xyz0) # [B, M, N]
    values, indices = torch.topk(cdist, k=k, dim=-1,largest=False)    # [B, M, K]
    return values, indices

class EmbeddingEncoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(nn.Conv1d(in_channels, 128, 1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1))

    def forward(self, point_groups):  
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.in_channels).transpose(2, 1)
        # encoder
        feature = self.first_conv(point_groups)  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature],
                            dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        
        return feature_global.reshape(bs, g, self.encoder_channel)

class SparseGroup(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        
    def group(self, xyz, feats=None):
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = knn(xyz, center, self.group_size)  # B G M
        batch_idx = idx # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(
            -1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group,
                                         self.group_size, 3).contiguous()
        # [B,G,M,C]
        patch_feats = None
        if feats is not None:
            patch_feats = feats.view(batch_size * num_points, -1)[idx, :]
            patch_feats = patch_feats.view(batch_size, self.num_group,
                                            self.group_size, -1).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, batch_idx, patch_feats
    
    def forward(self, xyz, feature=None):
        # batch ----- batch
        lengths = []
        all_neighborhood = []
        all_center = []
        all_batch_idx = []
        max_batch = torch.max(xyz[:, 0])
        for i in range(max_batch + 1):
            length = torch.sum(xyz[:, 0] == i)
            lengths.append(length)
        # batch ----- batch
        start = 0
        end = 0
        batch_xyz = []
        batch_feature = []
        for length in lengths:
            end += length
            one_batch_C = torch.unsqueeze(xyz[start:end, 1:], dim=0).contiguous().to(dtype=torch.float32)
            one_batch_F = torch.unsqueeze(feature[start:end, :], dim=0).contiguous().to(dtype=torch.float32)  # [B,N,C]
            neighborhood, center, batch_idx, patch_feats = self.group(one_batch_C, one_batch_F)
            start += length
            batch_xyz.append(one_batch_C)
            all_neighborhood.append(neighborhood)
            all_center.append(center)
            all_batch_idx.append(batch_idx)
            batch_feature.append(patch_feats)
        neighborhoods = torch.cat(all_neighborhood, dim=0) # [B,G,M,C]
        centers = torch.cat(all_center, dim=0) # [B,G,C]
        batch_idxs = torch.cat(all_batch_idx, dim=0) # [B,G,M]
        input_features = torch.cat(batch_feature, dim=0) # dim = 64
        return neighborhoods, centers, batch_idxs, batch_xyz, input_features
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                nn.Conv1d(last_channel, out_channel, 1).cuda())
            self.mlp_bns.append(nn.BatchNorm1d(out_channel).cuda())
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(
                points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points