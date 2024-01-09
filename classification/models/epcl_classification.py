import torch
import torch.nn as nn
from .build import MODELS
from utils import misc
from utils.logger import *

import clip

### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

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

class TaskEmbEncoder(torch.nn.Module):
    def __init__(
        self,
        token_num=40,
        emb_dim=384
    ):
        super().__init__()
        # Use a two-layer MLP to encode the prefix
        self.embedding = torch.nn.Embedding(token_num, emb_dim)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, te: torch.Tensor):
        te_tokens = self.embedding(te)
        past_key_values = self.trans(te_tokens)

        return past_key_values


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(nn.Conv1d(3, 128, 1),
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
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature],
                            dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
    
    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(
            -1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group,
                                         self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

# finetune model
@MODELS.register_module()
class EPCLClassification(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.group_divider = Group(num_group=self.num_group,
                                   group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # point position encoding
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                       nn.Linear(128, self.trans_dim))

        self.use_task_emb = config.use_task_emb

        if self.use_task_emb:
            self.te_tok = torch.arange(self.cls_dim).long()
            self.te_encoder = TaskEmbEncoder(
                token_num=self.cls_dim,
                emb_dim=self.trans_dim
            )

        # load clip visual transformer
        clip_model_name = 'ViT-B/32'
        print_log(f"Loading CLIP-{clip_model_name}...")
        clip_model = clip.load(clip_model_name, device=self.device)[0]

        self.clip_vit = clip_model.visual
        self.clip_vit.float()
        
        self.clip_vit.eval()
        for k, v in self.clip_vit.named_parameters():
            v.requires_grad = False

        # finetune cls token
        self.cls_token = self.clip_vit.class_embedding.reshape(1, 1, -1)
        self.cls_token.requires_grad = True
        self.cls_pos = self.clip_vit.positional_embedding[0].reshape(1, 1, -1)
        self.cls_pos.requires_grad = True

        self.norm = nn.LayerNorm(self.trans_dim)
        vit_drop=0.
        self.vit_dropout = nn.Dropout(vit_drop)
        self.proj2emb = nn.Sequential(nn.Linear(512+768, 512),
                                               nn.BatchNorm1d(512),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.2),
                                               nn.Linear(512, 512),
                                               nn.LayerNorm(512))
        
        self.cls_head_finetune = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

    def get_task_emb(self, batch_size, te_token, te_encoder):
        prompts_tokens = te_token.expand(batch_size,
                                        -1).view(batch_size, -1).to(
                                            self.device)
        past_key_values = te_encoder(
            prompts_tokens)

        return past_key_values

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B 64 C

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1,
                                           -1)  # [B.1,C]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1,
                                      -1)  # [B,1,C]

        pos = self.pos_embed(center)  # [B,64,C]

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # [B,1+64,384]
        pos = torch.cat((cls_pos, pos), dim=1)  # [B,1+64,C]

        self.clip_vit.eval()
        x = x + pos
        x = self.vit_dropout(x)
        x = self.clip_vit.ln_pre(x)
        if self.use_task_emb:
            task_emb = self.get_task_emb(batch_size=cls_tokens.size(0),
                                           te_token=self.te_tok,
                                           te_encoder=self.te_encoder)
            x = torch.cat([x, task_emb], dim=1) # [B, 1+N+num_cls, C]

        x = x.permute(1, 0, 2)
        x = self.clip_vit.transformer(x)
        x = x.permute(1, 0, 2)

        x_clip_cls = self.clip_vit.ln_post(x[:, 0, :])
        x_clip_cls = (x_clip_cls @ self.clip_vit.proj) # 768 --> 512
        x = self.norm(x[:, 1:, :])
        concat_f = torch.cat([x_clip_cls, x.max(1)[0]], dim=-1)

        tok_emb = self.proj2emb(concat_f)
        logits = self.cls_head_finetune(tok_emb)

        return tok_emb, logits
