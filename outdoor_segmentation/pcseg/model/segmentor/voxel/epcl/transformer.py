import torch.nn as nn
import torch
try:
    from pointnet2_ops import pointnet2_utils
except:
    import pointnet2_utils
from .util_modules import SparseGroup, EmbeddingEncoder, PointNetFeaturePropagation
import clip

class EPCLPreEncoder(nn.Module):
    def __init__(self, num_group, group_size, enc_dim, in_channels) -> None:
        super().__init__()
        self.group_divider = SparseGroup(num_group=num_group, group_size=group_size)
        self.encoder = EmbeddingEncoder(encoder_channel=enc_dim, in_channels=in_channels)

    def forward(self, pts, feats=None):
   
        batch_xyz = []
        _, center, batch_idx, batch_xyz, local_feats = self.group_divider(pts, feats)
        group_input_tokens = self.encoder(local_feats)
        return center, group_input_tokens, batch_idx, batch_xyz
    
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
    
class ClipTransformer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        emb_dim = kwargs.get("embed_dim")
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, emb_dim),
        )
        num_tokens = 20
        self.te_tok = torch.arange(num_tokens).long()
        self.te_encoder = TaskEmbEncoder(
            token_num=num_tokens,
            emb_dim=emb_dim
            )
        self.blocks, self.cls_token, self.cls_pos = self.clip_transformer()
     

    def clip_transformer(self, freeze=True):
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = 'ViT-B/32'
        clip_model = clip.load(clip_model_name, device=device)[0]

        clip_vit = clip_model.visual
        clip_vit.float()
        if freeze:
            print("------------------- frozen ---------------------")
            clip_vit.eval()
            for k, v in clip_vit.named_parameters():
                v.requires_grad = False
                print(k)
            print("------------------- frozen ---------------------")
        return clip_vit.transformer, clip_vit.class_embedding, clip_vit.positional_embedding[0]

    def get_prompt(self, batch_size, te_token, te_encoder, device):
        prompts_tokens = te_token.expand(batch_size,
                                        -1).view(batch_size, -1).to(device)
        past_key_values = te_encoder(
            prompts_tokens)

        return past_key_values

    def forward(self, feats, xyz):
       
        B, N, C = feats.shape

        pos = self.pos_embed(xyz)

        cls_tok = self.cls_token.expand(B, 1, C)
        cls_pos = self.cls_pos.expand(B, 1, C)
        
        feats = torch.cat([cls_tok, feats], dim=1)   # [B, N+1, C]
        pos = torch.cat([cls_pos, pos], dim=1)
        feats = feats + pos
        task_emb = self.get_prompt(batch_size=B,
                                   te_token=self.te_tok,
                                   te_encoder=self.te_encoder,
                                   device=feats.device)
        feats = torch.cat([feats, task_emb], dim=1)
        
        # load clip transformer
        new_feats = self.blocks(feats)
        return xyz, new_feats[:, 1:N+1, :], None

def pc_norm(pc):
        """ pc: [batch_size, num_points, num_channels], return [batch_size, num_points, num_channels] """
        centroid = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2, keepdim=True)), dim=1, keepdim=True).values
        pc = pc / m
        return pc

class EPCLEncoder(nn.Module):
    
    def __init__(self, args) -> None:
        
        super().__init__()
        self.tokenizer = EPCLPreEncoder(
            num_group=args.NUM_GROUP,
            group_size=args.GROUP_SIZE,   
            enc_dim=args.ENC_DIM,
            in_channels=args.TOKENIZER_DIM
        )
        self.encoder = ClipTransformer(embed_dim=args.ENC_DIM)
        self.upsamping = PointNetFeaturePropagation(
            args.ENC_DIM, [args.TOKENIZER_DIM])
        self.voxel_size = args.VOXEL_SIZE
        
    def forward(self, xyz, feats):
        
        pre_enc_xyz, pre_enc_features, pre_enc_inds, ori_xyz = self.tokenizer(
            xyz, feats)
        pos = pc_norm(pre_enc_xyz*self.voxel_size)
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pos
        )
        ori_size_feature = []
        for i, one_batch_ori_xyz in enumerate(ori_xyz):
            one_batch_ori_size_features = self.upsamping(
                one_batch_ori_xyz, enc_xyz[i:i+1, ...], None, enc_features[i:i+1, ...])
            ori_size_feature.append(
                one_batch_ori_size_features.squeeze(0))  # [N,C]
        features = torch.cat(ori_size_feature, dim=0)
        new_features = features + feats
    
        return new_features