import clip
import torch
import numpy as np

def get_clip_text_feats(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    with torch.no_grad():
        texts_emb = clip.tokenize(texts).to(device)
        texts_feats = clip_model.encode_text(texts_emb)

    return texts_feats

def get_text_emd():
    text_prompts = "point cloud of a big "  # refer to point-clip
    # modelnet40
    cat_file = "./data/ModelNet/modelnet40_normal_resampled/modelnet40_shape_names.txt"
    cls_names = [line.rstrip() for line in open(cat_file)]  

    texts = []
    for n in cls_names:
        texts.append(text_prompts + n)

    texts_feats = get_clip_text_feats(texts)

    np.save('clip_texts_emb.npy', texts_feats.cpu().numpy())