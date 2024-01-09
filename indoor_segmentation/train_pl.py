from __future__ import print_function
import random
from copy import deepcopy

import torch # why should I located it here?
import os
import numpy as np
import cv2
cv2.setNumThreads(0)
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import yaml
from easydict import EasyDict

from utils.my_args import my_args
from model import get as get_model
from dataset import get as get_dataset

seed=0
pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False


# FIXME: should not be necessary, but some memory are not free between train and val
class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

# https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_mnist.py
def jaesungchoe():
    # ------------
    # args
    # ------------

    parser = my_args()
    args = parser.parse_args()

    # ------------
    # randomness or seed
    # ------------
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    
    # ------------
    # data
    # ------------
    dataset = get_dataset(args.dataset)
    train_loader_kwargs = \
        {
            "batch_size": args.train_batch,
            "num_workers": args.train_worker,
            "collate_fn": dataset.TrainValCollateFn,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": True,
        }
    train_loader = torch.utils.data.DataLoader(
        dataset.myImageFloder(args, mode=args.mode_train), **train_loader_kwargs)

    val_loader_kwargs = \
        {
            "batch_size": args.val_batch,
            "num_workers": args.val_worker,
            "collate_fn": dataset.TrainValCollateFn,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
        }
    val_loader = torch.utils.data.DataLoader(
        dataset.myImageFloder(args, mode=args.mode_eval), **val_loader_kwargs)
    # ------------
    # logger (neptune.__version__ == 0.14.2)
    # ------------
    if args.on_neptune:
        from pytorch_lightning.loggers import NeptuneLogger
        neptune_path = os.path.join(args.MYCHECKPOINT, 'neptune.npz')
        if args.resume and os.path.exists(neptune_path):
            neptune_info = np.load(neptune_path)
            logger = NeptuneLogger(
                api_key=args.neptune_key, 
                project=args.neptune_proj)
            logger._run_short_id = str(neptune_info['id'])
            print(">> re-use the neptune: id[%s]"%(neptune_info['id']))
        else:
            logger = NeptuneLogger(
                api_key=args.neptune_key, 
                project=args.neptune_proj)
            logger.experiment["sys/tags"].add('train_pl.py')
            for key, value in (vars(args)).items(): logger.experiment['params/' + key] = value
            neptune_file_path = os.path.join(args.MYCHECKPOINT, 'neptune.npz')
            neptune_id_str = str(logger._run_short_id)
            if (neptune_id_str != 'None') and (not os.path.exists(neptune_file_path)):
                np.savez(neptune_file_path, project=args.neptune_proj, id=neptune_id_str)
                for _ in range(10): print(">> saved the neptune: id[%s]"%(neptune_id_str))
    else:
        logger = None

    # ------------
    # model
    # ------------
    args.load_model = 'last.ckpt'
    resume_from_checkpoint = None
    file_path = os.path.join(args.MYCHECKPOINT, 'last.ckpt')
    if os.path.exists(file_path):
        resume_from_checkpoint = file_path
        model = get_model(args.model).load_from_checkpoint(
            resume_from_checkpoint, 
            args=args,
            strict=args.strict_load)
    else:
        model = get_model(args.model)(args=args)   

    # ------------
    # trainer
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.MYCHECKPOINT),
        filename='{epoch:03d}--{mIoU_val:.4f}--',
        monitor="mIoU_val",
        save_top_k=3,
        mode="max",
        save_last=True)
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu", 
        devices=args.NUM_GPUS, 
        strategy=DDPStrategy(find_unused_parameters=False), # 'ddp'
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, CudaClearCacheCallback()],
        check_val_every_n_epoch=args.CHECKPOINT_PERIOD,
    )
    
    if args.on_train:
        print("TRAIN START")
        trainer.fit(model, train_loader, val_loader)
        print("TRAIN END")

    if args.on_neptune:
        logger.experiment["sys/failed"] = False
        logger.experiment.stop()

if __name__ == '__main__':

    jaesungchoe()