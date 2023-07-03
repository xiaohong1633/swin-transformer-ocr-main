import torch
import pytorch_lightning as pl
import argparse
from models import SwinTransformerOCR
from dataset import LmdbDataset, LmdbCollate, init_tokenizer
from utils import load_setting, CustomTensorBoardLogger, load_tokenizer

from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/demo.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=0,
                        help="Train experiment version")
    parser.add_argument("--load_tokenizer", "-bt", type=str, default="",
                        help="Load pre-built tokenizer")
    parser.add_argument("--num_workers", "-nw", type=int, default=15,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=64,
                        help="Batch size for training and validate")
    parser.add_argument("--resume_train", "-rt", type=str,
                        default="checkpoints/checkpoints-epoch=104-accuracy=0.96095.ckpt",
                        help="Resume train from certain checkpoint")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    # ----- dataset -----

    train_set = LmdbDataset(cfg.train_data)
    val_set = LmdbDataset(cfg.val_data)

    # ----- tokenizer -----
    if cfg.load_tokenizer:
        tokenizer = load_tokenizer(cfg.load_tokenizer)
    else:
        tokenizer = init_tokenizer(dict_path=cfg.dict_path)

    train_collate = LmdbCollate(cfg, tokenizer, is_train=True)
    val_collate = LmdbCollate(cfg, tokenizer, is_train=False)
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=train_collate)
    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=val_collate)

    cfg.num_train_step = len(train_dataloader)
    model = SwinTransformerOCR(cfg, tokenizer)

    logger = CustomTensorBoardLogger("tb_logs", name="model", version=cfg.version,
                                     default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="accuracy",
        dirpath=f"{cfg.save_path}/version_{cfg.version}",
        filename="checkpoints-{epoch:03d}-{accuracy:.5f}",
        save_top_k=3,
        mode="max",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    device_cnt = torch.cuda.device_count()
    strategy = pl.plugins.DDPPlugin(find_unused_parameters=False) if device_cnt > 1 else None
    trainer = pl.Trainer(gpus=device_cnt,
                         max_epochs=cfg.epochs,
                         logger=logger,
                         num_sanity_val_steps=1,
                         strategy=strategy,
                         callbacks=[ckpt_callback, lr_callback],
                         resume_from_checkpoint=cfg.resume_train if cfg.resume_train else None)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    print("---over---")
