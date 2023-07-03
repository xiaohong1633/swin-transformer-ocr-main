import torch
import argparse
import time
from pathlib import Path

from utils import load_setting, load_tokenizer
from models import  SwinTransformerOCR
from dataset import  LmdbCollate

def tail(name):
    name = str(name)
    num = name[name.rfind("_")+1:name.rfind(".")]
    return int(num)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/demo.yaml",
                        help="Experiment settings")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    # load
    save_path = "ataset/dict_en_num_token.pkl"

    tokenizer = load_tokenizer(save_path)
    model = SwinTransformerOCR(cfg, tokenizer)
    saved = torch.load("checkpoints/checkpoints-epoch=104-accuracy=0.96095.ckpt", map_location=device)
    model.load_state_dict(saved['state_dict'])
    collate = LmdbCollate(cfg, tokenizer=tokenizer, is_train=False)

    target = Path("/home/xiaohong/Pictures/tt")
    if target.is_dir():
        target = list(target.glob("*.jpg")) + list(target.glob("*.png"))
    else:
        target = [target]
    target = sorted(target, key=lambda x: tail(x))
    for image_fn in target:
        start = time.time() * 1000
        x = collate.ready_image(image_fn)
        text = model.predict(x)
        print("{} [{}]sec | image_fn : {}".format(image_fn, time.time()*1000-start, text))
        print("---over---")
