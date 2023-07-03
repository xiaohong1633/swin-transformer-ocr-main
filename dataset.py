import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import lmdb
import sys
import six
import re
import pickle

def init_tokenizer(dict_path = "dataset/dict_en_num.txt", save_path = "dataset/dict_en_num_token.pkl"):
    token_id_dict = {
        "token2id": {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[OOV]": 3,
            " ": 4
        },
        "id2token": {
            0: "[PAD]",
            1: "[BOS]",
            2: "[EOS]",
            3: "[OOV]",
            4: " "
        }
    }

    idx = 5
    lines = open(dict_path, "r+").readlines()
    for item in lines:
        item = item.strip()
        if len(item) > 0:
            token_id_dict['token2id'][item] = idx
            token_id_dict['id2token'][idx] = item
            idx += 1

    tokenizer = Tokenizer(token_id_dict)

    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print("tokenizer saved in {}".format(save_path))
    return tokenizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, txt_fn):
        self.cfg = cfg
        self.images = []
        self.texts = []

        # build one
        self.token_id_dict = {
            "token2id": {
                "[PAD]": cfg.pad_token,
                "[BOS]": cfg.bos_token,
                "[EOS]": cfg.eos_token,
                "[OOV]": cfg.oov_token
            },
            "id2token": {
                cfg.pad_token: "[PAD]",
                cfg.bos_token: "[BOS]",
                cfg.eos_token: "[EOS]",
                cfg.oov_token: "[OOV]"
            }
        }

        skip_cnt, token_cnt = 0, 4
        with open(txt_fn, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    fn, text = line.strip().split('\t')
                except ValueError:
                    skip_cnt += 1
                    continue
                if cfg.max_seq_len < len(text) + 2:
                    # we will add [BOS] and [EOS]
                    skip_cnt += 1
                    continue
                self.images.append(fn)
                self.texts.append(text)
                if not cfg.load_tokenizer:
                    for token in text:
                        if token not in self.token_id_dict["token2id"]:
                            self.token_id_dict["token2id"][token] = token_cnt
                            self.token_id_dict["id2token"][token_cnt] = token
                            token_cnt += 1

        print(f"{len(self.images)} data loaded. ({skip_cnt} data skipped)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Read the image (use PIL Image to load unicode name images)
        if cfg.channels == 1, need to change alb transform methods
        """
        idx = idx % len(self.images)
        image = cv2.imread(str(Path(self.cfg.image_dir) / self.images[idx]))
        text = self.texts[idx]
        return image, text

class LmdbDataset(torch.utils.data.Dataset):

    def __init__(self, root):

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        if index == 0:
            index = 1
        # index = self.filtered_index_list[index]
        # index = 67
        # print("index:" + str(index))
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            # label_key = 'label-000000067'.encode()
            # print("index:"+str(label_key))
            #label-000000712 label-000000550
            label_txn = txn.get(label_key)
            if label_txn is None:
                print("error:"+str(label_key))
            label = label_txn.decode('utf-8')
            label = label.strip()
            # if label.__contains__(" "):
            #     label = label.replace(" ", "")
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:

                img = Image.open(buf).convert('RGB')  # for color image
                np_img = np.asarray(img)
                img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'
                np_img = np.asarray(img)
                img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            #if not self.opt.sensitive:
            #    label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            #out_of_char = f'[^{self.opt.character}]'
            #label = re.sub(out_of_char, '', label)
            # label = filter_label(self.opt.character, label)
            # label = label.replace()
            label = re.sub(" +", " ", label)
        return (img, label)

class LmdbCollate(object):
    def __init__(self, cfg, tokenizer, is_train=True):
        self.cfg = cfg
        self.tokenizer = tokenizer

        if is_train:
            self.transform = alb.Compose([
                        alb.Resize(cfg.height, cfg.width),
                        # alb.ShiftScaleRotate(shift_limit=0, scale_limit=(0., 0.15), rotate_limit=1,
                        #     border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        # alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                        #     value=[255, 255, 255], p=.5),
                        # alb.GaussNoise(10, p=.2),
                        # alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        alb.ImageCompression(95, p=.3),
                        alb.ToGray(always_apply=True),
                        alb.Normalize(),
                        # alb.Sharpen()
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = alb.Compose(
                [
                    alb.Resize(cfg.height, cfg.width),
                    alb.ImageCompression(95, p=.3),
                    alb.ToGray(always_apply=True),
                    alb.Normalize(),
                    # alb.Sharpen()
                    ToTensorV2(),
                ]
            )

    def __call__(self, batch):
        """
        return:
            images, (seq, mask)
        """
        np_images, texts = zip(*batch)
        images = []
        for img in np_images:
            try:
                images.append(self.transform(image=img)["image"])
            except TypeError as e:
                continue
        images = torch.stack(images)
        labels = self.tokenizer.encode(texts)

        return (images, labels)

    def resize_320_width(self, pil_img):
        pil_width = pil_img.width
        pil_height = pil_img.height
        ratio1 = pil_width * 1.0 / 320
        ratio2 = pil_height * 1.0 / 32

        new_w = int(pil_width * 1.0 / max(ratio1, ratio2))
        new_h = int(pil_height * 1.0 / max(ratio1, ratio2))

        new_img = pil_img.resize((new_w, new_h))
        bg_img = Image.fromarray(np.zeros((32, 320))).convert("RGB")
        bg_img.paste(new_img)
        return bg_img

    def ready_image(self, image):
        if isinstance(image, Path):
            pil_img = Image.open(image).convert("RGB")
            pil_img = self.resize_320_width(pil_img)

            image = np.array(pil_img)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError
        image = self.transform(image=image)["image"].unsqueeze(0)
        return image

class CustomCollate(object):
    def __init__(self, cfg, tokenizer, is_train=True):
        self.cfg = cfg
        self.tokenizer = tokenizer

        if is_train:
            self.transform = alb.Compose([
                        alb.Resize(112, 448),
                        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(0., 0.15), rotate_limit=1,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        alb.GaussNoise(10, p=.2),
                        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        alb.ImageCompression(95, p=.3),
                        alb.ToGray(always_apply=True),
                        alb.Normalize(),
                        # alb.Sharpen()
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = alb.Compose(
                [
                    alb.Resize(cfg.height, cfg.width),
                    alb.ImageCompression(95, p=.3),
                    alb.ToGray(always_apply=True),
                    alb.Normalize(),
                    # alb.Sharpen()
                    ToTensorV2(),
                ]
            )

    def __call__(self, batch):
        """
        return:
            images, (seq, mask)
        """
        np_images, texts = zip(*batch)
        images = []
        for img in np_images:
            try:
                images.append(self.transform(image=img)["image"])
            except TypeError as e:
                continue
        images = torch.stack(images)
        labels = self.tokenizer.encode(texts)

        return (images, labels)

    def ready_image(self, image):
        if isinstance(image, Path):
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError
        image = self.transform(image=image)["image"].unsqueeze(0)
        return image


class Tokenizer:
    def __init__(self, d):
        self.token2id = d["token2id"]
        self.id2token = d["id2token"]

    def __len__(self):
        return len(self.token2id)

    def encode(self, texts: list):
        """
        text:
            list of string form text
            [str, str, ...]
        return:
            tensors
        """
        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]
        oov = self.token2id["[OOV]"]

        ids = []
        for text in texts:
            encoded = [bos,]
            for token in text:
                try:
                    encoded.append(self.token2id[token])
                except KeyError:
                    encoded.append(oov)
            encoded.append(eos)
            ids.append(torch.tensor(encoded))

        seq = pad_sequence(ids, batch_first=True, padding_value=pad)
        mask = torch.zeros_like(seq)
        for i, encoded in enumerate(ids):
            mask[i, :len(encoded)] = 1

        return seq.long(), mask.bool()

    def decode(self, labels):
        """
        labels:
            [B, L] : B for batch size, L for Sequence Length
        """

        pad = self.token2id["[PAD]"]
        bos = self.token2id["[BOS]"]
        eos = self.token2id["[EOS]"]
        oov = self.token2id["[OOV]"]

        texts = []
        for label in labels.tolist():
            text = ""
            for id in label:
                if id == bos:
                    continue
                elif id == pad or id == eos:
                    break
                else:
                    text += self.id2token[id]

            texts.append(text)

        return texts


if __name__=="__main__":

    init_tokenizer(save_path="dataset/dict_en_num_token2.pkl")
    pass





