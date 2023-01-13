import random
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import gcsfs
fs = gcsfs.GCSFileSystem()

def get_train_transforms():
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    train_transforms = transforms.Compose([rand_flip, cast, rand_scale, rand_noise])
    return train_transforms


class RandBalancedCrop:
    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling

    def __call__(self, data):
        image, label = data["image"], data["label"]
        if random.random() < self.oversampling:
            image, label, cords = self.rand_foreg_cropd(image, label)
        else:
            image, label, cords = self._rand_crop(image, label)
        data.update({"image": image, "label": label})
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.measurements.label(label==cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


class RandFlip:
    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            if random.random() < self.prob:
                data = self.flip(data, axis)
        return data


class Cast:
    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        data["image"] = data["image"].astype(self.types[0])
        data["label"] = data["label"].astype(self.types[1])
        return data


class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        image = data["image"]
        if random.random() < self.prob:
            factor = np.random.uniform(low=1.0-self.factor, high=1.0+self.factor, size=1)
            image = (image * (1 + factor)).astype(image.dtype)
            data.update({"image": image})
        return data


class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        image = data["image"]
        if random.random() < self.prob:
            scale = np.random.uniform(low=0.0, high=self.std)
            noise = np.random.normal(loc=self.mean, scale=scale, size=image.shape).astype(image.dtype)
            data.update({"image": image + noise})
        return data


class PytTrain(Dataset):
    def __init__(self, images, labels, dataset, **kwargs):
        self.dataset = dataset
        self.images, self.labels = images, labels
        self.train_transforms = get_train_transforms()
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        self.patch_size = patch_size
        self.rand_crop = RandBalancedCrop(patch_size=patch_size, oversampling=oversampling)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with fs.open(os.path.join(self.dataset, self.images[idx]), 'rb') as f, fs.open(os.path.join(self.dataset, self.labels[idx]), 'rb') as g:
            data = {"image": np.load(f), "label": np.load(g)}
        data = self.rand_crop(data)
        data = self.train_transforms(data)
        data["image"]=np.transpose(data["image"],(1,2,3,0))
        data["label"]=np.transpose(data["label"],(1,2,3,0))        
        return data["image"], data["label"]

class PytTrain(Dataset):
    def __init__(
        self,
        images, labels, 
        **kwargs): 
        self.images, self.labels = images, labels
        channels_in=4
        channels_out=4
        shape=(128, 128, 128)
        device="cpu"
        layout="NCDHW"
        scalar=True
        shape = tuple(shape)
        x_shape = (channels_in,) + shape if layout == "NCDHW" else shape + (channels_in,)
        self.x = torch.rand(
            (32, *x_shape), dtype=torch.float32, device=device, requires_grad=False
        )
        if scalar:
            self.y = torch.randint(
                low=0,
                high=channels_out - 1,
                size=(32, *shape),
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )
            self.y = torch.unsqueeze(self.y, dim=1 if layout == "NCDHW" else -1)
        else:
            y_shape = (channels_out,) + shape if layout == "NCDHW" else shape + (channels_out,)
            self.y = torch.rand(
                (32, *y_shape), dtype=torch.float32, device=device, requires_grad=False
            )

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return self.x[idx % 32], self.y[idx % 32]

class PytVal(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with fs.open(self.images[idx], 'rb') as f, fs.open(self.labels[idx], 'rb') as g:
            return np.load(f), np.load(g)
import os
import tensorflow as tf
def load_data(path, files_pattern):
    data = sorted(tf.io.gfile.glob((os.path.join(path, files_pattern))))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]

def get_data_split(path: str, num_shards: int, shard_id: int, use_brats: bool, foldidx: int):
    if use_brats:
        listfile = "brats_evaluation_cases_{}.txt".format(foldidx)
    else:
        listfile = "evaluation_cases.txt"

    with open(listfile, "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    if "tfrecord" in path:
        imgs = load_data(path, "*.tfrecord")
        imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
        for case_img in imgs:
            if case_img.split("_")[-1].split('.')[0] in val_cases_list:
                imgs_val.append(case_img)
            else:
                imgs_train.append(case_img)
        #mllog_event(key="train_samples", value=len(imgs_train), sync=False)
        #mllog_event(key="eval_samples", value=len(imgs_val), sync=False)
        imgs_val, lbls_val = split_eval_data(imgs_val, [], num_shards, shard_id)
        return imgs_train, imgs_val, lbls_train, lbls_val

    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    #mllog_event(key="train_samples", value=len(imgs_train), sync=False)
    #mllog_event(key="eval_samples", value=len(imgs_val), sync=False)
    return imgs_train, imgs_val, lbls_train, lbls_val
if __name__ == "__main__":
    path="gs://mlperf-dataset/data/2021_Brats_np/11_3d"

    x_train, x_val, y_train, y_val = get_data_split(
        path, 4, shard_id=0, use_brats=True, foldidx = 0
    )

    train_dataset = PytTrain(x_train, y_train, patch_size=[128,128,128], oversampling=0.4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        sampler=None,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    for image in train_dataset:
        print(image[0].shape)
        print(image[1].shape)

    val_dataset = TFVal(x_val)
    for image in val_dataset:
        print(image[0].shape)
        print(image[1].shape)