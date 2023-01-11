import glob
import os
from argparse import Namespace

import data_loading.data_loader.cuda_data_loader as cl
import data_loading.data_loader.xla_data_loader as xl
import numpy as np
import torch
from data_loading.pytorch_loader import PytTrain, PytVal
from runtime.logging import mllog_event
from torch.utils.data import Dataset
import glob
import gcsfs
fs = gcsfs.GCSFileSystem()

def list_files_with_pattern(path, files_pattern):
    data = sorted(fs.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(fs.glob((os.path.join(path, files_pattern))))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]

def get_data_split(path: str, num_shards: int, shard_id: int, use_brats: bool, foldidx: int):
    # if use_brats:
    #     listfile = "brats_evaluation_cases_{}.txt".format(foldidx)
    # else:
    #     listfile = "evaluation_cases.txt"

    # with open(listfile, "r") as f:
    #     val_cases_list = f.readlines()
    #val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    val_cases_list = ['00025', '00048', '00056', '00060', '00071', '00096', '00097', '00112', '00127', '00128', '00132', '00139', '00146', '00151', '00152', '00158', '00162', '00172', '00185', '00192', '00206', '00214', '00231', '00234', '00237', '00241', '00246', '00249', '00259', '00269', '00283', '00289', '00294', '00297', '00301', '00304', '00334', '00341', '00348', '00364', '00386', '00400', '00404', '00410', '00412', '00417', '00429', '00430', '00431', '00440', '00442', '00466', '00469', '00485', '00498', '00500', '00507', '00512', '00528', '00530', '00540', '00551', '00561', '00568', '00569', '00570', '00572', '00575', '00576', '00579', '00582', '00590', '00601', '00607', '00613', '00616', '00620', '00622', '00623', '00630', '00636', '00659', '00667', '00675', '00679', '00680', '00689', '00709', '00718', '00724', '00727', '00733', '00740', '00744', '00757', '00758', '00759', '00774', '00775', '00778', '00788', '00799', '00808', '00818', '00824', '00831', '01001', '01002', '01005', '01013', '01043', '01044', '01046', '01055', '01057', '01059', '01060', '01061', '01071', '01082', '01102', '01112', '01120', '01123', '01129', '01132', '01135', '01138', '01146', '01151', '01154', '01159', '01160', '01161', '01164', '01169', '01172', '01177', '01179', '01180', '01191', '01206', '01209', '01214', '01216', '01223', '01224', '01232', '01237', '01240', '01242', '01244', '01250', '01256', '01262', '01265', '01268', '01270', '01271', '01272', '01275', '01278', '01280', '01284', '01286', '01289', '01298', '01308', '01316', '01320', '01322', '01323', '01326', '01336', '01338', '01340', '01342', '01343', '01351', '01352', '01353', '01354', '01356', '01361', '01364', '01368', '01369', '01378', '01379', '01387', '01392', '01394', '01405', '01407', '01412', '01423', '01424', '01436', '01441', '01447', '01448', '01450', '01451', '01452', '01459', '01460', '01467', '01468', '01470', '01473', '01475', '01479', '01482', '01485', '01489', '01490', '01493', '01514', '01516', '01520', '01525', '01539', '01549', '01553', '01554', '01560', '01561', '01564', '01567', '01571', '01574', '01576', '01579', '01583', '01591', '01592', '01599', '01601', '01602', '01604', '01605', '01608', '01610', '01611', '01626', '01628', '01631', '01643', '01654', '01663']
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
    
    mllog_event(key="train_samples", value=len(imgs_train), sync=False)
    mllog_event(key="eval_samples", value=len(imgs_val), sync=False)
    imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)
    return imgs_train, imgs_val, lbls_train, lbls_val
class SyntheticDataset(Dataset):
    def __init__(
        self,
        channels_in=1,
        channels_out=3,
        shape=(128, 128, 128),
        device="cpu",
        layout="NCDHW",
        scalar=False,
    ):
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


def get_data_loaders(flags: Namespace, num_shards: int, global_rank: int, device: torch.device):
    """Initializes and returns (train_data_loader, val_data_loader)

    :param Namespace flags: the runtime arguments
    :param int num_shards: number of shards for the train dataset
    :param int global_rank: global rank associated with the device
    :param torch.device device: the device to use for MpDeviceLoader
    :return: the tuple (train_loader, val_loader)
    :rtype: Union[Tuple[pl.MpDeviceLoader, pl.MpDeviceLoader], Tuple[DataLoader, DataLoader]]
    """
    if flags.loader == "synthetic":
        train_dataset = SyntheticDataset(channels_in=4, channels_out=4, scalar=True, shape=flags.input_shape, layout=flags.layout)
        val_dataset = SyntheticDataset(channels_in=4, channels_out=4, 
            scalar=True, shape=flags.val_input_shape, layout=flags.layout
        )

    elif flags.loader == "pytorch":
        x_train, x_val, y_train, y_val = get_data_split(
            flags.data_dir, num_shards, shard_id=global_rank, use_brats=flags.use_brats, foldidx = flags.fold_idx
        )
        train_data_kwargs = {
            "patch_size": flags.input_shape,
            "oversampling": flags.oversampling,
            "seed": flags.seed,
        }
        train_dataset = PytTrain(x_train, y_train, **train_data_kwargs)
        val_dataset = PytVal(x_val, y_val)
    else:
        raise ValueError(f"Loader {flags.loader} unknown. Valid loaders are: synthetic, pytorch")

    if flags.device == "xla":
        train_loader, val_loader = xl.get_data_loaders(
            flags, num_shards, global_rank, device, train_dataset, val_dataset
        )
    elif flags.device == "cuda":
        train_loader, val_loader = cl.get_data_loaders(
            flags, num_shards, train_dataset, val_dataset
        )
    else:
        raise ValueError(f"Device {flags.device} unknown. Valid devices are: cuda, xla")
    return train_loader, val_loader
