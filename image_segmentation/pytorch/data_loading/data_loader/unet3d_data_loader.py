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
import tensorflow.io as io

def load_data(path, files_pattern):
    data = sorted(io.gfile.glob((os.path.join(path, files_pattern))))
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
    if use_brats:
        listfile = "brats_evaluation_cases_{}.txt".format(foldidx)
    else:
        listfile = "evaluation_cases.txt"

    with open(listfile, "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = ≈
    #val_cases_list = ['00025', '00048', '00056', '00060', '00071', '00096', '00097', '00112', '00127', '00128', '00132', '00139', '00146', '00151', '00152', '00158', '00162', '00172', '00185', '00192', '00206', '00214', '00231', '00234', '00237', '00241', '00246', '00249', '00259', '00269', '00283', '00289', '00294', '00297', '00301', '00304', '00334', '00341', '00348', '00364', '00386', '00400', '00404', '00410', '00412', '00417', '00429', '00430', '00431', '00440', '00442', '00466', '00469', '00485', '00498', '00500', '00507', '00512', '00528', '00530', '00540', '00551', '00561', '00568', '00569', '00570', '00572', '00575', '00576', '00579', '00582', '00590', '00601', '00607', '00613', '00616', '00620', '00622', '00623', '00630', '00636', '00659', '00667', '00675', '00679', '00680', '00689', '00709', '00718', '00724', '00727', '00733', '00740', '00744', '00757', '00758', '00759', '00774', '00775', '00778', '00788', '00799', '00808', '00818', '00824', '00831', '01001', '01002', '01005', '01013', '01043', '01044', '01046', '01055', '01057', '01059', '01060', '01061', '01071', '01082', '01102', '01112', '01120', '01123', '01129', '01132', '01135', '01138', '01146', '01151', '01154', '01159', '01160', '01161', '01164', '01169', '01172', '01177', '01179', '01180', '01191', '01206', '01209', '01214', '01216', '01223', '01224', '01232', '01237', '01240', '01242', '01244', '01250', '01256', '01262', '01265', '01268', '01270', '01271', '01272', '01275', '01278', '01280', '01284', '01286', '01289', '01298', '01308', '01316', '01320', '01322', '01323', '01326', '01336', '01338', '01340', '01342', '01343', '01351', '01352', '01353', '01354', '01356', '01361', '01364', '01368', '01369', '01378', '01379', '01387', '01392', '01394', '01405', '01407', '01412', '01423', '01424', '01436', '01441', '01447', '01448', '01450', '01451', '01452', '01459', '01460', '01467', '01468', '01470', '01473', '01475', '01479', '01482', '01485', '01489', '01490', '01493', '01514', '01516', '01520', '01525', '01539', '01549', '01553', '01554', '01560', '01561', '01564', '01567', '01571', '01574', '01576', '01579', '01583', '01591', '01592', '01599', '01601', '01602', '01604', '01605', '01608', '01610', '01611', '01626', '01628', '01631', '01643', '01654', '01663']
    allfilelist = ['00000', '00002', '00003', '00005', '00006', '00008', '00009', '00011', '00012', '00014', '00016', '00017', '00018', '00019', '00020', '00021', '00022', '00024', '00025', '00026', '00028', '00030', '00031', '00032', '00033', '00035', '00036', '00043', '00044', '00045', '00046', '00048', '00049', '00051', '00052', '00053', '00054', '00056', '00058', '00059', '00060', '00061', '00062', '00063', '00064', '00066', '00068', '00070', '00071', '00072', '00074', '00077', '00078', '00081', '00084', '00085', '00087', '00088', '00089', '00090', '00094', '00095', '00096', '00097', '00098', '00099', '00100', '00101', '00102', '00103', '00104', '00105', '00106', '00107', '00108', '00109', '00110', '00111', '00112', '00113', '00115', '00116', '00117', '00118', '00120', '00121', '00122', '00123', '00124', '00126', '00127', '00128', '00130', '00131', '00132', '00133', '00134', '00136', '00137', '00138', '00139', '00140', '00142', '00143', '00144', '00146', '00147', '00148', '00149', '00150', '00151', '00152', '00154', '00155', '00156', '00157', '00158', '00159', '00160', '00162', '00165', '00166', '00167', '00170', '00171', '00172', '00176', '00177', '00178', '00183', '00184', '00185', '00186', '00187', '00188', '00191', '00192', '00193', '00194', '00195', '00196', 
    '00199', '00201', '00203', '00204', '00206', '00207', '00209', '00210', '00211', '00212', '00214', '00216', '00217', '00218', '00219', '00220', '00221', '00222', '00227', '00228', '00230', '00231', '00233', '00234', '00235', '00236', '00237', '00238', '00239', '00240', '00241', '00242', '00243', '00246', '00247', '00249', '00250', '00251', '00253', '00254', '00258', '00259', '00260', '00261', '00262', '00263', '00266', '00267', '00269', '00270', '00271', '00273', '00274', '00275', '00280', '00281', '00282', '00283', '00284', '00285', '00286', '00288', '00289', '00290', '00291', '00292', '00293', '00294', '00296', '00297', '00298', '00299', '00300', '00301', '00303', '00304', '00305', '00306', '00309', '00310', '00311', '00312', '00313', '00314', '00316', '00317', '00318', '00320', '00321', '00322', '00324', '00325', '00327', '00328', '00329', '00331', '00332', '00334', '00336', '00338', '00339', '00340', '00341', '00343', '00344', '00346', '00347', '00348', '00349', '00350', '00351', '00352', '00353', '00356', '00359', '00360', '00364', '00366', '00367', '00369', '00370', '00371', '00373', '00375', '00376', '00377', '00378', '00379', '00380', '00382', '00383', '00386', '00387', '00388', '00389', '00390', '00391', '00392', '00395', '00397', '00399', '00400', 
    '00401', '00402', '00403', '00404', '00405', '00406', '00407', '00409', '00410', '00412', '00413', '00414', '00416', '00417', '00418', '00419', '00421', '00423', '00425', '00426', '00429', '00430', '00431', '00432', '00433', '00436', '00440', '00441', '00442', '00443', '00444', '00445', '00446', '00448', '00449', '00451', '00452', '00453', '00454', '00455', '00456', '00457', '00459', '00464', '00466', '00468', '00469', '00470', '00472', '00477', '00478', '00479', '00480', '00481', '00483', '00485', '00488', '00491', '00493', '00494', '00495', '00496', '00498', '00499', '00500', '00501', '00502', '00504', '00505', '00506', '00507', '00510', '00511', '00512', '00513', '00514', '00516', '00517', '00518', '00519', '00520', '00523', '00524', '00525', '00526', '00528', '00529', '00530', '00532', '00533', '00537', '00538', '00539', '00540', '00542', '00543', '00544', '00545', '00547', '00548', '00549', '00550', '00551', '00552', '00554', '00555', '00556', '00557', '00558', '00559', '00561', '00563', '00565', '00567', '00568', '00569', '00570', '00571', '00572', '00574', '00575', '00576', '00577', '00578', '00579', '00580', '00581', '00582', '00583', '00584', '00586', '00587', '00588', '00589', '00590', '00591', '00593', '00594', '00596', '00597', '00598', '00599', 
    '00601', '00602', '00604', '00605', '00606', '00607', '00608', '00610', '00611', '00612', '00613', '00615', '00616', '00618', '00619', '00620', '00621', '00622', '00623', '00624', '00625', '00626', '00628', '00630', '00631', '00636', '00638', '00639', '00640', '00641', '00642', '00645', '00646', '00649', '00650', '00651', '00652', '00654', '00655', '00656', '00657', '00658', '00659', '00661', '00663', '00667', '00668', '00674', '00675', '00676', '00677', '00679', '00680', '00682', '00683', '00684', '00685', '00686', '00687', '00688', '00689', '00690', '00691', '00692', '00693', '00694', '00697', '00698', '00703', '00704', '00705', '00706', '00707', '00708', '00709', '00714', '00715', '00716', '00718', '00723', '00724', '00725', '00727', '00728', '00729', '00730', '00731', '00732', '00733', '00734', '00735', '00736', '00737', '00739', '00740', '00742', '00744', '00746', '00747', '00750', '00751', '00753', '00756', '00757', '00758', '00759', '00760', '00764', '00765', '00767', '00768', '00772', '00773', '00774', '00775', '00777', '00778', '00780', '00781', '00782', '00784', '00787', '00788', '00789', '00791', '00792', '00793', '00795', '00796', '00797', '00799', '00800', '00801', '00802', '00803', '00804', '00805', '00806', '00807', '00808', '00809', '00810', 
    '00811', '00814', '00816', '00818', '00819', '00820', '00823', '00824', '00828', '00830', '00831', '00834', '00836', '00837', '00838', '00839', '00840', '00999', '01000', '01001', '01002', '01003', '01004', '01005', '01007', '01008', '01009', '01010', '01011', '01012', '01013', '01014', '01015', '01016', '01017', '01018', '01019', '01020', '01021', '01022', '01023', '01024', '01025', '01026', '01027', '01028', '01029', '01030', '01031', '01032', '01033', '01034', '01035', '01036', '01037', '01038', '01039', '01040', '01041', '01042', '01043', '01044', '01045', '01046', '01047', '01048', '01049', '01050', '01051', '01052', '01053', '01054', '01055', '01056', '01057', '01058', '01059', '01060', '01061', '01062', '01063', '01064', '01065', '01066', '01067', '01068', '01069', '01070', '01071', '01072', '01073', '01074', '01075', '01076', '01077', '01078', '01079', '01080', '01081', '01082', '01083', '01084', '01085', '01086', '01087', '01088', '01089', '01090', '01091', '01092', '01093', '01094', '01095', '01096', '01097', '01098', '01099', '01100', '01101', '01102', '01103', '01104', '01105', '01106', '01107', '01108', '01109', '01110', '01111', '01112', '01113', '01114', '01115', '01116', '01117', '01118', '01119', '01120', '01121', '01122', '01123', '01124',
    '01125', '01126', '01127', '01128', '01129', '01130', '01131', '01132', '01133', '01134', '01135', '01136', '01137', '01138', '01139', '01140', '01141', '01142', '01143', '01144', '01145', '01146', '01147', '01148', '01149', '01150', '01151', '01152', '01153', '01154', '01155', '01156', '01157', '01158', '01159', '01160', '01161', '01162', '01163', '01164', '01165', '01166', '01167', '01168', '01169', '01170', '01171', '01172', '01173', '01174', '01175', '01176', '01177', '01178', '01179', '01180', '01181', '01182', '01183', '01184', '01185', '01186', '01187', '01188', '01189', '01190', '01191', '01192', '01193', '01194', '01195', '01196', '01197', '01198', '01199', '01200', '01201', '01202', '01203', '01204', '01205', '01206', '01207', '01208', '01209', '01210', '01211', '01212', '01213', '01214', '01215', '01216', '01217', '01218', '01219', '01220', '01221', '01222', '01223', '01224', '01225', '01226', '01227', '01228', '01229', '01230', '01231', '01232', '01233', '01234', '01235', '01236', '01237', '01238', '01239', '01240', '01241', '01242', '01243', '01244', '01245', '01246', '01247', '01248', '01249', '01250', '01251', '01252', '01253', '01254', '01255', '01256', '01257', '01258', '01259', '01260', '01261', '01262', '01263', '01264', '01265', '01266', 
    '01267', '01268', '01269', '01270', '01271', '01272', '01273', '01274', '01275', '01276', '01277', '01278', '01279', '01280', '01281', '01282', '01283', '01284', '01285', '01286', '01287', '01288', '01289', '01290', '01291', '01292', '01293', '01294', '01295', '01296', '01297', '01298', '01299', '01300', '01301', '01302', '01303', '01304', '01305', '01306', '01307', '01308', '01309', '01310', '01311', '01312', '01313', '01314', '01315', '01316', '01317', '01318', '01319', '01320', '01321', '01322', '01323', '01324', '01325', '01326', '01327', '01328', '01329', '01330', '01331', '01332', '01333', '01334', '01335', '01336', '01337', '01338', '01339', '01340', '01341', '01342', '01343', '01344', '01345', '01346', '01347', '01348', '01349', '01350', '01351', '01352', '01353', '01354', '01355', '01356', '01357', '01358', '01359', '01360', '01361', '01362', '01363', '01364', '01365', '01366', '01367', '01368', '01369', '01370', '01371', '01372', '01373', '01374', '01375', '01376', '01377', '01378', '01379', '01380', '01381', '01382', '01383', '01384', '01385', '01386', '01387', '01388', '01389', '01390', '01391', '01392', '01393', '01394', '01395', '01396', '01397', '01398', '01399', '01400', '01401', '01402', '01403', '01404', '01405', '01406', '01407', '01408', 
    '01409', '01410', '01411', '01412', '01413', '01414', '01415', '01416', '01417', '01418', '01419', '01420', '01421', '01422', '01423', '01424', '01425', '01426', '01427', '01428', '01429', '01430', '01431', '01432', '01433', '01434', '01435', '01436', '01437', '01438', '01439', '01440', '01441', '01442', '01443', '01444', '01445', '01446', '01447', '01448', '01449', '01450', '01451', '01452', '01453', '01454', '01455', '01456', '01457', '01458', '01459', '01460', '01461', '01462', '01463', '01464', '01465', '01466', '01467', '01468', '01469', '01470', '01471', '01472', '01473', '01474', '01475', '01476', '01477', '01478', '01479', '01480', '01481', '01482', '01483', '01484', '01485', '01486', '01487', '01488', '01489', '01490', '01491', '01492', '01493', '01494', '01495', '01496', '01497', '01498', '01499', '01500', '01501', '01502', '01503', '01504', '01505', '01506', '01507', '01508', '01509', '01510', '01511', '01512', '01513', '01514', '01515', '01516', '01517', '01518', '01519', '01520', '01521', '01522', '01523', '01524', '01525',
    '01526', '01527', '01528', '01529', '01530', '01531', '01532', '01533', '01534', '01535', '01536', '01537', '01538', '01539', '01540', '01541', '01542', '01543', '01544', '01545', '01546', '01547', '01548', '01549', '01550', '01551', '01552', '01553', '01554', '01555', '01556', '01557', '01558', '01559', '01560', '01561', '01562', '01563', '01564', '01565', '01566', '01567', '01568', '01569', '01570', '01571', '01572', '01573', '01574', '01575', '01576', '01577', '01578', '01579', '01580', '01581', '01582', '01583', '01584', '01585', '01586', '01587', '01588', '01589', '01590', '01591', '01592', '01593', '01594', '01595', '01596', '01597', '01598', '01599', '01600', '01601', '01602', '01603', '01604', '01605', '01606', '01607', '01608', '01609', '01610', '01611', '01612', '01613', '01614', '01615', '01616', '01617', '01618', '01619', '01620', '01621', '01622', '01623', '01624', '01625', '01626', '01627', '01628', '01629', '01630', '01631', '01632', '01633', '01634', '01635', '01636', '01637', '01638', '01639', '01640', '01641', '01642', '01643', '01644', '01645', '01646', '01647', '01648', '01649', '01650', '01651', '01652', '01653', '01654', '01655', '01656', '01657', '01658', '01659', '01660', '01661', '01662', '01663', '01664', '01665', '01666']
    imgs = [ os.path.join(path, "BraTS2021_" + idx + "_x.npy") in allfilelist]
    lbls = [ os.path.join(path, "BraTS2021_" + idx + "_y.npy") in allfilelist]
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
    #imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)
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
