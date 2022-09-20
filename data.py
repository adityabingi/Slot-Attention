import numpy as np
import torch
import h5py
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class TetrominoesDataset(Dataset):

    """
    Tetrominoes dataset from HDf5 file

    The .h5 dataset is assumed to be organized as follows:
    {train|val|test}/
        imgs/  <-- a tensor of shape [dataset_size,H,W,C]
        masks/ <-- a tensor of shape [dataset_size,num_objects,H,W,C]
        factors/  <-- a tensor of shape [dataset_size,...]
    """
    def __init__(self, data_h5_path, 
                 masks = False, factors=False, d_set='train'):

        super(TetrominoesDataset, self).__init__()
        self.h5_path = str(data_h5_path)
        self.d_set = d_set.lower()
        self.masks = masks
        self.factors = factors

        # transforms.ConvertImageDtype(torch.float32)
        img_transform = [transforms.ToTensor(),
                         transforms.Lambda(lambda x: (x - 0.5) * 2)]

        self.transform_img = transforms.Compose(img_transform)

    def __len__(self):
        with h5py.File(self.h5_path,  'r') as data:
            data_size, _, _, _ = data[self.d_set]['imgs'].shape
            return data_size

    def __getitem__(self, i):

        with h5py.File(self.h5_path,  'r') as data:
            outs = {}
            outs['imgs'] = self.transform_img(np.uint8(data[self.d_set]['imgs'][i].astype('float32')))
            if self.masks:
                outs['masks'] = np.transpose(data[self.d_set]['masks'][i].astype('float32'), (0,3,1,2))
            if self.factors:
                outs['factors'] = data[self.d_set]['factors'][i]
            return outs

