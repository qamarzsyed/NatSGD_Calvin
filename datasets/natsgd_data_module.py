import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import pandas as pd
from .utils.episode_utils import load_dataset_statistics
from .utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import torch
from language_network import SBert
#from datasets import Dataset
logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"

class NatSGDDataset(Dataset):
    def __init__(self, dataframe, length):
        self.length = length
        self.start = dataframe["start"]
        self.end = dataframe["end"]
        self.bbox = dataframe["bbox"]
        self.lang = dataframe["lang"]
        self.gesture = dataframe["gesture"]
        self.actions = dataframe["actions"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = self.start.iloc[idx]
        end = self.end.iloc[idx]
        bbox = self.bbox.iloc[idx]
        lang = self.lang.iloc[idx]
        gesture = self.gesture.iloc[idx]
        actions = self.actions.iloc[idx]
        return start, end, bbox, lang, gesture, actions  

class NatSgdDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        training_repo_root: Optional[Path] = None,
        root_data_dir: str = "./../NatSGD_v0.9.0e.npz",
        image_data_dir: str = "./img_v0.9.0e",
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.images = os.listdir(image_data_dir)
        root_data_path = Path(root_data_dir)
        
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.norms = None
        self.use_shm = False#"shm_dataset" in self.datasets_cfg.vision_dataset._target_
        self.sbert = SBert("paraphrase-multilingual-MiniLM-L12-v2")
    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            if "CI" not in os.environ:
                print(f"No dataset found in {self.training_dir}.")
                print("For information how to download to full CALVIN dataset, please visit")
                print("https://github.com/mees/calvin/tree/main/dataset")
                print("Do you wish to download small debug dataset to continue training?")
                s = input("YES / no")
                if s == "no":
                    exit()
            logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)

        if self.use_shm:
            # When using shared memory dataset, initialize lookups
            train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.training_dir)
            train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()

            val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
            val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

            save_shm_lookup(train_shm_lookup, val_shm_lookup)
    def standardize(self, data_element = None):
        if data_element is None:
            return None
        #Get all Data Elements that will be fed into the same encoder to be the same size

        return
    def setup(self, unsplit_dataset=None, npz_norms=False):
        #Get Normalize Information
        #Get Language Annotations
        #train_datasets, val_datasets = {},{}
        self.train_datasets, self.val_datasets = {}, {}
        self.norms = {'baxter_state':{'std':[], 'mean':[]} ,'start_img':{'std':[], 'mean':[]},'goal_img':{'std':[], 'mean':[]}, 'bbox':{'std':[], 'mean':[]}, "gesture":{'std':[], 'mean':[]}}
        baxter_state = 0
        size = []
        bbox_blank = [np.zeros((17, 4))]
        bboxes = []
        gesture_length = []
        for s in range(unsplit_dataset.shape[0]):
            bbox = np.zeros((17, 4))
            for b in unsplit_dataset[s][14]:
                bbox[b[0]] += b[1:]
            bboxes.append(bbox)
            if s == 0:
                baxter_state = unsplit_dataset[s][12]
            else:
                baxter_state = np.append(baxter_state, unsplit_dataset[s][12], axis=0)
            gesture_length.append(np.array(unsplit_dataset[s][7]).shape[0])

        print("Max Length Gesture: ", max(gesture_length))
        bboxes = np.array(bboxes)
        
            #size.append(unsplit_dataset[0][12].shape[0])
        baxter_state = baxter_state.astype(float)
        self.norms['baxter_state']['std']  = np.std(baxter_state, axis=0)#(baxter_state - np.mean(baxter_state, axis=0))/np.std(baxter_state, axis=0)
        self.norms['baxter_state']['mean'] = np.mean(baxter_state,axis=0)

        self.norms['bbox']['std']  = np.std(bboxes, axis=0)#(baxter_state - np.mean(baxter_state, axis=0))/np.std(baxter_state, axis=0)
        self.norms['bbox']['mean'] = np.mean(bboxes,axis=0)

        for i in range(self.datasets_cfg['training'].shape[0]):
            if i ==0:
                self.train_datasets[i]={'start':[], 'end':[], 'bbox':[], 'lang':[], 'gesture':[], 'actions':[]}
            else:
                self.train_datasets[i]={'start':[], 'end':[], 'bbox':[], 'lang':[], 'gesture':[], 'actions':[]}
            start_obs = cv2.imread('./img_v0.9.0e/'+self.datasets_cfg['training'][i][10]+'.png')
            start_obs = cv2.resize(start_obs, (200,200), interpolation=cv2.INTER_AREA)
            end_obs = cv2.imread('./img_v0.9.0e/'+self.datasets_cfg['training'][i][11]+'.png')
            end_obs = cv2.resize(end_obs, (200,200), interpolation=cv2.INTER_AREA)
            bbox = np.zeros((200,200))
            for b in self.datasets_cfg['training'][i][14]:
                bbox[b[0],0:4] += b[1:]
            self.train_datasets[i]['start'] = torch.from_numpy(start_obs.astype(np.float16))
            self.train_datasets[i]['end'] = torch.from_numpy(end_obs.astype(np.float16))
            self.train_datasets[i]['bbox'] = torch.from_numpy(np.array([bbox,bbox,bbox]).astype(np.float16))
            #print("String: ", self.datasets_cfg['training'][i][11])
            lang_em = self.sbert.forward([str(self.datasets_cfg['training'][i][11])])
            self.train_datasets[i]['lang'] = lang_em
            actions_zero = np.zeros((175*16))
            actions_flat = self.datasets_cfg['training'][i][12].flatten()
            actions_zero[:actions_flat.shape[0]] = actions_flat
            self.train_datasets[i]['actions'] = torch.from_numpy(actions_zero.astype(np.float16))
            
            if len(self.datasets_cfg['training'][i][7]) == 0:
                self.train_datasets[i]['gesture'] = torch.from_numpy(np.zeros((200,200,3)).astype(np.float16))
            else:
                gest = np.array(self.datasets_cfg['training'][i][7])
                zz = np.zeros((3,250,50))
                zz[0:gest.shape[0],0:gest.shape[1],0:gest.shape[2]] += gest
                gest = np.reshape(zz,(250,50,3))
                gest_zeros = np.zeros((200,200,3))
                g1 = gest[0:125,0:50, 0:3]
                g2 = gest[125:,0:50, 0:3]
                gest_zeros[0:125,0:50,0:3] = g1
                gest_zeros[0:125,50:100,0:3] = g2
                self.train_datasets[i]['gesture'] = torch.from_numpy(gest_zeros.astype(np.float16))

        for i in range(self.datasets_cfg['validation'].shape[0]):
            if i ==0:
                self.val_datasets=[{'start':[], 'end':[], 'bbox':[], 'lang':[], 'gesture':[], 'actions':[]}]
            else:
                self.val_datasets.append({'start':[], 'end':[], 'bbox':[], 'lang':[], 'gesture':[], 'actions':[]})
            start_obs = cv2.imread('./img_v0.9.0e/'+self.datasets_cfg['validation'][i][10]+'.png')
            start_obs = cv2.resize(start_obs, (200,200), interpolation=cv2.INTER_AREA)
            end_obs = cv2.imread('./img_v0.9.0e/'+self.datasets_cfg['validation'][i][11]+'.png')
            end_obs = cv2.resize(end_obs, (200,200), interpolation=cv2.INTER_AREA)
            bbox = np.zeros((200,200))
            for b in self.datasets_cfg['training'][i][14]:
                bbox[b[0],0:4] += b[1:]
            self.val_datasets[i]['start'] = torch.from_numpy(start_obs.astype(np.float16))
            self.val_datasets[i]['end'] = torch.from_numpy(end_obs.astype(np.float16))
            self.val_datasets[i]['bbox'] = torch.from_numpy(np.array([bbox,bbox,bbox]).astype(np.float16))
            lang_em = self.sbert.forward([str(self.datasets_cfg['validation'][i][11])])
            self.val_datasets[i]['lang'] = lang_em
            actions_zero = np.zeros((175*16))
            actions_flat = self.datasets_cfg['validation'][i][12].flatten()
            actions_zero[:actions_flat.shape[0]] = actions_flat
            self.val_datasets[i]['actions'] = torch.from_numpy(actions_zero.astype(np.float16))
            
            if len(self.datasets_cfg['validation'][i][7]) == 0:
                self.val_datasets[i]['gesture'] = torch.from_numpy(np.zeros((200,200,3)).astype(np.float16))
            else:
                gest = np.array(self.datasets_cfg['validation'][i][7])
                zz = np.zeros((3,250,50))
                zz[0:gest.shape[0],0:gest.shape[1],0:gest.shape[2]] += gest
                gest = np.reshape(zz,(250,50,3))
                gest_zeros = np.zeros((200,200,3))
                g1 = gest[0:125,0:50, 0:3]
                g2 = gest[125:,0:50, 0:3]
                gest_zeros[0:125,0:50,0:3] = g1
                gest_zeros[0:125,50:100,0:3] = g2
                self.val_datasets[i]['gesture'] = torch.from_numpy(gest_zeros.astype(np.float16))

            #Language 
            #self.temp_train[i]['rgb_obs'] = cv2.imread(self.train_datasets[i][10], 'RGB')
        return

    def train_dataloader(self):
        #ds = NatSGDDataset(pd.DataFrame.from_dict(self.train_datasets, orient='index'), 60)
        return DataLoader({0: self.train_datasets})

        # train_dataloader ={
        #     key: DataLoader(
        #         Dataset.from_dict(dataset),
        #         batch_size=1,
        #         num_workers=0,
        #         pin_memory=False,
        #     )
        #     for key, dataset in self.train_datasets.items()
        # }
        # return CombinedLoader(train_dataloader, "max_size_cycle")
    
    def val_dataloader(self):
        return DataLoader({0: self.val_datasets})
        
        # val_dataloaders = {
        #     key: DataLoader(
        #         dataset,
        #         batch_size=1,
        #         num_workers=0,
        #         pin_memory=False,
        #         shuffle=self.shuffle_val,
        #     )
        #     for key, dataset in self.val_datasets.items()
        # }
        # combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        # return combined_val_loaders
