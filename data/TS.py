import math
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl


class TSDataset(Dataset):
    '''Time series init-ed from an array of shape (n, t)
    one sample: a window that forecast `pred_steps`, given previous `cond_steps`
    '''
    def __init__(self, 
                 array: Union[np.array, torch.Tensor], scaler: Optional[StandardScaler] = None,
                 cond_steps=168, pred_steps=24, hbstep=24,
                 mode: Literal["M", "S"] = "M", batch_size: Optional[int] = None,
                 start_date="2016-1-1", freq="H"):
        '''
        @array: raw time series
        @hbstep: stride when rolling sample wnd along time dim
        @mode: Multivariate setting ("M") or Univariate setting ("S")
        @batch_size: used only in mode "S" for now
        @start_date: timestamp of the beginning of the dataset
        '''
        super().__init__()
        assert array.ndim == 2
        assert array.shape[0] < 10000  # TODO vertical batching
        assert array.shape[-1] >= cond_steps+pred_steps
        self.cond_steps = cond_steps
        self.pred_steps = pred_steps
        self.hbstep = hbstep
        self.mode = mode
        self.batch_size = batch_size
        self.start_date = pd.Timestamp(start_date)
        self.freq = freq
        # rolling windows as if hbstep==1
        wnd_sz = cond_steps+pred_steps
        # -> (n_series, t_start, wnd_sz)
        self.windows = torch.tensor(array).unfold(-1, wnd_sz, 1)
        self.len = self.windows.size(-2)
        if scaler is not None:
            if scaler.n_features_in_ == 1:
                norm_flat = scaler.transform(array.reshape(-1, 1))
                norm_array = norm_flat.reshape(array.shape)
            else:
                norm_array = scaler.transform(array.T).T
            self.norm_windows = torch.tensor(norm_array).unfold(-1, wnd_sz, 1)
        self.scaler = scaler

    def __len__(self):
        n_ts = self.windows.size(0)
        if self.mode == "S":  # ref. https://github.com/Nixtla/neuralforecast
            return n_ts
        return math.ceil(self.len / self.hbstep)

    # todo Use slices, int or list for getitem
    # todo batching (b, n, t) also needs rev model code
    def __getitem__(self, index):
        '''Return: 
        mode "M": tensors of shape (n, t), (n, t); ...
        mode "S": tensors of shape (b, t), (b, t); ... 
        '''
        if self.mode == "S":  # index == ts_idx
            assert self.hbstep == 1  # todo
            assert isinstance(index, int)
            # 1 epoch: for each ts,
            windows = self.windows[index]
            # sample `bs` windows
            n_windows = np.prod(windows.shape[:-1])
            wnd_idxs = np.random.choice(n_windows, size=self.batch_size,
                                        replace=(n_windows < self.batch_size))
            samples = windows[wnd_idxs]
            samples_n = self.norm_windows[index][wnd_idxs]
        else:
            assert self.mode == "M"  # `index` be along time dim
            assert isinstance(index, int)
            # translate hbstep
            index = torch.clamp(torch.tensor(index) * self.hbstep, max=self.len-1)
            windows = self.windows[:, index]
            # (n_series, wnd_sz)
            samples = windows
            samples_n = self.norm_windows[:, index]

        inp = samples[..., :-1]  # teacher forcing
        out = samples[..., -self.pred_steps:]
        if self.scaler is not None:
            inp_n = samples_n[..., :-1]
            out_n = samples_n[..., -self.pred_steps:]
        else:
            inp_n = None; out_n = None

        wnd_start = NotImplemented  #self.start_date + pd.Timedelta(wnd_start, unit=self.freq)
        if self.mode == "M":
            return (inp, inp_n), (out, out_n), self.scaler, wnd_start,
        else:
            return (inp, inp_n), (out, out_n), -1, -1,  # todo collate_fn


class TSDataModule(pl.LightningDataModule):
    def __init__(self, name: str,
                 cond_steps=168, pred_steps=24, val_wnd=7,
                 hbsize=None, hbstep=None,
                 mode='M', bs=None):
        super().__init__()
        self.name = name
        self.data_path = Path.cwd() / 'data' / f'{name}.npy'
        self.cond_steps = cond_steps
        self.pred_steps = pred_steps
        self.val_wnd = val_wnd  # todo deprecate val_wnd
        self.hbsize = cond_steps+pred_steps if hbsize is None else hbsize
        self.hbstep = pred_steps if hbstep is None else hbstep
        self.mode = mode
        self.bs = bs

    def prepare_data(self) -> None:
        assert self.data_path.exists()
    
    def setup(self, stage: Optional[str] = None) -> None:
        array = np.load(self.data_path).astype(np.float32)
        if self.name == 'electricity':
            self.start_date = "2012-1-1"
            # ? in TLAE Table 3, 25920 + 24*7 != array.shape[-1]
            train_end = 25920
            # test as val (skipping hparam select in TLAE 6.2
            val_end = train_end + self.val_wnd * self.pred_steps
            array = array[..., :val_end]
        elif self.name == 'electricity-small':
            self.start_date = "2014-1-1"
            train_end = 5832
        elif self.name == 'traffic':
            self.start_date = "2008-1-1"
            train_end = 10392
        else:
            raise ValueError()
        # split
        self.train_data = array[..., :train_end]
        val_start = train_end - self.cond_steps
        self.val_data = array[..., val_start:]
        self.val_start = val_start
        # z-score w. param evaled on the whole training set
        try:
            self.scaler
        except AttributeError:
            assert self.train_data.ndim == 2
            if self.name == "traffic":
                # across entire training set
                self.scaler = StandardScaler().fit(self.train_data.reshape(-1, 1))
            else:
                # per series
                self.scaler = StandardScaler().fit(self.train_data.T)

    def train_dataloader(self):
        pred_steps = self.hbsize - self.cond_steps
        assert pred_steps > 0
        train_ds = TSDataset(self.train_data, self.scaler, 
                             self.cond_steps, pred_steps, hbstep=self.hbstep, 
                             mode=self.mode, batch_size=self.bs, 
                             start_date=self.start_date)
        if self.mode == "S":
            sampler = SequentialSampler(train_ds)
        elif self.mode == "M":
            if self.bs is None:
                sampler = RandomSampler(train_ds)
            else:
                sampler = RandomSampler(train_ds, replacement=True, num_samples=self.bs)
        return DataLoader(train_ds, batch_size=None, sampler=sampler, num_workers=0, pin_memory=True)  #2

    def val_dataloader(self):
        start_date = pd.Timestamp(self.start_date) + pd.Timedelta(self.val_start, unit="H")
        val_ds = TSDataset(self.val_data, self.scaler, 
                           self.cond_steps, self.pred_steps, hbstep=self.pred_steps,
                           mode="M", batch_size=None,
                           start_date=start_date)
        sampler = SequentialSampler(val_ds)
        return DataLoader(val_ds, batch_size=None, sampler=sampler, num_workers=0, pin_memory=True)