import argparse
import sys
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('./')

from model.TLAE import TLAE
from model.TLAE_prob import TLAE_prob
from data.TS import TSDataModule


def main_TLAE(*args):
    rg = 24*7
    dm = TSDataModule("traffic", rg, 24, val_wnd=7, hbsize=2*rg, hbstep=24,
                      mode="M", bs=1024)
    
    enc_channels = [96, 64, 32, 16]
    rnn_hidden = 32 * 2
    rnn_layers = 4
    dropout_X = 0
    model = TLAE(num_series=963,
                 enc_channels=enc_channels,
                 rnn_hidden=rnn_hidden,
                 rnn_layers=rnn_layers,
                 dropout_X=dropout_X,
                 rg=rg,
                 )
    trainer = pl.Trainer(gpus=1, log_every_n_steps=37,  # fast_dev_run=5,
                         )  #max_epochs=20
    trainer.fit(model, datamodule=dm)


warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")

if __name__ == "__main__":
    pl.seed_everything(2021)
    main_TLAE()