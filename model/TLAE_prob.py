import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from model.metrics import crps_ensemble


def inv_transform(scaler: StandardScaler, Y: torch.FloatTensor):
    '''pytorch version and takes (n_features, n_samples)
    '''
    dev = Y.device
    scale = torch.from_numpy(scaler.scale_).to(dev).unsqueeze(-1)
    mean = torch.from_numpy(scaler.mean_).to(dev).unsqueeze(-1)
    Y = Y * scale + mean

    return Y

class TLAE_prob(pl.LightningModule):
    '''Probabilistic version with variational method
    '''
    def __init__(
        self,
        num_series=370,
        enc_channels=[64, 32],
        rnn_hidden=32,
        rnn_layers=3,
        dropout_X=.0,
        rg=7*24,
        reg=0.005,
        lr=0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()  #
        # enc
        num_layers = len(enc_channels)
        assert num_layers > 1
        layers = []
        for i in range(num_layers):
            in_feat = enc_channels[i-1] if i > 0 else num_series
            out_feat = enc_channels[i]
            layers.append(nn.Linear(in_feat, out_feat))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.encoder = nn.Sequential(*layers)
        # temporal model
        # input of shape (L, N=N_latent, H_in=1)
        self.Xseq = nn.GRU(
            input_size=1,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout_X
        )
        self.Xrg = rg
        self.distribution_mu = nn.Linear(rnn_hidden, 1)
        # N(0, I)
        self.noise_distrib = MultivariateNormal(torch.zeros(
            self.num_latent), torch.eye(self.num_latent))
        # decoder
        dec_channels = enc_channels[::-1]
        layers = []
        for i in range(num_layers):
            in_feat = dec_channels[i]
            out_feat = dec_channels[i+1] if i != num_layers-1 else num_series
            layers.append(nn.Linear(in_feat, out_feat))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.decoder = nn.Sequential(*layers)

    @property
    def num_latent(self):
        return self.hparams.enc_channels[-1]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (_, inpY), (_, tgtY), _, wnd_start = batch  # use normed data
        pred_steps = tgtY.shape[-1]
        assert inpY.shape[-1] == self.Xrg + (pred_steps-1)  # Yseq cond_steps
        wnd_len = self.Xrg + pred_steps  # inpY.dim_time + 1
        num_series = inpY.shape[0]
        
        Y = torch.cat((inpY, tgtY[:, [-1]]), -1)  # the whole window
        # TLAE alg
        X = self.encoder(Y.T)  # L, N_latent
        rnn_out, _ = self.Xseq(X[:-1, :, None])  # L, N_latent, H_hidden
        mu = self.distribution_mu(rnn_out).squeeze(-1)  # L, N_latent
        noise = self.noise_distrib.sample(
            (pred_steps,)).to(self.device)  # prob
        X_h = mu[-pred_steps:, ...] + noise
        X_h = torch.cat((X[:self.Xrg, ...], X_h), 0)
        Y_h = self.decoder(X_h)  # L, N_input
        Y_h = Y_h.T  # N_input, L

        l1loss = nn.L1Loss()
        recov_loss = l1loss(Y_h, Y)
        distrib = MultivariateNormal(
            mu[-pred_steps:, ...], torch.eye(self.num_latent, device=self.device))
        reg_loss = -distrib.log_prob(X[-pred_steps:, ...]).mean()
        loss = recov_loss + self.hparams.reg * reg_loss

        self.log("loss/train", loss,
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_recov", recov_loss,
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_reg", reg_loss,
                 on_step=True, on_epoch=False, batch_size=1)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr)

        return {"optimizer": optimizer}

    def forward(self, inpY, wnd_start, pred_steps, nsamples=200):
        '''TLAE alg with intermediates for loss compute
        distribution obtained from multiple sampling
        return ?
        '''
        assert inpY.ndim == 2
        assert inpY.shape[-1] == self.Xrg

        X = self.encoder(inpY.T)  # L, N_latent
        mus = []  # no teacher forcing
        rnn_out, hidden = self.Xseq(X[:, :, None])  # L, N_latent, H_hidden
        mu = self.distribution_mu(rnn_out[[-1], ...])  # 1, N_latent, 1
        mus.append(mu)
        for i in range(pred_steps-1):
            rnn_out, hidden = self.Xseq(mu, hidden)
            mu = self.distribution_mu(rnn_out[[-1], ...])  # 1, N_latent, 1
            mus.append(mu)
        mu = torch.cat(mus, 0).squeeze(-1)  # L, N_latent
        X_h = torch.tile(mu, (nsamples, 1, 1))
        noise = self.noise_distrib.sample(
            (nsamples, pred_steps)).to(self.device)  # prob
        X_h += noise  # B, L, N_latent
        Y_h = self.decoder(X_h)  # B, L, N_input
        Y_h = Y_h.swapaxes(1, 2)  # B, N_input, L

        return Y_h, mu

    def validation_step(self, batch, batch_idx):
        (_, inpY_n), (tgtY, tgtY_n), scaler, wnd_start = batch
        pred_steps = tgtY.shape[-1]
        assert inpY_n.shape[-1] == self.Xrg + (pred_steps-1)  # Yseq cond_steps
        wnd_len = self.Xrg + pred_steps  # inpY.dim_time + 1
        num_series = inpY_n.shape[0]

        Y = torch.cat((inpY_n, tgtY_n[:, [-1]]), -1)  # the whole window
        # get prediction
        Y_h1s, mu = self(Y[..., :self.Xrg], wnd_start, pred_steps)
        # for reg loss
        X = self.encoder(Y[..., self.Xrg:].T)
        distrib = MultivariateNormal(
            mu, torch.eye(self.num_latent, device=self.device))
        reg_loss = -distrib.log_prob(X).mean()
        # for metrics
        outYs = inv_transform(scaler, Y_h1s)

        crps = crps_ensemble(tgtY, outYs).mean()
        self.log("loss/val", crps,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_reg", reg_loss,
                 on_step=False, on_epoch=True, batch_size=1)

        return crps, outYs, tgtY

    def validation_epoch_end(self, outputs) -> None:
        outYss = []
        tgtYs = []
        for loss, outYs, tgtY in outputs:
            outYss.append(outYs)
            tgtYs.append(tgtY)
        outYs = torch.cat(outYss, -1)
        tgtY = torch.cat(tgtYs, -1)

        mse_fn = nn.MSELoss()
        outY = outYs.mean(0)
        mse = mse_fn(outY, tgtY)
        crps = crps_ensemble(tgtY, outYs).mean()
        crpssum = crps_ensemble(tgtY.sum(axis=0), outYs.sum(axis=1)).mean()
        self.log("metrics/mse", mse, batch_size=1)
        self.log("metrics/crps", crps, batch_size=1)
        self.log("metrics/crpssum", crpssum, batch_size=1)

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor='loss/val', mode='min', save_last=True)
        self.earlystop_callback = EarlyStopping(
            monitor='loss/val', patience=20, mode='min')

        return [self.earlystop_callback, self.checkpoint_callback]  #

    def on_train_epoch_end(self):
        if self.trainer.should_stop:
            hp_metric = self.checkpoint_callback.best_model_score
            self.log('hp_metric', hp_metric)
