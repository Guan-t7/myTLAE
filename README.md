# My impl. of Temporal Latent Auto-Encoder (TLAE)

TLAE decomposes high dimensional multivariate time series into a few basis series using MLPs, producing global predictions.

## ./model

- This is not an official implementation.
- `TLAE*.py` contain the two models as described in [Temporal Latent Auto-Encoder: A Method for Probabilistic Multivariate Time Series Forecasting](https://arxiv.org/abs/2101.10460). 
- Probabilistic version is not fully evaluated. It seems that the prediction should be somehow normalized before calculating CRPS.

## ./data

Get `electricity`, `traffic` dataset from [DeepGLO](https://github.com/rajatsen91/deepglo/blob/master/datasets/download-data.sh). For example, `./data/electricity.npy` contains an ndarray of the shape (370, 25000+). Construct the subset `electricity-small` from the period between 2014-01-01 and 2014-09-01.

## Results reproduced

| WAPE/MAPE/SMAPE | electricity       | traffic           |
|-----------------|-------------------|-------------------|
| TCN-MF          | 0.106/0.525/0.188 | 0.226/0.284/0.247 |
| TLAE reported   | 0.080/0.152/0.120 | 0.117/0.137/0.108 |
| TLAE reproduced | 0.090/0.200/0.128 | 0.135/0.156/0.127 |

| CRPS / MSE      | electricity-small | traffic        |
|-----------------|-------------------|----------------|
| GP-Copula       | 0.056 / 2.4e5     | 0.133 / 6.9e-4 |
| TLAE reported   | 0.058 / 2.0e5     | 0.097 / 4.4e-4 |
| TLAE reproduced | 0.069 / 3.7e6     | 0.121 / 4.5e-4 |

In short, metrics reproduced is better than previous SOTA (such as `TCN-MF` in `DeepGLO`), but not so good as reported in TLAE paper.

- RNNs are slow to train. Serious applications should replace GRU within the TLAE with TCN.
- The above metrics are calculated using rescaled (unnormalized) data. Final metrics vary using different data scaling schemes.
- Also, you might want to sequentially slide the training windows rather than randomly sampling.

Tested on a GeForce RTX 2080 Ti, using these packeges:

```
Python 3.9.9
torch 1.10.0
pytorch-lightning 1.5.10
```
