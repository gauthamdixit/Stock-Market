import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import matplotlib.pyplot as plt

# best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

#                     actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).to('cuda')
#                     predictions = best_tft.predict(val_dataloader)

#                     # average p50 loss overall
#                     p50_loss = (actuals - predictions).abs().mean().item()
#                     # average p50 loss per time series
#                     p50_time = (actuals - predictions).abs().mean(axis=1)
#                     key = "Hsize: " + str(h_size) + ", att_size: " + str(attention_size) + ", dropout: " + str(drop)
#                     with open('opt_data.txt', 'a') as file:
#                         file.write(f'{key}: {p50_loss}\n')
        
#                     # Take a look at what the raw_predictions variable contains

#                     raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
#                     min_loss[key] = (p50_loss, raw_predictions, best_model_path)