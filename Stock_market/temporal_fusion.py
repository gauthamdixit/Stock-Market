import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import ta.trend
import torch
import matplotlib.pyplot as plt
import ta

def main():
    torch.cuda.empty_cache()

    data = pd.read_csv('apple_stock_data.csv', index_col=0, sep=',', decimal='.')

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    start_date = min(data.fillna(method='ffill').dropna().index)
    end_date = max(data.fillna(method='bfill').dropna().index)
    print(start_date)
    print(end_date)
    ts = data.fillna(0.)

    tmp = ts.copy()

    date = tmp.index
    earliest_time = data.index.min()

    tmp['days_from_start'] = (date - earliest_time).days
    tmp['date'] = date
    tmp['id'] = 1
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month
    tmp['Volume_Lag1'] = tmp['Volume'].shift(1)
    tmp['Volume_Lag2'] = tmp['Volume'].shift(3)

    tmp['MACD_Signal'] = ta.trend.MACD(tmp['Close']).macd_signal()
    tmp['EMA_20'] = ta.trend.EMAIndicator(tmp['Close'], window=20).ema_indicator()
    tmp['SMA_50'] = ta.trend.SMAIndicator(tmp['Close'], window=50).sma_indicator()
    tmp['ADX'] = ta.trend.ADXIndicator(tmp['High'], tmp['Low'], tmp['Close'], window=14).adx()
    tmp['CCI'] = ta.trend.CCIIndicator(tmp['High'], tmp['Low'], tmp['Close'], window=20).cci()
    tmp = tmp.fillna(0.)  # Handle NaN values from shifting
  
    tmp.drop(columns=['Healthcare','Financials','Consumer Discretionary','Consumer Staples','Utilities','Materials','Real Estate','Communication Services','Industrials','Technology', 'Energy'], inplace=True)
  
    time_df = tmp
    max_prediction_length = 180
    years = 2
    max_encoder_length = 365 * years
    training_cutoff = time_df["days_from_start"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        time_df[lambda x: x.days_from_start <= training_cutoff],
        time_idx="days_from_start",
        target="Close",
        group_ids=['id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=["days_from_start", "day", "day_of_week", "month"],
        # need to add RSI
        time_varying_unknown_reals=['Close', 'Open', 'High', 'Low', 'Volume', 'value','RSI', 'Volume_Lag1', 'Volume_Lag2', 'MACD_Signal','EMA_20', 'SMA_50', 'ADX', 'CCI'],
        target_normalizer=GroupNormalizer(transformation="softplus"),  # we normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    training_full = TimeSeriesDataSet(
        time_df,
        time_idx="days_from_start",
        target="Close",
        group_ids=['id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_reals=["days_from_start", "day", "day_of_week", "month"],
        # need to add RSI
        time_varying_unknown_reals=['Close', 'Open', 'High', 'Low', 'Volume', 'value','RSI', 'Volume_Lag1', 'Volume_Lag2', 'MACD_Signal','EMA_20', 'SMA_50', 'ADX', 'CCI'],
        target_normalizer=GroupNormalizer(transformation="softplus"),  # we normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, time_df, predict=True, stop_randomization=True)

    # create dataloaders for our model
    batch_size = 32 # Reduce batch size to use less memory
    # if you have a strong GPU, feel free to increase the number of workers  
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=4, persistent_workers=True)

    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)]).to("cuda")
    baseline_predictions = Baseline().predict(val_dataloader)
    (actuals - baseline_predictions).abs().mean().item()

    # ➢25.139617919921875

    def train_and_tune(hidden_size, attention_head_size, dropout):
        min_loss = dict()
        for h_size in hidden_size:
            for attention_size in attention_head_size:
                for drop in dropout:
                    key = "Hsize: " + str(h_size) + ", att_size: " + str(attention_size) + ", dropout: " + str(drop)
                    print(key)
                    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
                    lr_logger = LearningRateMonitor()  
                    logger = TensorBoardLogger("lightning_logs")  

                    trainer = pl.Trainer(
                        max_epochs=45,
                        accelerator='gpu', 
                        enable_model_summary=True,
                        devices=1,
                        log_every_n_steps=24,  # Set a lower logging interval
                        gradient_clip_val=0.1,
                        callbacks=[lr_logger, early_stop_callback],
                        logger=logger,
                        precision=32  # Use mixed precision training
                    )
                    tft = TemporalFusionTransformer.from_dataset(
                        training,
                        learning_rate=0.001,
                        hidden_size=h_size,
                        attention_head_size=attention_size,
                        dropout=drop,
                        hidden_continuous_size=h_size,
                        output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
                        loss=QuantileLoss(),
                        log_interval=10, 
                        reduce_on_plateau_patience=4)
                    
                    trainer.fit(
                        tft,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)
                    best_model_path = trainer.checkpoint_callback.best_model_path
                    print(best_model_path)
                    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

                    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).to('cuda')
                    predictions = best_tft.predict(val_dataloader)

                    # average p50 loss overall
                    p50_loss = (actuals - predictions).abs().mean().item()
                    # average p50 loss per time series
                    p50_time = (actuals - predictions).abs().mean(axis=1)
                    key = "Hsize: " + str(h_size) + ", att_size: " + str(attention_size) + ", dropout: " + str(drop)
                    with open('opt_data.txt', 'a') as file:
                        file.write(f'{key}: {p50_loss}\n')
        
                    # Take a look at what the raw_predictions variable contains

                    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
                    min_loss[key] = (p50_loss, raw_predictions, best_model_path)
                    
                    torch.cuda.empty_cache()
                    # plt.show()
        return min_loss

    
    # Prepare future data
    def predict_future(best_model, data, earliest_time):
        last_date = data.index[-1]
        future_dates = pd.date_range(last_date, periods=max_prediction_length + 1, freq='D')[1:]  # 1 year of daily data
        future_df = pd.DataFrame(index=future_dates)
        future_df['days_from_start'] = (future_dates - earliest_time).days
        future_df['date'] = future_dates
        future_df['id'] = 1
        future_df['day'] = future_dates.day
        future_df['day_of_week'] = future_dates.dayofweek
        future_df['month'] = future_dates.month
        

        # Merging with original dataframe to get the lagged values
        combined_df = pd.concat([tmp, future_df], sort=False)
        combined_df['Volume_Lag1'] = combined_df['Volume'].shift(1)
        combined_df['Volume_Lag2'] = combined_df['Volume'].shift(3)
        combined_df['MACD_Signal'] = ta.trend.MACD(combined_df['Close']).macd_signal()
        combined_df['EMA_20'] = ta.trend.EMAIndicator(combined_df['Close'], window=20).ema_indicator()
        combined_df['SMA_50'] = ta.trend.SMAIndicator(combined_df['Close'], window=50).sma_indicator()
        combined_df['ADX'] = ta.trend.ADXIndicator(combined_df['High'], combined_df['Low'], combined_df['Close'], window=14).adx()
        combined_df['CCI'] = ta.trend.CCIIndicator(combined_df['High'], combined_df['Low'], combined_df['Close'], window=20).cci()
        combined_df = combined_df[combined_df.index.isin(future_dates)].fillna(0)

        # Ensure future_df has enough entries
        future_df = future_df[future_df['days_from_start'] > training_cutoff]

        full_data = pd.concat([time_df, future_df], sort=False)

        # Create a TimeSeriesDataSet for the future data
        future_dataset = TimeSeriesDataSet.from_dataset(training_full, full_data, predict=True, stop_randomization=True)
        future_dataloader = future_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

        # Make future predictions
        future_predictions = best_model.predict(future_dataloader)
        future_predictions = future_predictions.squeeze().cpu().numpy()
        # print(future_predictions)

        # Combine known data and future predictions for plotting
        known_data = tmp['Close']
        future_data = future_df[['date']].copy()
        future_data['Close'] = future_predictions
        future_data = future_data.iloc[:len(future_predictions)]

        #combined_data = pd.concat([known_data, future_data.set_index('date')['Close']])

        # Plot known data and future predictions
        plt.figure(figsize=(15, 6))
        plt.plot(known_data, label='Known Data')
        plt.plot(future_data.set_index('date'), label='Future Predictions', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Apple Stock Price Prediction')
        plt.legend()
        plt.show()

    train = False
    continue_training = False
    predict = True

    hidden_sizes = [350]
    attention_head_sizes = [3]
    dropouts = [0.3]

    if train:
        results = train_and_tune(hidden_sizes, attention_head_sizes, dropouts)
        min_key = min(results, key=lambda k: results[k][0])
        print("best model is: ", min_key)
        predictions = results[min_key][1]
        fig, ax = plt.subplots(figsize=(10, 4))
        best_tft = TemporalFusionTransformer.load_from_checkpoint(results[min_key][2])
        best_tft.plot_prediction(predictions.x, predictions.output, idx=0, add_loss_to_title=QuantileLoss(), ax=ax)
        plt.show()
    # model_path = "D:\Stock_market\lightning_logs\lightning_logs\\version_0\checkpoints\epoch=10-step=748.ckpt"
    # best_model = TemporalFusionTransformer.load_from_checkpoint(model_path)

   # Fine-tune the model on the validation dataset for a limited number of epochs
    
    
    if continue_training: 
        print("continuing training on full dataset!\n")   
        combined_dataloader = training_full.to_dataloader(train=True, batch_size=batch_size, num_workers=4, persistent_workers=True)

        trainer = pl.Trainer(
            max_epochs=1,  # You can set the desired number of epochs
            accelerator='gpu',
            devices=1,
            precision=32,  # Use 32-bit precision
            callbacks=[EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=1, verbose=True, mode="min"), LearningRateMonitor()],
            logger=TensorBoardLogger("lightning_logs", name="continued_training")
        )
        model_path = "D:\Stock_market\lightning_logs\lightning_logs\\version_3\checkpoints\epoch=13-step=952.ckpt"
        best_model = TemporalFusionTransformer.load_from_checkpoint(model_path)

        trainer.fit(best_model, combined_dataloader)
        
    
    if predict:
        best_model = TemporalFusionTransformer.load_from_checkpoint('D:\Stock_market\lightning_logs\lightning_logs\\version_3\checkpoints\epoch=13-step=952.ckpt')
        predict_future(best_model, tmp, earliest_time)

if __name__ == '__main__':
    main()
