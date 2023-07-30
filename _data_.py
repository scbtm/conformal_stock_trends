from dataclasses import dataclass
from pathlib import Path

import polars as pl #type: ignore
from datetime import datetime
from datetime import timedelta

from typing import Tuple
import numpy as np #type: ignore

@dataclass
class DataConfig:
    data_path: str = Path('data')
    dev_data_file: str = data_path / 'stocks_history.csv'
    train_max_date = datetime(2018,6,1)
    calibration_max_date = datetime(2019, 1, 1)
    validation_max_date = datetime(2019, 6, 1)

class DataETL:
    def __init__(self, config:DataConfig) -> None:
        self.config = config

class DataPreparationPipeline:
    def __init__(self, config:DataConfig) -> None:
        self.config = config

    @classmethod
    def get_label(self, df:pl.DataFrame) -> float:
        """Get the average of the Close price
        This is our target value, and it will span in a short period of time, like a week or so. That is determined by 
        the (length of the) dataframe passed to the function
        """
        return df['Close'].mean()
    
    @classmethod
    def week_day(self, date:datetime) -> bool:
        """Monday is 0
        """
        weekdays = {0:'monday',
                    1:'tuesday',
                    2:'wednesday',
                    3:'thursday',
                    4:'friday',
                    5:'saturday',
                    6:'sunday'}
        
        return weekdays[date.weekday()]
    
    @classmethod
    def rescale(self, df:pl.DataFrame, col = 'Close') -> Tuple[pl.DataFrame, float]:
        """Rescale time series values with respect to the first day in the series.
        What this does is that it will express any future value with respect to the first day in the series 
        as a relative number to the first value in the series. This is useful because it will allow us to compare
        different stocks at different prices and different points in time.

        This will typically be used to rescale the Close price, but it can be used for any other value in the series,
        and the dataframe will span a short period of time, like a week or so.
        """
        base = df.filter(pl.col('Date') == df['Date'].min())[col].to_list()[0]
        return df.with_columns( (pl.col(col) / base).alias(col) ), base

    @classmethod
    def get_info_chunk(self, pivot_date:datetime, 
                       df:pl.DataFrame, 
                       time_window:int = 8) -> pl.DataFrame:
        """Given a pivot_date, return data before that day up to time_window days before
        """
        #Get the first day of the time window
        first_day = pivot_date - timedelta(days = time_window)

        #Get all the real (not necessarily in the data) dates in the time window
        dates = [(first_day + timedelta(days=x)).date() for x in range((pivot_date-first_day).days + 1)]
        date_range = pl.DataFrame({'Date': dates})

        #Get whatever data available in the date range
        info = df.filter((pl.col("Date") <= pivot_date) & (pl.col("Date") >= first_day))
        chunk = date_range.join(info, on='Date', how='left')

        #Fill in missing data with linear interpolation (Polars default method)
        chunk = chunk.interpolate()

        return chunk
    
    @classmethod
    def get_data_label_pair(self,
                            df:pl.DataFrame, 
                            pivot_date:datetime, 
                            future:int, 
                            past:int) -> Tuple[pl.DataFrame, float]:
        
        """
        Args: 
            df: dataframe with the stock data
            pivot_date: the date we want to split each chunk of data on
            future: the number of days after the predictor pivot 
                (we use this offset because we need another pivot in the future). Therefore, we need the future data
                and the past data not to overlap, hence we use a strict rule that the future pivot must be at least
                15 days after the predictor pivot, since at that new pivot we will take 7 days of 'past' data.
                Essentially, we are ensuring that we are predicting the average target of an entire week that starts at least
                one week after the predictor pivot.

            past: the number of days before the predictor pivot

        Returns:
            past_data: the data before the predictor pivot
            label: the average of the Close price in the future pivot

        """
        
        assert future == 7, "One week offset is required in the future label"
        past_data = DataPreparationPipeline.get_info_chunk(pivot_date = pivot_date, 
                                                df = df, 
                                                time_window = past)
        
        future_data = DataPreparationPipeline.get_info_chunk(pivot_date = pivot_date + timedelta(days = future),
                                                  df = df,
                                                  time_window = 7)
        
        label = DataPreparationPipeline.get_label(future_data)
        return past_data, label
    
    @classmethod
    def transform_into_features(self, df:pl.DataFrame, label, n_split:int = 3) -> pl.DataFrame:
        """Transform the dataframe into a dataframe of features that we can use with ML. 

        Notice we receive two chunks of raw data. One is the past data to extract predictive features from, 
        and the other is the raw form of the target (1-week-average closing price of the week following the last information 
        from the predictor data).
        
        Args:
            df: dataframe with the stock data past data to be used for prediction (to be transformed into features)
            label: the average of the Close price in the future pivot
            n_split: the number of splits to use for the features. This is useful to get a summarized version of the
                features, since we will be splitting the data into n_split chunks and getting the average of each chunk.

        """

        #Get the relative spread and relative movement
        df = df.with_columns( ((pl.col('High') - pl.col('Low'))/(0.000001 + pl.col('Low'))).alias('relative_spread') )
        df = df.with_columns( ((pl.col('Close') - pl.col('Open'))/(0.000001 + pl.col('Open'))).alias('relative_movement') )
        df = df[['Date', 'relative_spread', 'relative_movement', 'Close', 'Volume']]
        df = df.with_columns([pl.col('relative_spread')
                                .fill_null(strategy = 'backward')
                                .fill_null(strategy = 'forward')
                                .alias('relative_spread'),
                              pl.col('relative_movement')
                                .fill_null(strategy = 'backward')
                                .fill_null(strategy = 'forward')
                                .alias('relative_movement'),
                              pl.col('Volume')
                                .fill_null(strategy = 'backward')
                                .fill_null(strategy = 'forward')
                                .alias('Volume'),
                              pl.col('Close')
                                .fill_null(strategy = 'backward')
                                .fill_null(strategy = 'forward')
                                .alias('Close')])

        #Rescale the Volume and Close price
        df, vol_base = DataPreparationPipeline.rescale(df, col = 'Volume')
        #This is to prevent division by zero in the future
        vol_base = vol_base+1

        df, price_base = DataPreparationPipeline.rescale(df, col = 'Close')
        df = df.with_columns([pl.col("relative_spread")
                                .diff(n=1)
                                .fill_null(strategy = 'forward')
                                .fill_null(strategy = 'backward')
                                .alias("relative_spread_diff"),
                              pl.col("relative_movement")
                                .diff(n=1)
                                .fill_null(strategy = 'forward')
                                .fill_null(strategy = 'backward')
                                .alias("relative_movement_diff"),
                              pl.col("Volume")
                                .diff(n=1)
                                .fill_null(strategy = 'forward')
                                .fill_null(strategy = 'backward')
                                .alias("volume_diff"),
                            pl.col("Close")
                                .diff(n=1)
                                .fill_null(strategy = 'forward')
                                .fill_null(strategy = 'backward')
                                .alias("price_diff")
                              ])

        price_values = df['Close'].to_list()
        volume_values = df['Volume'].to_list()
        relative_spread_values = df['relative_spread'].to_list()
        relative_movement_values = df['relative_movement'].to_list()

        price_diff_values = df['price_diff'].to_list()
        volume_diff_values = df['volume_diff'].to_list()
        relative_spread_diff_values = df['relative_spread_diff'].to_list()
        relative_movement_diff_values = df['relative_movement_diff'].to_list()

        #Get features for the dataset
        max_price = max(price_values)
        min_price = min(price_values)
        first_price = price_values[0]
        last_price = price_values[-1]
        delta_price = last_price - first_price
        spread_price = max_price - min_price
        standard_deviation_price = np.std(price_values)
        mean_price = np.mean(price_values)

        max_volume = max(volume_values)
        min_volume = min(volume_values)
        first_volume = volume_values[0]
        last_volume = volume_values[-1]
        delta_volume = last_volume - first_volume
        spread_volume = max_volume - min_volume
        standard_deviation_volume = np.std(volume_values)
        mean_volume = np.mean(volume_values)

        max_relative_spread = max(relative_spread_values)
        min_relative_spread = min(relative_spread_values)
        first_relative_spread = relative_spread_values[0]
        last_relative_spread = relative_spread_values[-1]
        delta_relative_spread = last_relative_spread - first_relative_spread
        spread_relative_spread = max_relative_spread - min_relative_spread
        standard_deviation_relative_spread = np.std(relative_spread_values)
        mean_relative_spread = np.mean(relative_spread_values)

        max_relative_movement = max(relative_movement_values)
        min_relative_movement = min(relative_movement_values)
        first_relative_movement = relative_movement_values[0]
        last_relative_movement = relative_movement_values[-1]
        delta_relative_movement = last_relative_movement - first_relative_movement
        spread_relative_movement = max_relative_movement - min_relative_movement
        standard_deviation_relative_movement = np.std(relative_movement_values)
        mean_relative_movement = np.mean(relative_movement_values)

        max_price_diff = max(price_diff_values)
        min_price_diff = min(price_diff_values)
        first_price_diff = price_diff_values[0]
        last_price_diff = price_diff_values[-1]
        delta_price_diff = last_price_diff - first_price_diff
        spread_price_diff = max_price_diff - min_price_diff
        standard_deviation_price_diff = np.std(price_diff_values)
        mean_price_diff = np.mean(price_diff_values)

        max_volume_diff = max(volume_diff_values)
        min_volume_diff = min(volume_diff_values)
        first_volume_diff = volume_diff_values[0]
        last_volume_diff = volume_diff_values[-1]
        delta_volume_diff = last_volume_diff - first_volume_diff
        spread_volume_diff = max_volume_diff - min_volume_diff
        standard_deviation_volume_diff = np.std(volume_diff_values)
        mean_volume_diff = np.mean(volume_diff_values)

        max_relative_spread_diff = max(relative_spread_diff_values)
        min_relative_spread_diff = min(relative_spread_diff_values)
        first_relative_spread_diff = relative_spread_diff_values[0]
        last_relative_spread_diff = relative_spread_diff_values[-1]
        delta_relative_spread_diff = last_relative_spread_diff - first_relative_spread_diff
        spread_relative_spread_diff = max_relative_spread_diff - min_relative_spread_diff
        standard_deviation_relative_spread_diff = np.std(relative_spread_diff_values)
        mean_relative_spread_diff = np.mean(relative_spread_diff_values)

        max_relative_movement_diff = max(relative_movement_diff_values)
        min_relative_movement_diff = min(relative_movement_diff_values)
        first_relative_movement_diff = relative_movement_diff_values[0]
        last_relative_movement_diff = relative_movement_diff_values[-1]
        delta_relative_movement_diff = last_relative_movement_diff - first_relative_movement_diff
        spread_relative_movement_diff = max_relative_movement_diff - min_relative_movement_diff
        standard_deviation_relative_movement_diff = np.std(relative_movement_diff_values)
        mean_relative_movement_diff = np.mean(relative_movement_diff_values)

        #Create the features dataframe
        df = pl.DataFrame({'max_price':max_price,
                            'min_price':min_price,
                           # 'first_price':first_price, this is always the same
                            'last_price':last_price,
                            'delta_price':delta_price,
                            'spread_price':spread_price,
                            'standard_deviation_price':standard_deviation_price,
                            'mean_price':mean_price,
                            'max_volume':max_volume,
                            'min_volume':min_volume,
                            'first_volume':first_volume,
                            'last_volume':last_volume,
                            'delta_volume':delta_volume,
                            'spread_volume':spread_volume,
                            'standard_deviation_volume':standard_deviation_volume,
                            'mean_volume':mean_volume,
                            'max_relative_spread':max_relative_spread,
                            'min_relative_spread':min_relative_spread,
                            'first_relative_spread':first_relative_spread,
                            'last_relative_spread':last_relative_spread,
                            'delta_relative_spread':delta_relative_spread,
                            'spread_relative_spread':spread_relative_spread,
                            'standard_deviation_relative_spread':standard_deviation_relative_spread,
                            'mean_relative_spread':mean_relative_spread,
                            'max_relative_movement':max_relative_movement,
                            'min_relative_movement':min_relative_movement,
                            'first_relative_movement':first_relative_movement,
                            'last_relative_movement':last_relative_movement,
                            'delta_relative_movement':delta_relative_movement,
                            'spread_relative_movement':spread_relative_movement,
                            'standard_deviation_relative_movement':standard_deviation_relative_movement,
                            'mean_relative_movement':mean_relative_movement,
                            'max_price_diff':max_price_diff,
                            'min_price_diff':min_price_diff,
                            'first_price_diff':first_price_diff,
                            'last_price_diff':last_price_diff,
                            'delta_price_diff':delta_price_diff,
                            'spread_price_diff':spread_price_diff,
                            'standard_deviation_price_diff':standard_deviation_price_diff,
                            'mean_price_diff':mean_price_diff,
                            'max_volume_diff':max_volume_diff,
                            'min_volume_diff':min_volume_diff,
                            'first_volume_diff':first_volume_diff,
                            'last_volume_diff':last_volume_diff,
                            'delta_volume_diff':delta_volume_diff,
                            'spread_volume_diff':spread_volume_diff,
                            'standard_deviation_volume_diff':standard_deviation_volume_diff,
                            'mean_volume_diff':mean_volume_diff,
                            'max_relative_spread_diff':max_relative_spread_diff,
                            'min_relative_spread_diff':min_relative_spread_diff,
                            'first_relative_spread_diff':first_relative_spread_diff,
                            'last_relative_spread_diff':last_relative_spread_diff,
                            'delta_relative_spread_diff':delta_relative_spread_diff,
                            'spread_relative_spread_diff':spread_relative_spread_diff,
                            'standard_deviation_relative_spread_diff':standard_deviation_relative_spread_diff,
                            'mean_relative_spread_diff':mean_relative_spread_diff,
                            'max_relative_movement_diff':max_relative_movement_diff,
                            'min_relative_movement_diff':min_relative_movement_diff,
                            'first_relative_movement_diff':first_relative_movement_diff,
                            'last_relative_movement_diff':last_relative_movement_diff,
                            'delta_relative_movement_diff':delta_relative_movement_diff,
                            'spread_relative_movement_diff':spread_relative_movement_diff,
                            'standard_deviation_relative_movement_diff':standard_deviation_relative_movement_diff,
                            'mean_relative_movement_diff':mean_relative_movement_diff,
                            'volume_base': vol_base,
                            })

        #the label is going to be relative to the maximum price in the observed week
        abs_max_price = max_price*price_base
        label = np.float32(label/abs_max_price)
        df = df.with_columns(pl.lit(label).alias('label'))

        return df

    def load_data(self) -> pl.DataFrame:
        df = pl.read_csv(self.config.dev_data_file, try_parse_dates=False)
        #df = df[['Date', 'Close', 'Volume', 'ticker']]
        df = df.with_columns(pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d"))

        return df
    
    def split_data(self, df:pl.DataFrame) -> pl.DataFrame:
        train_df = df.filter(pl.col("Date") <= self.config.train_max_date)
        calibration_df = df.filter((pl.col("Date") > self.config.train_max_date) & (pl.col("Date") <= self.config.calibration_max_date))
        validation_df = df.filter((pl.col("Date") > self.config.calibration_max_date) & (pl.col("Date") <= self.config.validation_max_date))
        test_df = df.filter(pl.col("Date") > self.config.validation_max_date)

        return train_df, calibration_df, validation_df, test_df
    



class DataPreparationFlow():
    def __init__(self, pipeline:DataPreparationPipeline):
        self.pipeline = pipeline
    
    def get_sample_points(self, 
                          tickers:list, 
                          pivot_dates:list, 
                          n_sample:int = 200, 
                          df:pl.DataFrame = None,
                          future:int = 30,
                          past:int = 30):
        import time
        import random
        start = time.time()
        chunks = []
        sample_points = dict()
        for i, ticker in enumerate(tickers):
            sample_points[ticker] = []
            random.seed(i)
            for pivot in random.sample(pivot_dates, n_sample):
                #We need a datetime object, not a datetime.date object
                pivot = datetime(pivot.year, pivot.month, pivot.day)
                try:
                    dfs = df.filter(pl.col("ticker") == ticker)
                    data, label = DataPreparationPipeline.get_data_label_pair(df = dfs,
                                                                              pivot_date = pivot,
                                                                              future = future,
                                                                              past = past)
                    
                    x = DataPreparationPipeline.transform_into_features(data, label)
                    x = x.with_columns([pl.lit(ticker).alias('ticker'), 
                                        pl.lit(pivot).alias('pivot_date')])
                    chunks.append(x)
                except:
                    pass

                sample_points[ticker].append(pivot)

            if i % 10 == 0:
                print(f"{i} tickers processed, at {time.time() - start} seconds. {len(chunks)} chunks created")

        end = time.time() - start
        print(f"Elapsed time: {end} seconds")

        final_data = pl.concat(chunks, how = 'vertical')
        sample_points = pl.DataFrame(sample_points)
        return final_data, sample_points
    

    def prep_dev_data(self, 
                      n_train_tickers:int = 4500, 
                      n_cal_tickers:int = 500,
                      n_val_tickers:int = 500, 
                      n_test_tickers:int = 500,
                      full_data = False,
                      n_train_sample:int = 250,
                      n_cal_sample:int = 100,
                      n_val_sample:int = 100,
                      n_test_sample:int = 100,
                      save_output:bool = False,
                      output_format:str = 'csv',
                      future:int = 7,
                      past:int = 8):
        import random, json

        df = self.pipeline.load_data()
        max_date = df['Date'].max()
        min_date = df['Date'].min()
        tickers = df['ticker'].unique().to_list()
        train, cal, val, test = self.pipeline.split_data(df)

        if full_data:
            train_tickers = tickers
            cal_tickers = tickers
            val_tickers = tickers
            test_tickers = tickers
        else:
            random.seed(1)
            train_tickers = random.sample(tickers, n_train_tickers)
            random.seed(2)
            cal_tickers = random.sample(tickers, n_cal_tickers)
            random.seed(3)
            val_tickers = random.sample(tickers, n_val_tickers)
            random.seed(4)
            test_tickers = random.sample(tickers, n_test_tickers)

        train_pivot_dates = list(set(train['Date'].to_list()))
        cal_pivot_dates = list(set(cal['Date'].to_list()))
        val_pivot_dates = list(set(val['Date'].to_list()))
        test_pivot_dates = list(set(test['Date'].to_list()))

        train_data, train_sample_points = self.get_sample_points(tickers = train_tickers,
                                                                 pivot_dates = train_pivot_dates,
                                                                 n_sample = n_train_sample,
                                                                 df = train,
                                                                 future = future,
                                                                 past = past)
        cal_data, cal_sample_points = self.get_sample_points(tickers = cal_tickers,
                                                             pivot_dates = cal_pivot_dates,
                                                             n_sample = n_cal_sample,
                                                             df = cal,
                                                             future = future,
                                                             past = past)
        val_data, val_sample_points = self.get_sample_points(tickers = val_tickers,
                                                             pivot_dates = val_pivot_dates,
                                                             n_sample = n_val_sample,
                                                             df = val,
                                                             future = future,
                                                             past = past)
        test_data, test_sample_points = self.get_sample_points(tickers = test_tickers,
                                                               pivot_dates = test_pivot_dates,
                                                               n_sample = n_test_sample,
                                                               df = test,
                                                               future = future,
                                                               past = past)

        if save_output:
            if output_format == 'parquet':
                train_data.write_parquet(self.pipeline.config.data_path/'train_data.parquet')
                cal_data.write_parquet(self.pipeline.config.data_path/'cal_data.parquet')
                val_data.write_parquet(self.pipeline.config.data_path/'val_data.parquet')
                test_data.write_parquet(self.pipeline.config.data_path/'test_data.parquet')

                train_sample_points.write_parquet(self.pipeline.config.data_path/'train_sample_points.parquet')
                cal_sample_points.write_parquet(self.pipeline.config.data_path/'cal_sample_points.parquet')
                val_sample_points.write_parquet(self.pipeline.config.data_path/'val_sample_points.parquet')
                test_sample_points.write_parquet(self.pipeline.config.data_path/'test_sample_points.parquet')
                
            elif output_format == 'csv':
                train_data.write_csv(self.pipeline.config.data_path/'train_data.csv')
                cal_data.write_csv(self.pipeline.config.data_path/'cal_data.csv')
                val_data.write_csv(self.pipeline.config.data_path/'val_data.csv')
                test_data.write_csv(self.pipeline.config.data_path/'test_data.csv')

                train_sample_points.write_csv(self.pipeline.config.data_path/'train_sample_points.csv')
                cal_sample_points.write_csv(self.pipeline.config.data_path/'cal_sample_points.csv')
                val_sample_points.write_csv(self.pipeline.config.data_path/'val_sample_points.csv')
                test_sample_points.write_csv(self.pipeline.config.data_path/'test_sample_points.csv')

            #If json metadata file exsists, read it:
            if (self.pipeline.config.data_path/'metadata.json').exists():
                with open(self.pipeline.config.data_path/'metadata.json', 'r') as f:
                    metadata = json.load(f)

                run_iter_num = metadata['run_iteration_number'] + 1
            else:
                run_iter_num = 1

            #Write json with metadata about the run:
            metadata = dict({'train_tickers':train_tickers,
                             'cal_tickers':cal_tickers,
                             'val_tickers':val_tickers,
                             'test_tickers':test_tickers,
                             'train_sample':n_train_sample,
                             'cal_sample':n_cal_sample,
                             'val_sample':n_val_sample,
                             'test_sample':n_test_sample,
                             'future':future,
                             'past':past,
                             'date_of_run': datetime.now().strftime("%d/%m/%Y"),
                             'output_format': output_format,
                             'all_possible_tickers': tickers,
                             'date_range_of_dev_file': [min_date, max_date],
                             'run_iteration_number': run_iter_num})
            
            with open(self.pipeline.config.data_path/'metadata.json', 'w') as f:
                json.dump(metadata, f, sort_keys=True, default=str)
            

            

        train_info = dict({'data':train_data, 'sample_points':train_sample_points})
        cal_info = dict({'data':cal_data, 'sample_points':cal_sample_points})
        val_info = dict({'data':val_data, 'sample_points':val_sample_points})
        test_info = dict({'data':test_data, 'sample_points':test_sample_points})

        return train_info, cal_info, val_info, test_info