import matplotlib.pyplot as plt
import pandas as pd
from src.sncf_dataset import SNCFDataset
from src.load_data import load_data

def price_date():
    # Load datasets
    train_dataset, y_train, _ = load_data()
    train_dataset['occupancy_rate'] = y_train
    train_dataset.date = pd.to_datetime(train_dataset.date)
    # Group by date
    group_by_date = train_dataset.groupby(by=['date'])
    avg_daily_occupancy_rate=group_by_date.occupancy_rate.mean()
    # Plots
    fig, ax = plt.subplots()
    ax.plot(group_by_date.date.mean(), avg_daily_occupancy_rate)
    ax.set_title("Variation of occupancy rate over time")
    ax.set_ylabel('Occupancy rate')
    fig.autofmt_xdate()
    return fig

def price_weekday():
    # Load datasets
    train_dataset = pd.read_pickle('datasets/X_train')
    train_dataset['occupancy_rate'] = pd.read_pickle('datasets/y_train')
    # Group by weekday
    group_by_weekdays = train_dataset.groupby(by=['weekday'])
    occupancy_rate_by_weekday = group_by_weekdays.occupancy_rate.mean()
    # Plots
    fig, ax = plt.subplots()
    ax.bar(
        x=range(len(occupancy_rate_by_weekday)),
        height=occupancy_rate_by_weekday,
        tick_label=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        )
    ax.set_title("Variation of occupancy rate by weekdays")
    ax.set_ylabel('Occupancy rate')
    return fig

def price_hour():
    # Load datasets
    train_dataset = pd.read_pickle('datasets/X_train')
    train_dataset['occupancy_rate'] = pd.read_pickle('datasets/y_train')
    # Group by weekday
    group_by_hours = train_dataset.groupby(by=['hour'])
    occupancy_rate_by_hours = group_by_hours.occupancy_rate.mean()
    fig, ax = plt.subplots(figsize=(3.,2.3))
    ax.plot(group_by_hours.hour.mean(), occupancy_rate_by_hours)
    ax.set_title("Variation of occupancy rate by hours")
    ax.set_xlabel('Hours')
    ax.set_ylabel('Occupancy rate')
    return fig