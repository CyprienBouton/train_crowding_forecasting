import matplotlib.pyplot as plt
import pandas as pd
from src.sncf_dataset import SNCFDataset
from src.load_data import load_data


def price_date():
    """Shows the average occupancy rates by days
    """
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
    """Shows the average occupancy rates by weekdays
    """
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
    ax.set_title("Average occupancy rate by weekdays")
    ax.set_ylabel('Occupancy rate')
    return fig


def price_hour():
    """ Shows the average occupancy rates by hours
    """
    # Load datasets
    train_dataset = pd.read_pickle('datasets/X_train')
    train_dataset['occupancy_rate'] = pd.read_pickle('datasets/y_train')
    # Group by hours
    group_by_hours = train_dataset.groupby(by=['hour'])
    occupancy_rate_by_hours = group_by_hours.occupancy_rate.mean()
    fig, ax = plt.subplots(figsize=(3.,2.3))
    ax.plot(group_by_hours.hour.mean(), occupancy_rate_by_hours)
    ax.set_title("Variation of occupancy rate by hours")
    ax.set_xlabel('Hours')
    ax.set_ylabel('Occupancy rate')
    return fig


def price_composition():
    """ Shows the average occupancy rates in relation with the number of train units
    """
    # Load datasets
    train_dataset = pd.read_pickle('datasets/X_train')
    train_dataset['occupancy_rate'] = pd.read_pickle('datasets/y_train')
    # Group by composition
    group_by_composition = train_dataset.groupby(by=['composition'])
    occupancy_rate_by_train_units = group_by_composition.occupancy_rate.mean()
    # Plots
    fig, ax = plt.subplots()
    ax.bar(
        x=range(len(occupancy_rate_by_train_units)),
        height=occupancy_rate_by_train_units,
        tick_label=['Simple train unit', 'Double train unit']
        )
    ax.set_title("Average occupancy rate by composition")
    ax.set_ylabel('Occupancy rate')
    return fig


def price_train_id():
    """ Shows the histogram of average occupancy rates by train ids
    """
    # Load datasets
    train_dataset, y_train, _ = load_data()
    train_dataset['occupancy_rate'] = y_train
    # Group by train ids
    group_by_train_ids = train_dataset.groupby(by=['train'])
    occupancy_rate_by_train_ids = group_by_train_ids.occupancy_rate.mean()
    # Plots
    fig, ax = plt.subplots()
    ax.hist(occupancy_rate_by_train_ids, bins=20)
    ax.set_title("Distribution of the occupancy rate by train ids")
    ax.set_xlabel('Occupancy rate')
    ax.set_ylabel('Number of train ids')
    return fig


def price_station():
    """ Shows the histogram of average occupancy rates by stations
    """
    # Load datasets
    train_dataset, y_train, _ = load_data()
    train_dataset['occupancy_rate'] = y_train
    # Group by weekday
    group_by_stations = train_dataset.groupby(by=['station'])
    occupancy_rate_by_stations = group_by_stations.occupancy_rate.mean()
    # Plots
    fig, ax = plt.subplots()
    ax.hist(occupancy_rate_by_stations, bins=20)
    ax.set_title("Distribution of the occupancy rate by stations")
    ax.set_xlabel('Occupancy rate')
    ax.set_ylabel('Number of stations')
    return fig
