import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
import os

def read_file_and_process_data(filename):
    """
    Reads a CSV file, performs data cleaning, and returns two dataframes.

    Parameters:
    - filename (str): Path to the CSV file.

    Returns:
    - df_years (pd.DataFrame): Yearly data for each country and indicator.
    - df_countries (pd.DataFrame): Data for each country and indicator over the years.
    """
    df = pd.read_csv(filename, skiprows=4)

    # Drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # Rename columns
    df = df.rename(columns={'Country Name': 'Country'})

    # Reshape the data
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # Convert 'Year' and 'Value' columns to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Pivot tables for analysis
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # Clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def subset_data_by_countries_and_indicators(df_years, countries, indicators):
    """
    Subsets data based on selected countries and indicators.

    Parameters:
    - df_years (pd.DataFrame): Yearly data for each country and indicator.
    - countries (list): List of country names.
    - indicators (list): List of indicator names.

    Returns:
    - df_subset (pd.DataFrame): Subset of the data.
    """
    years = list(range(1980, 2014))
    df_subset = df_years.loc[(countries, indicators), years]
    df_subset = df_subset.transpose()
    return df_subset

def filter_energy_use_data(filename, countries, indicators, start_year, end_year):
    """
    Filters and processes energy use data based on specified criteria.

    Parameters:
    - filename (str): Path to the CSV file.
    - countries (list): List of country names.
    - indicators (list): List of indicator names.
    - start_year (int): Start year for the data.
    - end_year (int): End year for the data.

    Returns:
    - energy_use_data (pd.DataFrame): Processed energy use data.
    """
    energy_use_data = pd.read_csv(filename, skiprows=4)
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    energy_use_data = energy_use_data.drop(cols_to_drop, axis=1)
    energy_use_data = energy_use_data.rename(
        columns={'Country Name': 'Country'})
    energy_use_data = energy_use_data[energy_use_data['Country'].isin(countries) &
                                      energy_use_data['Indicator Name'].isin(indicators)]
    energy_use_data = energy_use_data.melt(id_vars=['Country', 'Indicator Name'],
                                           var_name='Year', value_name='Value')
    energy_use_data['Year'] = pd.to_numeric(
        energy_use_data['Year'], errors='coerce')
    energy_use_data['Value'] = pd.to_numeric(
        energy_use_data['Value'], errors='coerce')
    energy_use_data = energy_use_data.pivot_table(index=['Country', 'Indicator Name'],
                                                  columns='Year', values='Value')
    energy_use_data = energy_use_data.loc[:, start_year:end_year]

    return energy_use_data


def exponential_growth_function(x, a, b):
    return a * np.exp(b * x)

def calculate_error_ranges(x, y, popt, pcov):
    perr = np.sqrt(np.diag(pcov))
    return perr * sem(y)

def predict_future_values(energy_use_data, countries, indicators, start_year, end_year):
    """
    Predicts future values using exponential growth fitting.

    Parameters:
    - energy_use_data (pd.DataFrame): Processed energy use data.
    - countries (list): List of country names.
    - indicators (list): List of indicator names.
    - start_year (int): Start year for prediction.
    - end_year (int): End year for prediction.

    Returns:
    - None
    """
    data = filter_energy_use_data(energy_use_data, countries,
                                  indicators, start_year, end_year)
    
    for i in range(data.shape[0]):
        country = data.index.get_level_values('Country')[i]
        indicator = data.index.get_level_values('Indicator Name')[i]

        x = np.arange(data.shape[1])
        y = data.iloc[i]

        popt, pcov = curve_fit(exponential_growth_function, x, y)
        growth_rate = popt[1]

        ci = calculate_error_ranges(x, y, popt, pcov)

        # Predict future values
        future_x = np.arange(start_year, end_year+1)
        future_y = predict_with_confidence_intervals(exponential_growth_function, popt, pcov, future_x)


        print(f"\nCountry: {country}")
        print(f"Indicator: {indicator}")
        print("Fitted Parameters (popt):", popt)
        print("Covariance Matrix (pcov):", pcov)
        print("Confidence Intervals (ci):", ci)
        print("Future Predictions:")
        print(f"Year\tValue\tConfidence Interval")
        for j in range(len(future_x)):
            print(f"{future_x[j]}\t{future_y[j]:.2f}\t[{future_ci[j][0]:.2f}, {future_ci[j][1]:.2f}]")

        # Plotting
        plt.plot(x, y, label=country)
        plt.fill_between(x, y - ci, y + ci, color='gray', alpha=0.2, label='Confidence Interval')
        plt.plot(future_x, future_y, '--', label=f"Prediction ({growth_rate:.2f})")

    plt.xlabel('Year')
    plt.ylabel('Indicator Value')
    plt.title(', '.join(indicators))
    plt.legend(loc='best')
    plt.show()

def predict_with_confidence_intervals(func, popt, pcov, x):
    """
    Predicts values with confidence intervals.
    """
    perr = np.sqrt(np.diag(pcov))
    ci = perr * sem(x)
    y = func(x, *popt)
    return y, ci

predict_future_values(r"climate_change_wbdata (1).csv", [
                        'India', 'European Union', 'United States', 'Canada'], ['Electricity production from hydroelectric sources (% of total)'], 1980, 2030)
