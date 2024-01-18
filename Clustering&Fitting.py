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


def plot_correlation_heatmap(df, size=6):
    """
    Plots a correlation heatmap for the given dataframe.

    Parameters:
    - df (pd.DataFrame): Dataframe for correlation analysis.
    - size (int): Size of the heatmap.

    Returns:
    - None
    """
    corr = df.corr()
    
    # Print the correlation matrix values
    print("Correlation Matrix:")
    print(corr)

    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(corr, cmap='ocean_r')

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    cbar = fig.colorbar(im)

    ax.set_title('Correlation Heatmap of selected countries')
    plt.tight_layout()


def normalize_dataframe(df):
    """
    Normalizes the values of a dataframe.

    Parameters:
    - df (pd.DataFrame): Dataframe to be normalized.

    Returns:
    - df_normalized (pd.DataFrame): Normalized dataframe.
    """
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


def plot_normalized_dataframe(df_normalized):
    """
    Plots a stacked bar chart of the normalized dataframe.

    Parameters:
    - df_normalized (pd.DataFrame): Normalized dataframe.

    Returns:
    - None
    """
    # Set Seaborn style
    sns.set(style="whitegrid", palette="pastel")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    df_normalized.plot(kind='bar', stacked=True, ax=ax)

    # Customize plot details
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Stacked Bar Chart of Normalized Data', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)

    ax.grid(axis='y', linestyle='-', alpha=0.8)

    # Remove top and right spines
    sns.despine(top=True, right=True)

    # Add legend
    ax.legend(title='Legend', title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plot
    plt.show()


def perform_kmeans_clustering(df, num_clusters):
    """
    Performs K-Means clustering on the given dataframe.

    Parameters:
    - df (pd.DataFrame): Dataframe for clustering.
    - num_clusters (int): Number of clusters.

    Returns:
    - cluster_labels (np.array): Labels assigned to each data point.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df)

    return cluster_labels


def plot_clustered_data(df, cluster_labels, cluster_centers):
    """
    Plots the clustered data.

    Parameters:
    - df (pd.DataFrame): Dataframe for clustering.
    - cluster_labels (np.array): Labels assigned to each data point.
    - cluster_centers (np.array): Coordinates of cluster centers.

    Returns:
    - None
    """
    plt.style.use('seaborn')

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1],
                         c=cluster_labels, cmap='PRGn')

    ax.scatter(cluster_centers[:, 0], cluster_centers[:,
               1], s=200, marker='2', c='black')

    ax.set_xlabel(df.columns[0], fontsize=12)
    ax.set_ylabel(df.columns[1], fontsize=12)
    ax.set_title("K-Means Clustering Results", fontsize=14)

    ax.grid(True)
    plt.colorbar(scatter)

    plt.show()


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
    """
    Exponential growth function.

    Parameters:
    - x (array): Input values.
    - a (float): Growth parameter.
    - b (float): Exponential growth rate.

    Returns:
    - y (array): Output values.
    """
    return a * np.exp(b * x)


def calculate_error_ranges(xdata, ydata, popt, pcov, alpha=0.05):
    """
    Calculates error ranges for curve fitting parameters.

    Parameters:
    - xdata (array): Input values.
    - ydata (array): Observed output values.
    - popt (array): Optimal values for the parameters so that the sum of the squared residuals is minimized.
    - pcov (2D array): The estimated covariance of popt.
    - alpha (float): Significance level.

    Returns:
    - ci (array): Confidence intervals for the parameters.
    """
    n = len(ydata)
    m = len(popt)
    df = max(0, n - m)
    tval = -1 * stats.t.ppf(alpha / 2, df)
    residuals = ydata - exponential_growth_function(xdata, *popt)
    stdev = np.sqrt(np.sum(residuals**2) / df)
    ci = tval * stdev * np.sqrt(1 + np.diag(pcov))
    return ci


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

    growth_rate = np.zeros(data.shape)
    for i in range(data.shape[0]):
        popt, pcov = curve_fit(
            exponential_growth_function, np.arange(data.shape[1]), data.iloc[i])
        ci = calculate_error_ranges(
            np.arange(data.shape[1]), data.iloc[i], popt, pcov)
        growth_rate[i] = popt[1]

        # Print the values
        print(f"\nCountry: {data.index.get_level_values('Country')[i]}")
        print(f"Indicator: {data.index.get_level_values('Indicator Name')[i]}")
        print("Fitted Parameters (popt):", popt)
        print("Covariance Matrix (pcov):", pcov)
        print("Confidence Intervals (ci):", ci)

    fig, ax = plt.subplots()
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1]), data.iloc[i],
                label=data.index.get_level_values('Country')[i])
    ax.set_xlabel('Year')
    ax.set_ylabel('Indicator Value')
    ax.set_title(', '.join(indicators))
    ax.legend(loc='best')
    plt.show()


def main():
    df_years, df_countries = read_file_and_process_data(
        r"climate_change_wbdata (1).csv")

    selected_indicators = [
        'Electricity production from hydroelectric sources (% of total)', 'Electricity production from coal sources (% of total)']
    selected_countries = ['India', 'United States', 'China', 'Australia', 'Canada']
    selected_data = subset_data_by_countries_and_indicators(
        df_years, selected_countries, selected_indicators)

    normalized_data = normalize_dataframe(selected_data)

    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    cluster_centers = kmeans.cluster_centers_

    print("Cluster Centers:")
    print(cluster_centers)
    plot_clustered_data(normalized_data, cluster_labels, cluster_centers)

    predict_future_values(r"climate_change_wbdata (1).csv", [
                          'India', 'European Union', 'United States', 'Canada'], ['Electricity production from hydroelectric sources (% of total)'], 1980, 2014)

    plot_correlation_heatmap(selected_data, size=8)

    plot_normalized_dataframe(normalized_data)


if __name__ == '__main__':
    if os.name == 'posix':
        os.environ['OMP_NUM_THREADS'] = '1'
    elif os.name == 'nt':
        os.environ['OMP_NUM_THREADS'] = '1'
    main()
