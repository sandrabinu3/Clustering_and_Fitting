import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import curve_fit
import errors as err
import seaborn as sns
import os

def read_file_and_process_data(filename,selected_indicators):
    """
    Reads a CSV file, performs data cleaning, and returns two dataframes.

    Parameters:
    - filename (string): Path to the CSV file.

    Returns:
    - df_years (DataFrame): Yearly data for each country and indicator.
    - df_countries (DataFrame): Data for each country and indicator over the years.
    - melted_2014 (Dataframe): Data of 2014 of selected indicators and all countries.
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
    
    # filter out 2014 data for the selected indicators of all countries
    df_2014_selected = df[df['Indicator Name'].isin(selected_indicators)]
    df_2014=df_2014_selected[df_2014_selected['Year']==2014]
    pivoted_data = df_2014.pivot(index='Country', columns='Indicator Name', values='Value')
    
    # Reset the index to flatten the DataFrame
    pivoted_data = pivoted_data.reset_index()
    # Select only the columns of interest
    selected_columns = ['Country', 'Electricity production from hydroelectric sources (% of total)',
                         'Electricity production from coal sources (% of total)']

    # Create the final melted DataFrame
    melted_2014 = pivoted_data[selected_columns]
    # returning the processed dataframes
    return df_years, df_countries,melted_2014


def clean_and_transposed_df(df_years, countries, indicators):
    """
    Transposed subset data based on selected countries and indicators.

    Parameters:
    - df_years (DataFrame): Yearly data for each country and indicator.
    - countries (list): List of country names.
    - indicators (list): List of indicator names.

    Returns:
    - df_subset (pd.DataFrame): Subset of the data.
    """
    years = list(range(1980, 2014))
    # cleaning the needed data
    df_subset = df_years.loc[(countries, indicators), years]
    # transpose the cleaned df
    df_subset = df_subset.transpose()
    return df_subset


def normalize_dataframe(df):
    """
    Normalizes the values of a dataframe.

    Parameters:
    - df (DataFrame): Dataframe to be normalized.

    Returns:
    - df_normalized (DataFrame): Normalized dataframe.
    """
    # droping the Nan values from the dataframe
    df = df.dropna()
    # selecting numeric columns
    numeric_columns = df.columns[1:]
    # apply StandardScaler for scaling values
    scaler = StandardScaler()
    # Fit and transform the numeric columns
    normalized_values = scaler.fit_transform(df[numeric_columns])
    # Create a new DataFrame with normalized values
    df_normalized = pd.DataFrame(normalized_values, columns=numeric_columns)
    # Add the 'Country' column back to the DataFrame
    df_normalized['Country'] = df['Country']
    # Reorder columns for better readability
    df_normalized = df_normalized[['Country'] + list(numeric_columns)]
    #return normalized dataframe
    return df_normalized


def num_of_Clusters(scaled_df):
    # define an empty list
    silhouette_scores = []
    for num_clusters in range(2, 11):
        # apply Kmeans for finding the number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # find the cluster labels
        cluster_labels = kmeans.fit_predict(scaled_df)
        # find the Silhouette score and append it to the list
        silhouette_scores.append(silhouette_score(scaled_df, cluster_labels))
    #print(silhouette_scores)

    
def perform_kmeans_clustering(num_clusters,df):
    """
    Performs K-Means clustering on the given dataframe.

    Parameters:
    - num_clusters (int): Number of clusters.
    - df (DataFrame): Dataframe for clustering.

    Returns:
    - cluster_labels (array): Labels assigned to each data point.
    - cluster_centers (array): Centers of each clusters 
    """
    # apply KMeans to to clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # use fit_predict to cluster
    cluster_labels = kmeans.fit_predict(df.iloc[:,1:])
    # find the cluster centers
    cluster_centers = kmeans.cluster_centers_
    # define the list of countries
    countries = list(df['Country'])
    # Select one country from each cluster
    selected_countries = select_one_country_from_each_cluster(df,cluster_labels, countries)
    # print("Selected Countries:", selected_countries)
    return cluster_labels,cluster_centers


def plot_clustered_data(df, cluster_labels, cluster_centers):
    """
    Plots the clustered data.

    Parameters:
    - df (DataFrame): Dataframe for clustering.
    - cluster_labels (array): Labels assigned to each data point.
    - cluster_centers (array): Coordinates of cluster centers.
    
    """
    # use the library seaborn for styling
    plt.style.use('seaborn')
    # define the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # plot the clusters 
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1],
                         c=cluster_labels, cmap='viridis')
    # plot the cluster centers
    ax.scatter(cluster_centers[:, 0], cluster_centers[:,
               1], s=200, marker='2', c='red')
    # set labels and title
    ax.set_xlabel('Electricity production from hydroelectric sources (% of total)',fontsize=12)
    ax.set_ylabel('Electricity production from coal sources (% of total)',fontsize=12)
    ax.set_title("K-Means Clustering Results", fontsize=14)
    # set a grid and colorbar
    ax.grid(True)
    plt.colorbar(scatter)
    # save the fig
    # plt.savefig('Clusters.png')
    plt.show()

def select_one_country_from_each_cluster(df,cluster_labels, countries):
    """
    Selects one country from each cluster.

    Parameters:
    - cluster_labels (array): Labels assigned to each data point.
    - countries (list): List of country names.

    Returns:
    - selected_countries (list): One country from each cluster.
    """
    # get the cluster labels
    unique_clusters = np.unique(cluster_labels)
    # list all countries
    countries = list(df['Country'])
    # define an empty list
    selected_countries = []
    for cluster in unique_clusters:
        # Find indices of countries in the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        # Choose one country randomly from the cluster
        selected_country = np.random.choice(df.loc[df.index[cluster_indices], 'Country'])
        # append the selected countries to the lsit
        selected_countries.append(selected_country)
    return selected_countries


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
    energy_use_data = pd.read_csv(filename, skiprows=4) # read the data
    #drop unwanted columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    energy_use_data = energy_use_data.drop(cols_to_drop, axis=1)
    energy_use_data = energy_use_data.rename(
        columns={'Country Name': 'Country'})    # rename the column
    if isinstance(countries, str):  # Convert a single country to a list
        countries = [countries]
    # filter the df
    energy_use_data = energy_use_data[energy_use_data['Country'].isin(countries) &
                                      energy_use_data['Indicator Name'].isin(indicators)]
    #melt the df
    energy_use_data = energy_use_data.melt(id_vars=['Country', 'Indicator Name'],
                                           var_name='Year', value_name='Value')
    # convert the values to numeric
    energy_use_data['Year'] = pd.to_numeric(
        energy_use_data['Year'], errors='coerce')
    energy_use_data['Value'] = pd.to_numeric(
        energy_use_data['Value'], errors='coerce')
    # pivot the new df
    energy_use_data = energy_use_data.pivot_table(index=['Country', 'Indicator Name'],
                                                  columns='Year', values='Value')
    # slice the needed years
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


def predict_future_values(energy_use_data, countries, indicators, start_year, end_year, prediction_years):
    """
    Predicts and plots future values and error range using exponential growth fitting.

    Parameters:
    - energy_use_data (DataFrame): Processed energy use data.
    - indicators (list): List of indicator names.
    - start_year (int): Start year for prediction.
    - end_year (int): End year for prediction.
    - prediction_years (list): Years for which to predict values.
    
    """
    for country in countries:
        # filter the needed data
        data = filter_energy_use_data(energy_use_data, country, indicators, start_year, end_year)
        # initialize an array to store growth rates during curve fitting
        growth_rate = np.zeros(data.shape)
        for i in range(data.shape[0]):
            # fit the data using exponential growth model
            popt, pcov = curve_fit(
                exponential_growth_function, np.arange(data.shape[1]), data.iloc[i])
            # append the groth rates to the array
            growth_rate[i] = popt[1]
            # Print the values
            print(f"\nCountry: {data.index.get_level_values('Country')[i]}")
            print(f"Indicator: {data.index.get_level_values('Indicator Name')[i]}")
            print("Fitted Parameters (popt):", popt)
            print("Covariance Matrix (pcov):", pcov)
        # define the years
        future_years = np.arange(start_year, end_year+1)
        # extent till prediction year
        extended_years = np.arange(start_year, 2026)  
        # define the figure
        fig, ax = plt.subplots()
        for i in range(data.shape[0]):
            # get the country name
            country = data.index.get_level_values('Country')[i]
            # get the indicator name
            indicator = data.index.get_level_values('Indicator Name')[i]
            # Plot historical data
            ax.plot(future_years, data.loc[(country, indicator)],
                    label=f"{country} - Historical", linestyle='-', marker='o')
            # Predict and plot future values
            predicted_values = exponential_growth_function(extended_years - start_year, *popt)
            # standard deviation
            sigma = np.sqrt(np.diag(pcov))
            # lower and upper bounds
            low, up = err.err_ranges(extended_years-start_year,exponential_growth_function, popt,sigma)
            # plot the fitted prediction curve
            ax.plot(extended_years, predicted_values,
                    label=f"{country} - Predicted", linestyle='--', marker='x',color='indigo')
            # plot the error band
            ax.fill_between(extended_years, low, up,
                    color='yellow', alpha=0.4, label='Error range')
            # get the predicted value at 2025
            point_at_2025 = predicted_values[-1]
            # plot the predicted value for 2025
            ax.scatter(2025, point_at_2025, color='green', marker='o')
            ax.annotate(f'{point_at_2025:.2f}', (2025, point_at_2025),
                        textcoords="offset points", xytext=(0,10), ha='center')
        # set the labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Electricity production from coal (%)')
        ax.set_title(f'{country}')
        ax.set_xticks(np.arange(1980,2026,5)) # set the xticks
        ax.legend(loc='best') #set the legend
        plt.savefig(f'fit_curve_prediction_{country}.png') # save fig
    # show the plot
    plt.show()

if __name__ == '__main__':
    # selected indicators
    selected_indicators = [
        'Electricity production from hydroelectric sources (% of total)', 'Electricity production from coal sources (% of total)']
    # reading out processed dfs
    df_years, df_countries,melted_2014 = read_file_and_process_data(
        r"climate_change_wbdata (1).csv",selected_indicators)
    # normalising data
    normalized_data = normalize_dataframe(melted_2014)
    #defining the numeric columns
    numeric_columns = normalized_data.columns[1:]
    #getting the optimal number of clusters
    silhoute_Score(normalized_data[numeric_columns])
    # defining the number of clusters
    num_clusters = 3
    # perform clustering
    cluster_labels,cluster_centers=perform_kmeans_clustering(num_clusters,normalized_data)
    # plot the clustered data
    plot_clustered_data(normalized_data.iloc[:,1:], cluster_labels, cluster_centers)
    # taking one country from each cluster
    countries_from_each_cluster=['Malaysia','Japan','Norway']
    # prediction, fitting and error range
    predict_future_values(r"climate_change_wbdata (1).csv",countries_from_each_cluster,
                      ['Electricity production from coal sources (% of total)'],
                      1980, 2014, range(2015, 2026))

    