# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 21:31:14 2024

@author: bc975789
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
import sklearn.cluster as cluster


def read_Data(indicator):
    """ 
    Function which takes an indicator name as argument, read the file into a dataframe and returns two dataframes: one with years as columns and one with
    countries as columns.
    """
    years = ['Country Name', '1960 [YR1960]', '1970 [YR1970]', '1980 [YR1980]', '1990 [YR1990]',
             '2000 [YR2000]', '2010 [YR2010]', '2020 [YR2020]']
    yearsN = ['1960', '1970', '1980', '1990', '2000', '2010', '2020']
    rawdata = pd.read_csv('World_Develoment_indicator_POP_GDP.csv')
    rawdata = rawdata.set_index('Country Code')
    rawdata.replace("..", 0, inplace=True)
    data = rawdata[rawdata['Series Name'] ==
                   indicator].loc[:, years]
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
    data_T = data.transpose()
    data_T = data_T.drop('Country Name')
    data_T.reset_index(drop=False, inplace=True)
    data_T = data_T.rename(columns={'index': 'Years'})
    data_T['Years'] = yearsN
    return data, data_T


def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


def data_Fitting_plot(indicator, title, ylabel, image):
    """ Plots the fit parameter
    """
    data, data_T = read_Data(indicator)
    data_T["Years"] = pd.to_numeric(data_T["Years"])
    popt, pcorr = opt.curve_fit(logistics, data_T["Years"], data_T["CHN"],
                                p0=(1.2e8, 0.2, 2003.0))
    data_T["exp"] = logistics(data_T["Years"], *popt)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(data_T["Years"], data_T["CHN"], label="CHINA")
    plt.plot(data_T["Years"], data_T["exp"], label="fit")
    plt.title(title, fontsize=14)
    plt.xlabel('Years', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(image)
    plt.show()
    return


def data_prediction(indicator, title, ylabel, image):
    """ Plots the fit prediction from ranges
    """
    data, data_T = read_Data(indicator)
    data_T["Years"] = pd.to_numeric(data_T["Years"])
    popt, pcorr = opt.curve_fit(logistics, data_T["Years"], data_T["CHN"],
                                p0=(1.2e8, 0.2, 2003.0))
    year = np.arange(1960, 2031)
    forecast = logistics(year, *popt)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(data_T["Years"], data_T["CHN"], label="CHINA")
    plt.plot(year, forecast, label="forecast")
    plt.title(title, fontsize=14)
    plt.xlabel('Years', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(image)
    plt.show()
    return


def data_ErrorRange_plot(indicator,  title, ylabel, image):
    """ plot error ranges with transparency
    """
    data, data_T = read_Data(indicator)
    data_T["Years"] = pd.to_numeric(data_T["Years"])
    popt, pcorr = opt.curve_fit(logistics, data_T["Years"], data_T["CHN"],
                                p0=(1.2e8, 0.2, 2003.0))
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    data_T["logistics"] = logistics(data_T["Years"], *popt)
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.arange(1960, 2030)
    lower, upper = err.err_ranges(years, logistics, popt, sigmas)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(data_T["Years"], data_T["CHN"], label="CHINA")
    plt.plot(data_T["Years"], data_T["logistics"], label="fit")
    plt.fill_between(years, lower, upper, alpha=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel('Years', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.savefig(image)
    plt.show()
    return


def clustering(indicator, title, image):
    """ plot Population and GDP clusters of different countries
    """
    data, data_T = read_Data(indicator)
    df_ex = data[['1990 [YR1990]', '2020 [YR2020]']]
    df_norm, df_min, df_max = ct.scaler(df_ex)
    ncluster = 4
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)  # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0), dpi=300)
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_norm['1990 [YR1990]'],
                df_norm['2020 [YR2020]'], 40, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 65, "k", marker="d")
    plt.title(title, fontsize=14)
    plt.xlabel("1990", fontsize=14)
    plt.ylabel("2015", fontsize=14)

    # Applying the backscale function to convert the cluster centre
    scen = ct.backscale(cen, df_min, df_max)
    xcen = scen[:, 0]
    ycen = scen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0), dpi=300)
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_ex['1990 [YR1990]'], df_ex['2020 [YR2020]'],
                40, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 65, "k", marker="d")
    plt.title(title, fontsize=14)
    plt.xlabel("1990", fontsize=14)
    plt.ylabel("2015", fontsize=14)
    plt.legend()
    plt.savefig(image)
    plt.show()
    return


if __name__ == "__main__":
    #calling function to visualize all the plots
    data_Fitting_plot('GDP per capita (current US$)', 'GDP per capita of CHINA',
                      'GDP per capita (current US$)', 'Fit GDP.jpg')
    data_prediction('GDP per capita (current US$)', 'GDP per capita prediction for 2030 of CHINA',
                    'GDP per capita (current US$)', 'Prediction GDP.jpg')
    data_ErrorRange_plot('GDP per capita (current US$)', 'GDP per capita Error Ranges for 2030 of CHINA',
                         'GDP per capita (current US$)', 'Error Ranges GDP.jpg')
    data_Fitting_plot('Population, total', 'Population of CHINA',
                      'Population (million)', 'Fit POP.jpg')
    data_prediction('Population, total', 'Population prediction for 2030 of CHINA',
                    'Population (million)', 'Prediction POP.jpg')
    data_ErrorRange_plot('Population, total', 'Population Error Ranges for 2030 of CHINA',
                         'Population (million)', 'Error Ranges POP.jpg')
    clustering('GDP per capita (current US$)',
               'GDP Clusters 1990 vs 2020', 'Cluster GDP.jpg')
    clustering('Population, total',
               'Population Clusters 1990 vs 2020', 'Cluster POP.jpg')