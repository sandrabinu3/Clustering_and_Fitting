
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


co2_df=pd.read_csv('co2_n.csv')
pop_df=pd.read_csv('gdp_n.csv')

print(co2_df.columns)
year='2010 [YR2010]'

co2_yr=co2_df[['Country Code',year]]
pop_yr=pop_df[['Country Code',year]]

mer_df_raw=pd.merge(co2_yr,pop_yr,on='Country Code')
mer_df=mer_df_raw.dropna()
scaler=StandardScaler()
scaled_df=scaler.fit_transform(mer_df.drop('Country Code',axis=1))


silhouette_scores=[]
for num_clusters in range(2,11):
    kmeans=KMeans(n_clusters=num_clusters,random_state=42)
    cluster_labels=kmeans.fit_predict(scaled_df)
    silhouette_scores.append(silhouette_score(scaled_df,cluster_labels))

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 as range started from 2
print(optimal_clusters)

kmeans=KMeans(n_clusters=3,random_state=42)
cluster_labels=kmeans.fit_predict(scaled_df)

cen = kmeans.cluster_centers_
print(cen[:,0])
mer_df['Cluster']=cluster_labels


plt.scatter(mer_df[year+'_x'],mer_df[year+'_y'],c=mer_df['Cluster'],cmap='viridis')
plt.scatter(cen[:, 0], cen[:,1], marker='d', color='red', label='Cluster Centers')

plt.xlabel('CO2 Emissions (metric tons per capita)')
plt.ylabel('Population (total)')
plt.show()




