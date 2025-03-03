import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('K-means test.csv')
df.head()

plt.scatter(df["Age"],df["Income"])
plt.show()

km=KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[["Age","Income"]])
y_predicted

df["cluster"] = y_predicted
#print(df.head())

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Income"],color="green")
plt.scatter(df2.Age,df2["Income"],color="red")
plt.scatter(df3.Age,df3["Income"],color="blue")

plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()