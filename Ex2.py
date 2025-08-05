import pandas as pd
from sklearn.linear_model import Linear Regression
from sklearn.model_selection import train test split
import matplotlib.pyplot as pit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import Column Transformer
import seaborn as sns
dfpd.read_csv('Ex2.csv')
x = df[['Bedroom','Size', Age, Zipcode']]
ydf['Selling Price']
ct
ColumnTransformer(transformers-[('encoder', OneHotEncoder(),['Zipcode'])], remainder
='passthrough')
xen-ct.fit transform(x)
xtr,xte,ytr,yte-train_test_split(xen,y,test_size=0.2, random_state=42)
model-LinearRegression()
model.fit(xtr,ytr)
ypr-model.predict(xte)
print(ypr)
coefficients model.coef
intercept model.intercept
print("Coefficients:",coefficients)
print("Intercept:", intercept)
plt.figure(figsize-(8.6))
sns.scatterplot(x=yte,y=ypr,color='blue',s=100)
plt.plot([min(yte), max(yte)],[min(yte), max(yte)],r--')
plt.xlabel("Actual selling price")
plt.ylabel("Predicted Selling price")
plt.title("Actual Vs Predicted House Prices")
plt.grid(True)
plt.tight layout()
plt.show()
sns.heatmap(x.corr(),annot True,cmap"coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
