
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

orders = pd.read_csv(r'C:/Users/asus/ecommerce-return-analysis/data/orders.csv')
returns = pd.read_csv(r'C:/Users/asus/ecommerce-return-analysis/data/returns.csv')

df = pd.merge(orders, returns, on='Order_ID', how='left')
df['Return'] = df['Return'].fillna(0)

print("Return value counts after merge:")
print(df['Return'].value_counts())

le_cat = LabelEncoder()
le_loc = LabelEncoder()
le_chn = LabelEncoder()

df['Product_Category_Enc'] = le_cat.fit_transform(df['Product_Category'])
df['Customer_Location_Enc'] = le_loc.fit_transform(df['Customer_Location'])
df['Marketing_Channel_Enc'] = le_chn.fit_transform(df['Marketing_Channel'])

X = df[['Product_Category_Enc', 'Price', 'Customer_Rating', 'Marketing_Channel_Enc']]
y = df['Return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

df['Return_Probability'] = model.predict_proba(X)[:, 1]
df.to_csv(r'C:/Users/asus/Downloads/ecommerce-return-analysis-absolute-path/outputs/return_predictions.csv', index=False)
df[df['Return_Probability'] > 0.7].to_csv(r'C:/Users/asus/Downloads/ecommerce-return-analysis-absolute-path/data/high_risk_products.csv', index=False)
