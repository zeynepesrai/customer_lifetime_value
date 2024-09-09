import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format

df_ = pd.read_excel(
    "C:/Users/zei/Desktop/CRM_analytics/Kurs Materyalleri(CRM_Analytics)/datasets/online_retail_II.xlsx",
    sheet_name="Year 2009-2010")

#Veriyi hazırlama

df = df_.copy()
df.head()
df.isnull().sum()

df[~df["Invoice"].str.contains("C", na=False)]  #iade ürünlerin veri setinden çıkarılması
df.describe().T

df = df[(df["Quantity"] > 0)]

df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv = df.groupby("Customer ID").agg({'Invoice': lambda Invoice: Invoice.nunique(),
                                      'Quantity': lambda Quantity: Quantity.nunique(),
                                      'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

#total_transaction: recency, total_perice:monetary
cltv.columns = ['total_transaction', 'total_unit', 'total_price']

#Average Order Value (total_price / total_transaction)
cltv["average_order_value"] = cltv["total_price"] / cltv["total_transaction"]

#Purchase Frequency (total_transaction / total_number_of_customers)
cltv["purchase_frequency"] = cltv["total_transaction"] / cltv.shape[0]

#Repeat Rate & Churn Rate (transaction_number_more_than_one / total_number_of_customers
repeat_rate = cltv[cltv["total_transaction"] > 1].shape[0] / cltv.shape[0]

churn_rate = 1 - repeat_rate

#Profit Margin (total_price * 0.10)

cltv["profit_margin"] = cltv["total_price"] * 0.10

#Customer Value (average_order_value * purchase_frequency)

cltv['customer_value'] = cltv['average_order_value'] * cltv['purchase_frequency']

#Customer Lifetime Value (customer_value / churn_rate) * profit_margin

cltv['cltv'] = (cltv['customer_value'] / churn_rate) * cltv['profit_margin']
cltv.sort_values(by="cltv", ascending=False)

#Segment creation
cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
cltv.groupby("segment").agg({"sum", "mean", "count"})

cltv.to_csv("cltv_segments.csv")


#Function

def create_cltv(dataframe, profit=0.10):
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    cltv = dataframe.groupby("Customer ID").agg({'Invoice': lambda x: x.nunique(),
                                                 'Quantity': lambda x: x.nunique(),
                                                 'TotalPrice': lambda x: x.sum()})
    cltv.columns = ['total_transaction', 'total_unit', 'total_price']
    # Average Order Value
    cltv["average_order_value"] = cltv["total_price"] / cltv["total_transaction"]
    # Purchase Frequency
    cltv["purchase_frequency"] = cltv["total_transaction"] / cltv.shape[0]
    # Repeat Rate & Churn Rate
    repeat_rate = cltv[cltv["total_transaction"] > 1].shape[0] / cltv.shape[0]
    churn_rate = 1 - repeat_rate
    # Profit Margin (total_price * 0.10)
    cltv["profit_margin"] = cltv["total_price"] * 0.10
    # Customer Value
    cltv['customer_value'] = cltv['average_order_value'] * cltv['purchase_frequency']
    # Customer Lifetime Value
    cltv['cltv'] = (cltv['customer_value'] / churn_rate) * cltv['profit_margin']
    # Segment
    cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
    cltv.to_csv("cltv_segments.csv")

    return cltv


ddf = df.copy()
new_df = create_cltv(ddf)
