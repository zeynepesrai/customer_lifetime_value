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


ddf = df_.copy()
new_df = create_cltv(ddf)

# Customer Lifetime Value Prediction
# CLTV = Expected Number Of Transiction (Purchase Frequency) * Expected Average Profit (Average Order Value)
# CLTV = BG/NBD Model * Gamma Gamma Submodel
# BG/NBD : Buy Till You Die ( Expected Number Of Transiction (Purchase Frequency) tahminlemek için
# Gamma Gamma Submodel: Expected Average Profit (Average Order Value) tahminlemek için
# Transaction Process (Buy) + Dropout Process (Till You Die)
# Transaction rateler her müşteriye göre değişir ve tüm kitle için gamma dağılır (r,a)
# Dropout rateler her bir müşteriye göre değişir ve tüm kitle için beta dağılır (a,b)

import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


df_ = pd.read_excel(
    "C:/Users/zei/Desktop/CRM_analytics/Kurs Materyalleri(CRM_Analytics)/datasets/online_retail_II.xlsx",
    sheet_name="Year 2009-2010")

df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2010, 12, 11)

# Buradaki recency değeri : Son satın alma üzerinden geçen zaman. Haftalık, kullanıcı özelinde
# müşterinin kendi içinde son satın alması ve ilk satın alması üzerinden hesaplanır
# T müşteri yaşı, haftalık. Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış
# frequency: tekrar eden toplam satın alma sayısı. 1'den büyük olmalıdır. frequency > 1
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df.describe().T
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# BG-NBD modelinin kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# Soru: 1 Hafta içerisinde en çok satın alma beklenen 10 müşteri kimlerdir?

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

bgf.predict(1,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

# Soru: 1 Ay içerisinde en çok satın alma beklenen 10 müşteri kimlerdir?
bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])

# 1 aylık periyotta şirketin beklediği toplam satışlar
bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()
cltv_df["expected_purc_1_month"] = bgf.predict(4 * 3,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

# 3 aylık periyotta şirketin beklediği toplam satışlar
bgf.predict(4 * 3,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])

# Tahmin sonuçlarının değerlendirilmesi

plot_period_transactions(bgf)
plt.show()

# Gamma-Gamma Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# BG_NBD ve GG Modeli ile CLTV'nin hesaplanması

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3, # 3 Aylık
                                   freq="W", # T nin frekans bilgisi (veri haftalık mı aylık mı günlük mü)
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# CLTV'ye göre Segmentlerin Oluşturulması

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

# Çalışmanın Fonksiyonlaştırılması

def create_cltv_p(dataframe, month=3):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]

    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2010, 12, 11)

    cltv_df = dataframe.groupby("Customer ID").agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]
    cltv_df = cltv_df[cltv_df["frequency"] > 1]
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # BG-NBD modelinin kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    # Soru: 1 Hafta içerisinde en çok satın alma beklenen 10 müşteri kimlerdir?
    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df["frequency"],
                                                  cltv_df["recency"],
                                                  cltv_df["T"])

    # Soru: 1 Ay içerisinde en çok satın alma beklenen 10 müşteri kimlerdir?
    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

    # 1 aylık periyotta şirketin beklediği toplam satışlar
    cltv_df["expected_purc_1_month"] = bgf.predict(4 * 3,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])
    # 3 aylık periyotta şirketin beklediği toplam satışlar
    cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=3,  # 3 Aylık
                                       freq="W",  # T nin frekans bilgisi (veri haftalık mı aylık mı günlük mü)
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df_new = df_.copy()
cltv_final2 = create_cltv_p(df_new)
cltv_final2.to_csv("cltv_prediction.csv")