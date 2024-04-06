# importing the data and library
import numpy as np
from datetime import datetime
import pandas as pd

order_path = '/Users/clam/Desktop/GDE9/finalproject_Dataset/olist_orders_clean.csv'
order = pd.read_csv(order_path)

customer_path = '/Users/clam/Desktop/GDE9/finalproject_Dataset/cleaned_olist_customers_dataset.csv'
customer = pd.read_csv(customer_path)

payment_path = '/Users/clam/Desktop/GDE9/finalproject_Dataset/olist_order_payments_dataset.csv'
payment = pd.read_csv(payment_path)

# group by order_id and count total value 
new_payment = payment.groupby(by='order_id',as_index=False)['payment_value'].sum()

# merge three DF
combined = customer.merge(order,on='customer_id')
combined = combined.merge(new_payment,on='order_id')


# calculating Recency
df_recency = combined.groupby(by= 'customer_unique_id', as_index= False)['order_purchase_timestamp'].max()
df_recency.columns = ['Customer_unique_id','Last_Purchase']

# Convert to datetime
df_recency['Last_Purchase'] = pd.to_datetime(df_recency['Last_Purchase'])  
recent_date = df_recency['Last_Purchase'].max()

# add a column called 'Recency'
df_recency['Recency'] = (recent_date - df_recency['Last_Purchase']).dt.days

# calculating Frequency
df_frequency = combined.drop_duplicates().groupby(by='customer_unique_id', as_index=False)['order_purchase_timestamp'].count()
df_frequency.columns = ['Customer_unique_id','Frequency']

# calculating Monetary Value
df_monetary = combined.groupby(by='customer_unique_id',as_index=False)['payment_value'].sum()
df_monetary.columns=['Customer_unique_id','Monetary']

# merge all three DF in one DF called 'RFM'
RF = df_recency.merge(df_frequency,on='Customer_unique_id')
RFM = RF.merge(df_monetary,on='Customer_unique_id')
RFM


# rank customer's based upon their recency, frequency and monetary score
RFM['R_rank'] = RFM['Recency'].rank(ascending=False)
RFM['F_rank'] = RFM['Frequency'].rank(ascending=True)
RFM['M_rank'] = RFM['Monetary'].rank(ascending=True)
RFM['R_rank_norm'] = (RFM['R_rank'] / (RFM['R_rank'].max()))*100
RFM['F_rank_norm'] = (RFM['F_rank'] / (RFM['F_rank'].max()))*100
RFM['M_rank_norm'] = (RFM['M_rank'] / (RFM['M_rank'].max()))*100

# calculating RFM score
# RFM score = 0.15 * Recency score + 0.28 * Frequency score + 0.57 * Monetary score

RFM['RFM_Score'] = 0.15*RFM['R_rank_norm'] + 0.28* RFM['F_rank_norm'] + 0.57*RFM['M_rank_norm']
RFM['RFM_Score']  *= 0.05
RFM = RFM.round(2)

#drop rank columns
RFM.drop(columns=['R_rank','F_rank','M_rank'],inplace=True)

# rating Customer based upon the RFM score
# >4.5 : Top Customer
# 4.5 > rfm score > 4 : High Value Customer
# 4>rfm score >3 : Medium value customer
# 3>rfm score>1.6 : Low-value customer
# rfm score<1.6 :Lost Customer

RFM['Customer_segment'] = np.where(RFM['RFM_Score'] > 4.5, "Top Value",(np.where(RFM['RFM_Score'] > 4, "High Value",(np.where(RFM['RFM_Score'] > 3, "Medium Value",np.where(RFM['RFM_Score']>1.6,"Low Value","Lost Customer"))))))

RFM.head(20
         )
# visualize customer segments
import matplotlib.pyplot as plt
plt.pie(RFM.Customer_segment.value_counts(),
        labels=RFM.Customer_segment.value_counts().index,
        autopct='%.0f%%')
plt.show()