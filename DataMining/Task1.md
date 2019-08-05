
# 数据挖掘-预测贷款用户是否逾期
## Task1：数据分析（2天）

-------
### 1 &emsp;数据类型的分析

**数据说明：** 

这份数据集是金融数据（非原始数据，已经处理过了），我们要做的是预测贷款用户是否会逾期。

表格中 "status" 是结果标签：0表示未逾期，1表示逾期。

 - **导入宏包**


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
```

 - 导入数据


```python
data = pd.read_csv('./data/data.csv', encoding='gbk')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>custid</th>
      <th>trade_no</th>
      <th>bank_card_no</th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>2791858</td>
      <td>20180507115231274000000023057383</td>
      <td>卡号1</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>...</td>
      <td>2900.0</td>
      <td>1688.0</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>534047</td>
      <td>20180507121002192000000023073000</td>
      <td>卡号1</td>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>...</td>
      <td>3500.0</td>
      <td>1758.0</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>2849787</td>
      <td>20180507125159718000000023114911</td>
      <td>卡号1</td>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>...</td>
      <td>1600.0</td>
      <td>1250.0</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1809708</td>
      <td>20180507121358683000000388283484</td>
      <td>卡号1</td>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>...</td>
      <td>3200.0</td>
      <td>1541.0</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>2499829</td>
      <td>20180507115448545000000388205844</td>
      <td>卡号1</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>...</td>
      <td>2300.0</td>
      <td>1630.0</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>



 - 查看数据属性


```python
data.info()

print()
print("共有数据集：", data.shape[0])
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4754 entries, 0 to 4753
    Data columns (total 90 columns):
    Unnamed: 0                                    4754 non-null int64
    custid                                        4754 non-null int64
    trade_no                                      4754 non-null object
    bank_card_no                                  4754 non-null object
    low_volume_percent                            4752 non-null float64
    middle_volume_percent                         4752 non-null float64
    take_amount_in_later_12_month_highest         4754 non-null int64
    trans_amount_increase_rate_lately             4751 non-null float64
    trans_activity_month                          4752 non-null float64
    trans_activity_day                            4752 non-null float64
    transd_mcc                                    4752 non-null float64
    trans_days_interval_filter                    4746 non-null float64
    trans_days_interval                           4752 non-null float64
    regional_mobility                             4752 non-null float64
    student_feature                               1756 non-null float64
    repayment_capability                          4754 non-null int64
    is_high_user                                  4754 non-null int64
    number_of_trans_from_2011                     4752 non-null float64
    first_transaction_time                        4752 non-null float64
    historical_trans_amount                       4754 non-null int64
    historical_trans_day                          4752 non-null float64
    rank_trad_1_month                             4752 non-null float64
    trans_amount_3_month                          4754 non-null int64
    avg_consume_less_12_valid_month               4752 non-null float64
    abs                                           4754 non-null int64
    top_trans_count_last_1_month                  4752 non-null float64
    avg_price_last_12_month                       4754 non-null int64
    avg_price_top_last_12_valid_month             4650 non-null float64
    reg_preference_for_trad                       4752 non-null object
    trans_top_time_last_1_month                   4746 non-null float64
    trans_top_time_last_6_month                   4746 non-null float64
    consume_top_time_last_1_month                 4746 non-null float64
    consume_top_time_last_6_month                 4746 non-null float64
    cross_consume_count_last_1_month              4328 non-null float64
    trans_fail_top_count_enum_last_1_month        4738 non-null float64
    trans_fail_top_count_enum_last_6_month        4738 non-null float64
    trans_fail_top_count_enum_last_12_month       4738 non-null float64
    consume_mini_time_last_1_month                4728 non-null float64
    max_cumulative_consume_later_1_month          4754 non-null int64
    max_consume_count_later_6_month               4746 non-null float64
    railway_consume_count_last_12_month           4742 non-null float64
    pawns_auctions_trusts_consume_last_1_month    4754 non-null int64
    pawns_auctions_trusts_consume_last_6_month    4754 non-null int64
    jewelry_consume_count_last_6_month            4742 non-null float64
    status                                        4754 non-null int64
    source                                        4754 non-null object
    first_transaction_day                         4752 non-null float64
    trans_day_last_12_month                       4752 non-null float64
    id_name                                       4478 non-null object
    apply_score                                   4450 non-null float64
    apply_credibility                             4450 non-null float64
    query_org_count                               4450 non-null float64
    query_finance_count                           4450 non-null float64
    query_cash_count                              4450 non-null float64
    query_sum_count                               4450 non-null float64
    latest_query_time                             4450 non-null object
    latest_one_month_apply                        4450 non-null float64
    latest_three_month_apply                      4450 non-null float64
    latest_six_month_apply                        4450 non-null float64
    loans_score                                   4457 non-null float64
    loans_credibility_behavior                    4457 non-null float64
    loans_count                                   4457 non-null float64
    loans_settle_count                            4457 non-null float64
    loans_overdue_count                           4457 non-null float64
    loans_org_count_behavior                      4457 non-null float64
    consfin_org_count_behavior                    4457 non-null float64
    loans_cash_count                              4457 non-null float64
    latest_one_month_loan                         4457 non-null float64
    latest_three_month_loan                       4457 non-null float64
    latest_six_month_loan                         4457 non-null float64
    history_suc_fee                               4457 non-null float64
    history_fail_fee                              4457 non-null float64
    latest_one_month_suc                          4457 non-null float64
    latest_one_month_fail                         4457 non-null float64
    loans_long_time                               4457 non-null float64
    loans_latest_time                             4457 non-null object
    loans_credit_limit                            4457 non-null float64
    loans_credibility_limit                       4457 non-null float64
    loans_org_count_current                       4457 non-null float64
    loans_product_count                           4457 non-null float64
    loans_max_limit                               4457 non-null float64
    loans_avg_limit                               4457 non-null float64
    consfin_credit_limit                          4457 non-null float64
    consfin_credibility                           4457 non-null float64
    consfin_org_count_current                     4457 non-null float64
    consfin_product_count                         4457 non-null float64
    consfin_max_limit                             4457 non-null float64
    consfin_avg_limit                             4457 non-null float64
    latest_query_day                              4450 non-null float64
    loans_latest_day                              4457 non-null float64
    dtypes: float64(70), int64(13), object(7)
    memory usage: 3.3+ MB
    
    共有数据集： 4754
    

 - 数据类型


```python
for i,name in enumerate(data.columns):
    name_sum = data[name].value_counts().shape[0]
    print("{:2}、{:40}      The number of types of features is：{}".format(i + 1, name, name_sum))
```

     1、Unnamed: 0                                    The number of types of features is：4754
     2、custid                                        The number of types of features is：4754
     3、trade_no                                      The number of types of features is：4754
     4、bank_card_no                                  The number of types of features is：1
     5、low_volume_percent                            The number of types of features is：40
     6、middle_volume_percent                         The number of types of features is：90
     7、take_amount_in_later_12_month_highest         The number of types of features is：166
     8、trans_amount_increase_rate_lately             The number of types of features is：782
     9、trans_activity_month                          The number of types of features is：84
    10、trans_activity_day                            The number of types of features is：512
    11、transd_mcc                                    The number of types of features is：41
    12、trans_days_interval_filter                    The number of types of features is：147
    13、trans_days_interval                           The number of types of features is：114
    14、regional_mobility                             The number of types of features is：5
    15、student_feature                               The number of types of features is：2
    16、repayment_capability                          The number of types of features is：2390
    17、is_high_user                                  The number of types of features is：2
    18、number_of_trans_from_2011                     The number of types of features is：70
    19、first_transaction_time                        The number of types of features is：1693
    20、historical_trans_amount                       The number of types of features is：4524
    21、historical_trans_day                          The number of types of features is：476
    22、rank_trad_1_month                             The number of types of features is：20
    23、trans_amount_3_month                          The number of types of features is：3524
    24、avg_consume_less_12_valid_month               The number of types of features is：12
    25、abs                                           The number of types of features is：1697
    26、top_trans_count_last_1_month                  The number of types of features is：8
    27、avg_price_last_12_month                       The number of types of features is：330
    28、avg_price_top_last_12_valid_month             The number of types of features is：20
    29、reg_preference_for_trad                       The number of types of features is：5
    30、trans_top_time_last_1_month                   The number of types of features is：28
    31、trans_top_time_last_6_month                   The number of types of features is：97
    32、consume_top_time_last_1_month                 The number of types of features is：28
    33、consume_top_time_last_6_month                 The number of types of features is：94
    34、cross_consume_count_last_1_month              The number of types of features is：19
    35、trans_fail_top_count_enum_last_1_month        The number of types of features is：15
    36、trans_fail_top_count_enum_last_6_month        The number of types of features is：25
    37、trans_fail_top_count_enum_last_12_month       The number of types of features is：26
    38、consume_mini_time_last_1_month                The number of types of features is：1971
    39、max_cumulative_consume_later_1_month          The number of types of features is：863
    40、max_consume_count_later_6_month               The number of types of features is：29
    41、railway_consume_count_last_12_month           The number of types of features is：6
    42、pawns_auctions_trusts_consume_last_1_month      The number of types of features is：572
    43、pawns_auctions_trusts_consume_last_6_month      The number of types of features is：2730
    44、jewelry_consume_count_last_6_month            The number of types of features is：7
    45、status                                        The number of types of features is：2
    46、source                                        The number of types of features is：1
    47、first_transaction_day                         The number of types of features is：1693
    48、trans_day_last_12_month                       The number of types of features is：132
    49、id_name                                       The number of types of features is：4309
    50、apply_score                                   The number of types of features is：205
    51、apply_credibility                             The number of types of features is：41
    52、query_org_count                               The number of types of features is：46
    53、query_finance_count                           The number of types of features is：25
    54、query_cash_count                              The number of types of features is：17
    55、query_sum_count                               The number of types of features is：74
    56、latest_query_time                             The number of types of features is：207
    57、latest_one_month_apply                        The number of types of features is：36
    58、latest_three_month_apply                      The number of types of features is：56
    59、latest_six_month_apply                        The number of types of features is：65
    60、loans_score                                   The number of types of features is：247
    61、loans_credibility_behavior                    The number of types of features is：25
    62、loans_count                                   The number of types of features is：134
    63、loans_settle_count                            The number of types of features is：123
    64、loans_overdue_count                           The number of types of features is：26
    65、loans_org_count_behavior                      The number of types of features is：41
    66、consfin_org_count_behavior                    The number of types of features is：19
    67、loans_cash_count                              The number of types of features is：32
    68、latest_one_month_loan                         The number of types of features is：14
    69、latest_three_month_loan                       The number of types of features is：31
    70、latest_six_month_loan                         The number of types of features is：67
    71、history_suc_fee                               The number of types of features is：171
    72、history_fail_fee                              The number of types of features is：151
    73、latest_one_month_suc                          The number of types of features is：19
    74、latest_one_month_fail                         The number of types of features is：41
    75、loans_long_time                               The number of types of features is：202
    76、loans_latest_time                             The number of types of features is：232
    77、loans_credit_limit                            The number of types of features is：54
    78、loans_credibility_limit                       The number of types of features is：33
    79、loans_org_count_current                       The number of types of features is：32
    80、loans_product_count                           The number of types of features is：32
    81、loans_max_limit                               The number of types of features is：91
    82、loans_avg_limit                               The number of types of features is：961
    83、consfin_credit_limit                          The number of types of features is：327
    84、consfin_credibility                           The number of types of features is：24
    85、consfin_org_count_current                     The number of types of features is：19
    86、consfin_product_count                         The number of types of features is：20
    87、consfin_max_limit                             The number of types of features is：175
    88、consfin_avg_limit                             The number of types of features is：1677
    89、latest_query_day                              The number of types of features is：210
    90、loans_latest_day                              The number of types of features is：235
    

 - 数据统计


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>custid</th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4754.000000</td>
      <td>4.754000e+03</td>
      <td>4752.000000</td>
      <td>4752.000000</td>
      <td>4754.000000</td>
      <td>4751.000000</td>
      <td>4752.000000</td>
      <td>4752.000000</td>
      <td>4752.000000</td>
      <td>4746.000000</td>
      <td>...</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4457.000000</td>
      <td>4450.000000</td>
      <td>4457.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6008.414178</td>
      <td>1.690993e+06</td>
      <td>0.021806</td>
      <td>0.901294</td>
      <td>1940.197728</td>
      <td>14.160674</td>
      <td>0.804411</td>
      <td>0.365425</td>
      <td>17.502946</td>
      <td>29.029920</td>
      <td>...</td>
      <td>3390.038142</td>
      <td>1820.357864</td>
      <td>9187.009199</td>
      <td>76.042630</td>
      <td>4.732331</td>
      <td>5.227507</td>
      <td>16153.690823</td>
      <td>8007.696881</td>
      <td>24.112809</td>
      <td>55.181512</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3452.071428</td>
      <td>1.034235e+06</td>
      <td>0.041527</td>
      <td>0.144856</td>
      <td>3923.971494</td>
      <td>694.180473</td>
      <td>0.196920</td>
      <td>0.170196</td>
      <td>4.475616</td>
      <td>22.722432</td>
      <td>...</td>
      <td>1474.206546</td>
      <td>583.418291</td>
      <td>7371.257043</td>
      <td>14.536819</td>
      <td>2.974596</td>
      <td>3.409292</td>
      <td>14301.037628</td>
      <td>5679.418585</td>
      <td>37.725724</td>
      <td>53.486408</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>1.140000e+02</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.120000</td>
      <td>0.033000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3106.000000</td>
      <td>7.593358e+05</td>
      <td>0.010000</td>
      <td>0.880000</td>
      <td>0.000000</td>
      <td>0.615000</td>
      <td>0.670000</td>
      <td>0.233000</td>
      <td>15.000000</td>
      <td>16.000000</td>
      <td>...</td>
      <td>2300.000000</td>
      <td>1535.000000</td>
      <td>4800.000000</td>
      <td>77.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>7800.000000</td>
      <td>4737.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6006.500000</td>
      <td>1.634942e+06</td>
      <td>0.010000</td>
      <td>0.960000</td>
      <td>500.000000</td>
      <td>0.970000</td>
      <td>0.860000</td>
      <td>0.350000</td>
      <td>17.000000</td>
      <td>23.000000</td>
      <td>...</td>
      <td>3100.000000</td>
      <td>1810.000000</td>
      <td>7700.000000</td>
      <td>79.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13800.000000</td>
      <td>7050.000000</td>
      <td>14.000000</td>
      <td>36.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8999.000000</td>
      <td>2.597905e+06</td>
      <td>0.020000</td>
      <td>0.990000</td>
      <td>2000.000000</td>
      <td>1.600000</td>
      <td>1.000000</td>
      <td>0.480000</td>
      <td>20.000000</td>
      <td>32.000000</td>
      <td>...</td>
      <td>4300.000000</td>
      <td>2100.000000</td>
      <td>11700.000000</td>
      <td>80.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>20400.000000</td>
      <td>10000.000000</td>
      <td>24.000000</td>
      <td>91.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11992.000000</td>
      <td>4.004694e+06</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>68000.000000</td>
      <td>47596.740000</td>
      <td>1.000000</td>
      <td>0.941000</td>
      <td>42.000000</td>
      <td>285.000000</td>
      <td>...</td>
      <td>10000.000000</td>
      <td>6900.000000</td>
      <td>87100.000000</td>
      <td>87.000000</td>
      <td>18.000000</td>
      <td>20.000000</td>
      <td>266400.000000</td>
      <td>82800.000000</td>
      <td>360.000000</td>
      <td>323.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 83 columns</p>
</div>



 - **结果标签分布情况**


```python
data.status.value_counts()
```




    0    3561
    1    1193
    Name: status, dtype: int64



**结论一：** 可以看到数据共有4754行，其中浮点型数据70列、整数型13列、object 型7列，**需要对 object 类型的数据进行处理才能建立模型。**

**结论二：** 数据在整体上是相当完整的，**除了 student_feature 这个标签有较多缺失值以外，其他数据仅有很小的缺失。**

**结论三：** **大部分数据特征存在拖尾现象，** 即特征的 max(min) 明显偏离其 mean ，需要对这些数据进行进一步的处理才能建立准确的模型。

**结论四：** **数据集正例与负例数量存在显著差异，** 需要使用 **分层抽样** 的方式对数据集进行抽样划分，否则会导致 **模型对未逾期结果预测较为准确，而对逾期结果预测偏差较大。**

**注意：** 有的列像 'first_transaction_time'，它的值其实是日期的形式类似 20130817 这样，**pandas 把它认为是 int 型的**，其实不是，这也需要注意。

-------
### 2&emsp;无关特征的删除

显然，上述特征中存在一些与贷款用户是否会逾期无关的特征。

经分析，无关特征及字段如下：
 - 'Unnamed: 0'
 - 客户id：'custid'
 - 流水号：'trade_no'
 - 卡号：'bank_card_no'
 - 资源：'source'
 - 客户姓名：'id_name'
 - 最新查询时间：'latest_query_time'
 - 最新贷款时间：'loans_latest_time'


```python
data = data.drop(['Unnamed: 0', 'custid', 'trade_no', 'bank_card_no', 'source', 'id_name', 'latest_query_time','loans_latest_time'], axis = 1)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2900.0</td>
      <td>1688.0</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>3500.0</td>
      <td>1758.0</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1600.0</td>
      <td>1250.0</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3200.0</td>
      <td>1541.0</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>2300.0</td>
      <td>1630.0</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



--------
### 3&emsp;数据类型转换

经过无关特征删除后，我们还有两个特征需要进行转换：

 -  object 类型特征（城市等级：'reg_preference_for_trad'）
 
 
 - 时间格式特征（首次交易时间：'first_transaction_time_day'）

**3.1&ensp;&ensp;reg_preference_for_trad 转换**

首先我们查看一下该特征下有哪些数据


```python
print(data.reg_preference_for_trad.unique())
```

    ['一线城市' '三线城市' '境外' '二线城市' '其他城市' nan]
    

这里我们先简单的进行数据转换，特征提取时可以采用 one-hot 编码改进进行改进。


```python
for i in range(0, data.shape[0]):
    if data.reg_preference_for_trad[i] == '一线城市':
        data.reg_preference_for_trad[i] = 1
    elif data.reg_preference_for_trad[i] == '二线城市':
        data.reg_preference_for_trad[i] = 2
    elif data.reg_preference_for_trad[i] == '三线城市':
        data.reg_preference_for_trad[i] = 3
    elif data.reg_preference_for_trad[i] == '境外':
        data.reg_preference_for_trad[i] = 4
    elif data.reg_preference_for_trad[i] == '其他城市':
        data.reg_preference_for_trad[i] = 5
```

**3.2  first_transaction_time_day 转换**


```python
tmpdf = pd.DataFrame()
tmpdf['first_transaction_time_year'] = pd.to_datetime(data['first_transaction_time']).dt.year
tmpdf['first_transaction_time_month'] = pd.to_datetime(data['first_transaction_time']).dt.month
tmpdf['first_transaction_time_day'] = pd.to_datetime(data['first_transaction_time']).dt.day
data[tmpdf.columns] = tmpdf
data = data.drop('first_transaction_time_day', axis = 1)
```

**3.3&ensp;&ensp;结果查看**


```python
data = data.convert_objects(convert_numeric=True)
print(data.dtypes.value_counts())
```

    float64    73
    int64      11
    dtype: int64
    

至此，数据类型的转换工作完成

--------
### 4&emsp;处理缺失值

 - 删除缺失值较多的行列
 
 若某个特征的缺失值超过 30%，那么它会损失特征的关键信息。
 
 同样若某个样本的缺失值过多，那么它也不再具有统计学意义。
 
 因此我们对这样的行或列直接进行删除。


```python
data = data.dropna(axis=1, thresh = 1000)#删除缺失值超过 70% 的列
data = data.dropna(axis=0, thresh = 75)#删除缺失值超过7个的行
```

 - 填充缺失值
 
 对于剩余的缺失值，这里采用 **平均值填充法** 进行填充。


```python
data = data.fillna(data.median())

data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
      <th>first_transaction_time_year</th>
      <th>first_transaction_time_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>2000</td>
      <td>7.59</td>
      <td>1.00</td>
      <td>0.733</td>
      <td>27.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>11200.0</td>
      <td>80.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>20400.0</td>
      <td>8130.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>0</td>
      <td>23.67</td>
      <td>0.94</td>
      <td>0.087</td>
      <td>10.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>7600.0</td>
      <td>73.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>16800.0</td>
      <td>8900.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.03</td>
      <td>0.65</td>
      <td>0</td>
      <td>0.31</td>
      <td>0.76</td>
      <td>0.472</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>5500.0</td>
      <td>79.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>19200.0</td>
      <td>7987.0</td>
      <td>24.0</td>
      <td>7.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>500</td>
      <td>0.80</td>
      <td>1.00</td>
      <td>0.088</td>
      <td>15.0</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>142.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>2.48</td>
      <td>0.94</td>
      <td>0.322</td>
      <td>16.0</td>
      <td>29.0</td>
      <td>21.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>9900.0</td>
      <td>80.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>20400.0</td>
      <td>7757.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>500</td>
      <td>0.72</td>
      <td>0.63</td>
      <td>0.450</td>
      <td>17.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>8300.0</td>
      <td>85.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>13200.0</td>
      <td>9400.0</td>
      <td>31.0</td>
      <td>35.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.00</td>
      <td>0.73</td>
      <td>5000</td>
      <td>0.09</td>
      <td>0.70</td>
      <td>0.697</td>
      <td>28.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>19300.0</td>
      <td>79.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>42000.0</td>
      <td>16985.0</td>
      <td>38.0</td>
      <td>6.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.01</td>
      <td>0.79</td>
      <td>0</td>
      <td>1.24</td>
      <td>1.00</td>
      <td>0.447</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>5000.0</td>
      <td>83.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>11400.0</td>
      <td>5460.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01</td>
      <td>0.89</td>
      <td>4000</td>
      <td>0.07</td>
      <td>1.00</td>
      <td>0.233</td>
      <td>17.0</td>
      <td>32.0</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>193.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>1.13</td>
      <td>0.76</td>
      <td>0.480</td>
      <td>20.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>18400.0</td>
      <td>79.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>33600.0</td>
      <td>17633.0</td>
      <td>4.0</td>
      <td>27.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>0</td>
      <td>0.21</td>
      <td>1.00</td>
      <td>0.247</td>
      <td>13.0</td>
      <td>50.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>21300.0</td>
      <td>77.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>27600.0</td>
      <td>9900.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>300</td>
      <td>0.60</td>
      <td>0.43</td>
      <td>0.355</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>4700.0</td>
      <td>80.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>6600.0</td>
      <td>4062.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>2000</td>
      <td>0.87</td>
      <td>1.00</td>
      <td>0.377</td>
      <td>15.0</td>
      <td>62.0</td>
      <td>17.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>10100.0</td>
      <td>78.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>15600.0</td>
      <td>10275.0</td>
      <td>26.0</td>
      <td>77.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.01</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.37</td>
      <td>0.51</td>
      <td>0.486</td>
      <td>18.0</td>
      <td>21.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>10100.0</td>
      <td>79.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>20400.0</td>
      <td>8187.0</td>
      <td>12.0</td>
      <td>37.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>300</td>
      <td>0.11</td>
      <td>1.00</td>
      <td>0.229</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>18.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>8400.0</td>
      <td>74.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>3.0</td>
      <td>71.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>0</td>
      <td>0.43</td>
      <td>0.66</td>
      <td>0.369</td>
      <td>15.0</td>
      <td>72.0</td>
      <td>41.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>16500.0</td>
      <td>76.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>32400.0</td>
      <td>18600.0</td>
      <td>7.0</td>
      <td>23.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.00</td>
      <td>0.59</td>
      <td>0</td>
      <td>3.48</td>
      <td>0.92</td>
      <td>0.255</td>
      <td>19.0</td>
      <td>229.0</td>
      <td>19.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>21900.0</td>
      <td>79.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>39600.0</td>
      <td>20454.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>2400</td>
      <td>1.71</td>
      <td>0.57</td>
      <td>0.388</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>8100.0</td>
      <td>80.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>14400.0</td>
      <td>6100.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.02</td>
      <td>0.78</td>
      <td>2500</td>
      <td>1.90</td>
      <td>0.83</td>
      <td>0.336</td>
      <td>23.0</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>14000.0</td>
      <td>75.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>31200.0</td>
      <td>15500.0</td>
      <td>14.0</td>
      <td>104.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.40</td>
      <td>0.60</td>
      <td>0</td>
      <td>1.46</td>
      <td>0.69</td>
      <td>0.375</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>23.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>10400.0</td>
      <td>78.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>42600.0</td>
      <td>9841.0</td>
      <td>21.0</td>
      <td>82.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.02</td>
      <td>0.96</td>
      <td>0</td>
      <td>12.35</td>
      <td>0.96</td>
      <td>0.444</td>
      <td>20.0</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>6500.0</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8100.0</td>
      <td>7050.0</td>
      <td>10.0</td>
      <td>80.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.01</td>
      <td>0.91</td>
      <td>2000</td>
      <td>46.90</td>
      <td>0.69</td>
      <td>0.263</td>
      <td>19.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>7000.0</td>
      <td>78.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>13200.0</td>
      <td>6461.0</td>
      <td>20.0</td>
      <td>6.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>3000</td>
      <td>0.91</td>
      <td>0.93</td>
      <td>0.305</td>
      <td>21.0</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>21600.0</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>21600.0</td>
      <td>21600.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.02</td>
      <td>0.93</td>
      <td>0</td>
      <td>0.70</td>
      <td>0.53</td>
      <td>0.441</td>
      <td>25.0</td>
      <td>39.0</td>
      <td>23.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>9200.0</td>
      <td>76.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>19800.0</td>
      <td>8250.0</td>
      <td>152.0</td>
      <td>64.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>0</td>
      <td>0.38</td>
      <td>0.43</td>
      <td>0.052</td>
      <td>7.0</td>
      <td>62.0</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>18000.0</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>18000.0</td>
      <td>18000.0</td>
      <td>10.0</td>
      <td>139.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4723</th>
      <td>0.01</td>
      <td>0.96</td>
      <td>1000</td>
      <td>0.81</td>
      <td>1.00</td>
      <td>0.297</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>10200.0</td>
      <td>74.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>23400.0</td>
      <td>10880.0</td>
      <td>27.0</td>
      <td>145.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4724</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>1000</td>
      <td>1.37</td>
      <td>0.89</td>
      <td>0.287</td>
      <td>20.0</td>
      <td>31.0</td>
      <td>22.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>5000.0</td>
      <td>79.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>9000.0</td>
      <td>3460.0</td>
      <td>21.0</td>
      <td>5.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4725</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.28</td>
      <td>0.57</td>
      <td>0.137</td>
      <td>13.0</td>
      <td>34.0</td>
      <td>28.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>17200.0</td>
      <td>78.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>21600.0</td>
      <td>11850.0</td>
      <td>142.0</td>
      <td>100.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4726</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>900</td>
      <td>1.60</td>
      <td>0.90</td>
      <td>0.250</td>
      <td>12.0</td>
      <td>38.0</td>
      <td>35.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>4800.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4800.0</td>
      <td>4800.0</td>
      <td>19.0</td>
      <td>53.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4727</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>0</td>
      <td>1.33</td>
      <td>0.82</td>
      <td>0.341</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4800.0</td>
      <td>76.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>12000.0</td>
      <td>5125.0</td>
      <td>28.0</td>
      <td>57.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4728</th>
      <td>0.06</td>
      <td>0.94</td>
      <td>1500</td>
      <td>4.70</td>
      <td>1.00</td>
      <td>0.444</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>22.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>10800.0</td>
      <td>80.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>20400.0</td>
      <td>8933.0</td>
      <td>18.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4729</th>
      <td>0.17</td>
      <td>0.83</td>
      <td>0</td>
      <td>0.88</td>
      <td>1.00</td>
      <td>0.316</td>
      <td>16.0</td>
      <td>42.0</td>
      <td>27.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>6200.0</td>
      <td>80.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>16800.0</td>
      <td>7200.0</td>
      <td>3.0</td>
      <td>104.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4730</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>2000</td>
      <td>0.54</td>
      <td>0.44</td>
      <td>0.215</td>
      <td>17.0</td>
      <td>28.0</td>
      <td>32.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>15600.0</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15600.0</td>
      <td>15600.0</td>
      <td>28.0</td>
      <td>128.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4731</th>
      <td>0.00</td>
      <td>0.58</td>
      <td>2000</td>
      <td>2.63</td>
      <td>0.73</td>
      <td>0.525</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>60200.0</td>
      <td>78.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>124200.0</td>
      <td>25361.0</td>
      <td>2.0</td>
      <td>11.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4732</th>
      <td>0.11</td>
      <td>0.89</td>
      <td>4000</td>
      <td>49.37</td>
      <td>1.00</td>
      <td>0.115</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>4300.0</td>
      <td>80.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7200.0</td>
      <td>4950.0</td>
      <td>9.0</td>
      <td>250.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4733</th>
      <td>0.03</td>
      <td>0.97</td>
      <td>1000</td>
      <td>2.98</td>
      <td>0.88</td>
      <td>0.255</td>
      <td>14.0</td>
      <td>38.0</td>
      <td>18.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>9000.0</td>
      <td>81.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>13200.0</td>
      <td>6900.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4734</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0</td>
      <td>1.06</td>
      <td>0.85</td>
      <td>0.221</td>
      <td>11.0</td>
      <td>29.0</td>
      <td>57.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1500.0</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>24.0</td>
      <td>131.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4735</th>
      <td>0.01</td>
      <td>0.61</td>
      <td>5000</td>
      <td>1.15</td>
      <td>0.61</td>
      <td>0.197</td>
      <td>14.0</td>
      <td>42.0</td>
      <td>25.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>20300.0</td>
      <td>79.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>34800.0</td>
      <td>21100.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4736</th>
      <td>0.05</td>
      <td>0.95</td>
      <td>2000</td>
      <td>2.39</td>
      <td>1.00</td>
      <td>0.207</td>
      <td>12.0</td>
      <td>27.0</td>
      <td>22.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>10200.0</td>
      <td>85.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>10200.0</td>
      <td>10200.0</td>
      <td>18.0</td>
      <td>156.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4737</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>300</td>
      <td>1.00</td>
      <td>0.46</td>
      <td>0.547</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>14800.0</td>
      <td>78.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>20400.0</td>
      <td>11714.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4738</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>0</td>
      <td>6.14</td>
      <td>0.89</td>
      <td>0.133</td>
      <td>13.0</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>6800.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>7200.0</td>
      <td>5700.0</td>
      <td>48.0</td>
      <td>89.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4739</th>
      <td>0.05</td>
      <td>0.62</td>
      <td>2400</td>
      <td>0.28</td>
      <td>0.67</td>
      <td>0.403</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>3800.0</td>
      <td>72.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3900.0</td>
      <td>3600.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4740</th>
      <td>0.07</td>
      <td>0.93</td>
      <td>9100</td>
      <td>0.71</td>
      <td>1.00</td>
      <td>0.277</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>11300.0</td>
      <td>79.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>12000.0</td>
      <td>9200.0</td>
      <td>31.0</td>
      <td>163.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4741</th>
      <td>0.00</td>
      <td>0.83</td>
      <td>3000</td>
      <td>0.54</td>
      <td>1.00</td>
      <td>0.441</td>
      <td>20.0</td>
      <td>33.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>13700.0</td>
      <td>80.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>19800.0</td>
      <td>13900.0</td>
      <td>9.0</td>
      <td>140.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>0.02</td>
      <td>0.98</td>
      <td>11300</td>
      <td>0.39</td>
      <td>0.96</td>
      <td>0.525</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>9200.0</td>
      <td>77.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>21600.0</td>
      <td>7557.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0</td>
      <td>0.84</td>
      <td>1.00</td>
      <td>0.372</td>
      <td>19.0</td>
      <td>44.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1700.0</td>
      <td>72.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3600.0</td>
      <td>2300.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>0.01</td>
      <td>0.96</td>
      <td>0</td>
      <td>0.89</td>
      <td>0.45</td>
      <td>0.405</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>5200.0</td>
      <td>77.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>8400.0</td>
      <td>3344.0</td>
      <td>5.0</td>
      <td>88.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>400</td>
      <td>0.79</td>
      <td>0.67</td>
      <td>0.348</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>117.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>6800.0</td>
      <td>82.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>7200.0</td>
      <td>5850.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4747</th>
      <td>0.02</td>
      <td>0.40</td>
      <td>0</td>
      <td>1.08</td>
      <td>0.94</td>
      <td>0.394</td>
      <td>25.0</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>4500.0</td>
      <td>76.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>6000.0</td>
      <td>3260.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4748</th>
      <td>0.09</td>
      <td>0.91</td>
      <td>0</td>
      <td>0.15</td>
      <td>0.20</td>
      <td>0.159</td>
      <td>8.0</td>
      <td>36.0</td>
      <td>36.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>17300.0</td>
      <td>75.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>19800.0</td>
      <td>18300.0</td>
      <td>35.0</td>
      <td>54.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4749</th>
      <td>0.05</td>
      <td>0.68</td>
      <td>1000</td>
      <td>0.87</td>
      <td>0.94</td>
      <td>0.150</td>
      <td>16.0</td>
      <td>32.0</td>
      <td>24.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>49200.0</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49200.0</td>
      <td>49200.0</td>
      <td>44.0</td>
      <td>58.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4750</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>400</td>
      <td>0.93</td>
      <td>0.98</td>
      <td>0.605</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>10100.0</td>
      <td>78.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>19200.0</td>
      <td>7500.0</td>
      <td>17.0</td>
      <td>55.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>3000</td>
      <td>0.54</td>
      <td>0.64</td>
      <td>0.313</td>
      <td>16.0</td>
      <td>29.0</td>
      <td>25.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>7600.0</td>
      <td>79.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>21000.0</td>
      <td>7250.0</td>
      <td>18.0</td>
      <td>55.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4752</th>
      <td>0.05</td>
      <td>0.95</td>
      <td>0</td>
      <td>4.79</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>27.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>7100.0</td>
      <td>79.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>12000.0</td>
      <td>3950.0</td>
      <td>55.0</td>
      <td>16.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4753</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>1300</td>
      <td>21.35</td>
      <td>1.00</td>
      <td>0.372</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>21.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>4500.0</td>
      <td>74.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7800.0</td>
      <td>4360.0</td>
      <td>20.0</td>
      <td>54.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>4423 rows × 84 columns</p>
</div>



-------
### 5&emsp;异常值检验

这里我们采用 LOF 算法进行异常值检验。

LOF 算法参考文献：[LOF离群因子检测算法](https://zhuanlan.zhihu.com/p/37753692)

 - **构造检验函数**


```python
def localoutlierfactor(data, predict, k):
    from sklearn.neighbors import LocalOutlierFactor
    LOF = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    LOF.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = LOF.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = -LOF._decision_function(predict.iloc[:, :-1])
    return predict

def lof(data, predict=None, k=10, method=1):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers
```

 - **计算并删除异常样本**


```python
out_data, in_data = lof(data, k=10, method = 2)
data.drop(out_data.index, axis = 0)

out_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
      <th>first_transaction_time_year</th>
      <th>first_transaction_time_month</th>
      <th>k distances</th>
      <th>local outlier factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1002</th>
      <td>0.00</td>
      <td>0.71</td>
      <td>2000</td>
      <td>0.55</td>
      <td>1.00</td>
      <td>0.683</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>25200.0</td>
      <td>10840.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>4.139863e+05</td>
      <td>2.086765</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>0.00</td>
      <td>0.76</td>
      <td>3300</td>
      <td>0.02</td>
      <td>0.98</td>
      <td>0.941</td>
      <td>17.0</td>
      <td>24.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>12000.0</td>
      <td>11850.0</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>1.002926e+06</td>
      <td>2.908565</td>
    </tr>
    <tr>
      <th>2938</th>
      <td>0.01</td>
      <td>0.73</td>
      <td>1000</td>
      <td>0.36</td>
      <td>1.00</td>
      <td>0.897</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>13200.0</td>
      <td>5733.0</td>
      <td>15.0</td>
      <td>23.0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>2.383146e+06</td>
      <td>6.386036</td>
    </tr>
    <tr>
      <th>2603</th>
      <td>0.00</td>
      <td>0.28</td>
      <td>4000</td>
      <td>13.70</td>
      <td>1.00</td>
      <td>0.886</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>17400.0</td>
      <td>9000.0</td>
      <td>23.0</td>
      <td>-1.0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>1.305017e+07</td>
      <td>7.071480</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.01</td>
      <td>0.88</td>
      <td>0</td>
      <td>0.45</td>
      <td>0.78</td>
      <td>0.452</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>37200.0</td>
      <td>14016.0</td>
      <td>21.0</td>
      <td>5.0</td>
      <td>1970.0</td>
      <td>1.0</td>
      <td>1.440188e+06</td>
      <td>7.183463</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



-------
### 6&emsp;数据集切分

 - **分层抽样重构数据集**


```python
dfstatus0 = data[data.status == 0]
dfstatus1 = data[data.status == 1]
dfstatus1.sample(frac=data.status.value_counts()[1], replace=True, random_state=2018)

newdata = pd.concat([dfstatus0, dfstatus1], ignore_index=False)
newdata.sort_index(inplace=True)
newdata = newdata.reset_index(drop=True)
newdata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>trans_days_interval_filter</th>
      <th>trans_days_interval</th>
      <th>regional_mobility</th>
      <th>...</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
      <th>first_transaction_time_year</th>
      <th>first_transaction_time_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>30.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>68.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>66.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
      <td>1970.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>



 - **数据集切分**


```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(newdata, test_size=0.3, random_state=2018)
train_data.to_csv('./data/train_data.csv', index=False, header=True)
test_data.to_csv('./data/test_data.csv', index=False, header=True)
```
