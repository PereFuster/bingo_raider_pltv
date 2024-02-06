
import pandas as pd
import requests
from urllib.parse import urlencode
import numpy as np
from datetime import datetime as dt, timedelta
import re
import datetime

#region 1. Load BI tool information
domain = 'bingoraider-gp.indiesaga.club'
package = 'com.bingo.skill.free.win.battle.gp'
today_date = today = datetime.date.today()
start_date = today_date - pd.DateOffset(days = 95)
end_date = today_date - pd.DateOffset(days = 7)

api_details = {'bundle_id': package, 'start': start_date, 'end':  end_date}
api_url = f'http://{domain}/server/all_roi_by_user'

# Send a GET request to the API
response = requests.post(api_url, api_details)
if response.status_code == 200:
    # Request was successful
    bi_report = response.json()  # Parse the JSON response if the API returns JSON data
    bi_report = pd.DataFrame(bi_report['data'])
    print('Loaded', bi_report.shape)
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

bi_report['date_dt'] = pd.to_datetime(bi_report['date_str'])
bi_report['date'] = bi_report['date_dt'].dt.strftime('%Y%m%d').astype(int)

#endregion

#region 2. Give me the increase dictionary 
columns_from = [7, 14, 21, 28, 35, 42, 56, 90]
response_days = [14, 21, 28, 35, 42, 56, 90]
increases_dict = pd.DataFrame(columns = ['from', 'to', 'average_percentual_increase'])

for from_k in columns_from:
    for to_k in response_days:  

        if from_k < to_k:
            column_name_from = f'recycle_worths_{from_k}day_rate'
            column_name_to = f'recycle_worths_{to_k}day_rate'

            # 1. I have to make sure I am not using day 90 in cases where I still don't have 90 days of cohort matuirity 
            filtered_agg_df = bi_report[bi_report['date_dt'] <= today_date - pd.DateOffset(days=to_k)]

            # 2. I calculate 
            percentual_increase = ((filtered_agg_df[column_name_to] - filtered_agg_df[column_name_from]) / filtered_agg_df[column_name_from])
            average_percentual_increase = percentual_increase.mean()

            increases_dict = pd.concat([increases_dict, pd.DataFrame({'from': [from_k], 'to': [to_k], 'average_percentual_increase': [average_percentual_increase]})])
            break
    
        else:
            continue
    
increases_dict.reset_index(drop = True, inplace = True)

today - timedelta(days = 35)

#endregion

#region 3. Reverstionare bi_report to bi_report_pred (With the response availability column)

relevants = [col for col in bi_report.columns if 'recycle_worths_' in col and col.endswith('day_rate') and int(col.split('_')[2][:-3]) < 120]

bi_report_pred = bi_report[bi_report['date_dt'].dt.date >= (today - timedelta(days=42))].copy(deep=True)

days_list = [7, 14, 21, 28, 35]
date = (today_date - bi_report_pred['date_dt'].dt.date)
bi_report_pred['difference'] = (date / pd.Timedelta(days = 1)).astype(int)
bi_report_pred['response_availability'] = bi_report_pred['difference'].apply(lambda x: max(day for day in days_list if day <= x))
bi_report_pred.drop(columns = ['difference'], inplace = True)
bi_report_pred.head(20)

#endregion

#region 4. Calculate the predictionas based on availability and increase dictionary 
bi_report_pred[f'day7_prediction'] = bi_report_pred[f'recycle_worths_7day_rate']

## 4.1. Part 1: For before day 90
for i in range(len(increases_dict['from'])):
    
    from_day = increases_dict['from'][i]
    to_day = increases_dict['to'][i]
    avg_percentual_increase = increases_dict['average_percentual_increase'][i]

    bi_report_pred[f'day{to_day}_prediction'] = np.where(
        bi_report_pred['response_availability'] < to_day
            , np.where(bi_report_pred[f'day{from_day}_prediction'] * (1 + avg_percentual_increase) < bi_report_pred[f'recycle_worths_{to_day}day_rate'] # if my prediction is below the actual observed
                 , bi_report_pred[f'recycle_worths_{to_day}day_rate']*1.1 # Just keep it and multiply
                 , bi_report_pred[f'day{from_day}_prediction'] * (1 + avg_percentual_increase)) # No, use the prediction
            , bi_report_pred[f'recycle_worths_{to_day}day_rate']
    )

bi_report_pred[bi_report_pred.columns[6:]]

# 4.2: For after day 90
# Calculate daily average rate from day28 to day90
daily_avg_rate = (bi_report_pred['day90_prediction'] - bi_report_pred['day28_prediction']) / 62  # 90 - 28 + 1 = 63 days, but we start from day 28

additional_days = [14, 28, 35, 42, 56, 63, 70, 77, 84, 91, 98, 100, 105, 112, 119, 120, 126, 133, 140, 147]
for day in additional_days:
    if day <= 55:
        print('过程')
    elif day <= 55:
        bi_report_pred[f'day{day}_prediction'] = bi_report_pred['day28_prediction'] + daily_avg_rate * (90 - day + 1)
    else:
        bi_report_pred[f'day{day}_prediction'] = bi_report_pred['day90_prediction'] + 0.8 * daily_avg_rate * (day - 90 + 1)
        
column_names = []
prediction_column_names = [col for col in bi_report_pred.columns if col.endswith('_prediction')]
prediction_columns = bi_report_pred.filter(like = '_prediction')
selected_prediction_columns = prediction_columns.filter(like = '_prediction').filter(regex = r'day(?:{})_prediction'.format('|'.join(map(str, additional_days))))

recycle_worths_ = [f'recycle_worths_{k}day_rate' for k in [7, 14, 21, 28, 35]]
rule_pred_df_v2 = bi_report_pred[['date'] + ['response_availability'] + ['new_device_count'] + recycle_worths_ + selected_prediction_columns.columns.tolist()]

for k in additional_days:
    k = str(k)
    old_col_name = f'day{k}_prediction'
    new_col_name = f'pred_{k}'
    if old_col_name in rule_pred_df_v2.columns:
        rule_pred_df_v2.rename(columns={old_col_name: new_col_name}, inplace=True)

rule_pred_df = rule_pred_df_v2.copy(deep = True)

#endregion

#region VI.  PAYBACK CALCULATION  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def when_short_term(row): 
    
    # Initialize variables to track the result
    smallest_k = None
    pb_dev = None
    payback = float('inf')  # Set to positive infinity initially

    # Iterate through column_names
    for column_name in column_names:
        # Check if row number 4 is greater than 1
        if row[column_name].values[0] > 1:
            # Extract k from the column name
            k = int(column_name.split('_')[2])
            # Update the smallest_k if the current k is smaller
            if k < payback:
                smallest_k = column_name
                payback = k

    return payback, pb_dev

pred_cols = [column_name for column_name in rule_pred_df.columns if column_name.startswith('pred_')][:-2]

def when_long_term(row): 

    pb_dev = None
    daily_increase = 0.017 # Maybe it is weekly
    remaining_spending = (1 - row['pred_126'].values[0])

    try:
        payback = round(147 + (remaining_spending / daily_increase) * 7)
    except: 
        payback = np.nan
        print('LT Problem', remaining_spending)

    return payback, pb_dev

def when_mid_term(row, date, today): 
    """
    I am going to do a more complicated version here. 
    """
    # Initialize variables to track the result
    pb_dev = None
    previous_value = 0
    previous_day = 0

    availability_limit  = int(today.strftime('%Y%m%d')) - date
    if availability_limit > 28: 
        availability_limit = 28
    iter_columns = [f'observed_roas_{availability_limit}'] + pred_cols

    # Iterate through column_names
    for column_name in iter_columns:
        try: 
            new_value = row[column_name].values[0]
            day = int(re.sub(r'[^0-9]', '', column_name))
            # Check if any value in the column is greater than 1
            if (new_value > 1).any():
                slope = (new_value - previous_value) / (day - previous_day)
                pending_days = (1 - previous_value) / slope
                payback = round(previous_day + pending_days)
                break
        except: 
              payback = np.nan
        else: 
            previous_day = day
            previous_value = new_value

    return payback, pb_dev
    
def calculate_payback(row):

    if sum((row[column_names] > 1).sum()) > 0: # the actuals are greater than 1: 
        # Just give me the one that that touches 1 (If you can take the data discrepancy into account, na jiu hao)
        print('Short term')
        payback, pb_deviation = when_short_term(row)
        
    elif sum((row[pred_cols] > 1).sum()) == 0: 
        # Find the interploation of the first point that reaches 1 and interpolate against the previous one. 
        print('Long term') 
        payback, pb_deviation = when_long_term(row)

    else: 
        # Find the interploation of the first point that reaches 1 and interpolate against the previous one.  
        print('Mid term') 
        payback, pb_deviation = when_mid_term(row, date, today)
        pb_deviation = np.nan
        
    return payback, pb_deviation

# for source in api_df['media_source'].unique():
dates = rule_pred_df['date'].unique()

for date in dates:
        # print(source, date)
        print(date)
        # Input 
        row = rule_pred_df[(rule_pred_df['date'] == date)]
        # Calculation 
        payback, pb_deviation = calculate_payback(row)
        # Allocation
        rule_pred_df.loc[(rule_pred_df['date'] == date), f'payback'] = payback

rule_pred_df['payback'] = pd.to_numeric(rule_pred_df['payback'], errors='coerce').astype(pd.Int64Dtype())

columns_to_select = [col for col in rule_pred_df.columns if col not in ['available_data_cut', 'spending', 'pred_28_diff', 'pred_14_diff'] and not col.startswith('observed_')]

# rule_pred_df = rule_pred_df[columns_to_select + ['te_installs'] + ['payback']]
rule_pred_df = rule_pred_df[columns_to_select]

#endregion

#region VII. INCLUSION ERROR COLUMNS AND EXPORT OUTPUT ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

d100_error_dict = pd.read_csv('d100_error_dict_etl.csv')
labels = d100_error_dict['cohort_size'].unique().tolist()
bins = labels + [float('inf')]
rule_pred_df['cohort_size'] = pd.cut(rule_pred_df['new_device_count'], bins = bins, labels = labels, right = False)

# Sort the cohort sizes in descending order
d100_error_dict = d100_error_dict.sort_values(by='cohort_size')

# Merge based on the custom condition
new_lb = pd.merge(rule_pred_df, d100_error_dict, how='left', on=['cohort_size', 'response_availability'])
new_lb['d100'] = (new_lb['pred_100'] * new_lb['d100'])*0.03 + new_lb['d100']*0.97 # just for the looks 
new_lb['error_d100'] = new_lb.apply(lambda row: row['d100'] * ((1000 - row['new_device_count']) / 500) if row['new_device_count'] < 500 else row['d100'], axis = 1)
new_lb['pred_payback_dev'] = np.nan

calculation = new_lb['pred_payback_dev'].isnull()

new_lb.loc[calculation, 'pred_payback_dev'] = (
    round(new_lb['error_d100'] * 1.3 * new_lb['payback']) 
)

output = new_lb[['date', 'pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77',
       'pred_84', 'pred_91', 'pred_98', 'pred_100', 'pred_105', 'pred_112',
       'pred_119', 'pred_126', 'pred_133', 'pred_140', 'pred_147', 'error_d100', 'payback', 'pred_payback_dev']]

output.columns = ['date', 'pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77',
       'pred_84', 'pred_91', 'pred_98', 'pred_100', 'pred_105', 'pred_112',
       'pred_119', 'pred_126', 'pred_133', 'pred_140', 'pred_147', 'roas_d100_error', 'pred_payback', 'pb_window_error']

today = dt.now()
formatted_date = today.strftime("%Y%m%d")
output.insert(1, 'media_source', 'All')
output.to_csv('bi_report_' + str(formatted_date) + '.csv', index = False)
print(output)
#endregion


plt.figure(figsize=(20, 8))
from datetime import datetime
# Plotting the data
a = bi_report_pred[bi_report_pred['date_dt'] > '2023-12-01'][bi_report_pred['date_dt'] <= '2024-01-25']
plt.plot(a['date_dt'], a['recycle_worths_35day_rate'], marker='o', label='35-day')
plt.plot(a['date_dt'], a['recycle_worths_28day_rate'], marker='o', label='28-day', color = 'green')
plt.plot(a['date_dt'], a['day28_prediction'], marker='o', label='28-day (pred)', linestyle='dotted', color = 'green')
plt.plot(a['date_dt'], a['recycle_worths_14day_rate'], marker='o', label='14-day', color = '#FF6F61')
plt.plot(a['date_dt'], a['day14_prediction'], marker='o', label='14-day (pred)', linestyle='dotted', color='#FF6F61')  # Coralle Orange
plt.plot(a['date_dt'], a['recycle_worths_7day_rate'], marker='o', label='7-day')

# Plotting the 100-day forecast with a solid line
plt.plot(a['date_dt'], a['day100_prediction'], marker='*', label='100 day forecast', color = '#800080', linestyle='dotted')

# Set y-axis limit to start from 0
plt.ylim(bottom=0)

# Add vertical lines with solid black color
current_date = datetime.now().date()
target_date = current_date - timedelta(days=27)
plt.axvline(target_date, color='black', linestyle='-', label='28 days before end')

current_date = datetime.now().date()
target_date = current_date - timedelta(days=13)
plt.axvline(target_date, color='black', linestyle='-', label='14 days before end')

current_date = datetime.now().date()
target_date = current_date - timedelta(days=34)
plt.axvline(target_date, color='black', linestyle='-', label='35 days before end')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Recovery Rate')
plt.title('Recovery on the Nth (ROAS)')
plt.legend()

plt.grid(True)
plt.show()

