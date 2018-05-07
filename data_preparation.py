"""
Run this script first

This script reads 'dataScienceChallenge_callLog,csv' and 'dataScienceChallenge_repayment.csv' files and cleans the data
for the outliers and missing values. Then, some new features are defined and extracted from the existing features.
Finally, all data are merged to build a new and clean dataframe that is saved as a csv file.

__author__ = "Nastaran Bassamzadeh"
__email__ = "nsbassamzadeh@gmail.com"

"""

import os
import re
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

sb.set(color_codes=True)


def checking_na(df):
    """ This function finds the count and percentage of missing values in the columns of a dataframe """
    df_na_bool = pd.concat([df.isnull().any(), df.isnull().sum(), (df.isnull().sum() / df.shape[0]) * 100],
                           axis=1, keys=['df_bool', 'df_amt', 'missing_ratio_percent'])
    df_na_bool = df_na_bool.loc[df_na_bool['df_bool'] == True]
    return df_na_bool

# Create the directory 'figures' if it is does not exist
if not os.path.isdir("./figures/basic_plots"):
    os.makedirs("./figures/basic_plots")

# -----------------------------------------------------------------------------------------------------------------------

# Reading loan data

# -----------------------------------------------------------------------------------------------------------------------

loandf = pd.read_csv("./dataScienceChallenge_repayment.csv")
print(checking_na(loandf))
print('\n')
loandf.set_index('person_id_random', inplace=True)
print(loandf.describe(include='all'))
print('\n')
print(loandf.paid_first_loan.value_counts())
print('\n')
loandf.signup_date = pd.to_datetime(loandf.signup_date)
loandf.disbursement_date = pd.to_datetime(loandf.disbursement_date)
loandf['days_from_signup_to_disbursement'] = (loandf['disbursement_date'] - loandf['signup_date']).dropna().dt.days

# -----------------------------------------------------------------------------------------------------------------------

# Reading call data

# -----------------------------------------------------------------------------------------------------------------------

calldf = pd.read_csv("./dataScienceChallenge_callLog.csv")
print(checking_na(calldf))
print('\n')

calldf.phone_randomized = calldf.phone_randomized.astype(str)
calldf.contact_name_redacted = calldf.contact_name_redacted.astype(str)
calldf.direction = calldf.direction.astype('category')
calldf.local_timestamp = pd.to_datetime(calldf.local_timestamp)

print(calldf.describe(include='all'))
print('\n')
print(calldf.dtypes)
print('\n')
print(calldf.columns.tolist())
print('\n')
print("The data contains {} unique person IDs".format(calldf.person_id_random.nunique()))
print("The data contains {} observations with {} columns".format(calldf.shape[0], calldf.shape[1]))
print("There are {} unique categories in the call directions".format(calldf.direction.nunique()))
print("The duration of calls range from {} to {}".format(calldf.duration.min(), calldf.duration.max()))
print("The data are collected from {} to {}".format(calldf.local_timestamp.min(), calldf.local_timestamp.max()))

print('\n')

# -----------------------------------------------------------------------------------------------------------------------

# Treating outlier and missing values for call data

# -----------------------------------------------------------------------------------------------------------------------

# Replacing wrong contact numbers with null
r = re.compile('\d{9}')


def iscontactlegible(x):
    if r.match(x) and len(x) == 9 and str(x[0]) != '0':
        return x
    else:
        return np.nan


calldf.phone_randomized = calldf.phone_randomized.apply(iscontactlegible)

# -----------------------------------------------------------------------------------------------------------------------
# Assigning a value to missing entries in the "contact_name_redacted" column
calldf.contact_name_redacted = calldf.contact_name_redacted.apply(lambda x: 'not_in_contacts' if x == 'nan' else x)

# -----------------------------------------------------------------------------------------------------------------------
# checking the distribution of duration column
sb.distplot(calldf.duration, kde=False)
plt.savefig("./figures/basic_plots/duration_hist_with_outlier.pdf", dpi=150)
plt.clf()
calldf.loc[calldf.duration < 0, 'duration'] = 0
non_outlier_mean = calldf.duration[calldf.duration < np.percentile(calldf.duration, 95)].mean()
calldf.loc[calldf.duration > np.percentile(calldf.duration, 95), 'duration'] = non_outlier_mean
sb.distplot(calldf.duration, kde=False)
plt.savefig("./figures/basic_plots/duration_hist_no_outlier.pdf", dpi=150)
plt.clf()

# -----------------------------------------------------------------------------------------------------------------------
calldf.loc[((calldf.duration == 0) & (calldf.direction == 'incoming')), 'direction'] = 'missed'

# -----------------------------------------------------------------------------------------------------------------------
# checking the timestamp column
sb.distplot(calldf.local_timestamp.dt.year, kde=False)
plt.savefig("./figures/basic_plots/local_timestamp_year_with_outlier.pdf", dpi=150)
plt.clf()
sb.distplot(calldf.local_timestamp.dt.hour, kde=False)
plt.savefig("./figures/basic_plots/local_timestamp_hour_hist.pdf", dpi=150)
plt.clf()
sb.distplot(calldf.local_timestamp.dt.weekday, kde=False)
plt.savefig("./figures/basic_plots/local_timestamp_weekday_hist.pdf", dpi=150)
plt.clf()

calldf.loc[~calldf.local_timestamp.dt.year.isin(range(2012, 2019)), 'local_timestamp'] = np.nan
sb.distplot(calldf.local_timestamp.dt.year[~np.isnan(calldf.local_timestamp.dt.year)], kde=False)
plt.savefig("./figures/basic_plots/local_timestamp_year_no_outliers.pdf", dpi=150)
plt.clf()
print(checking_na(calldf))
print('\n')

# -----------------------------------------------------------------------------------------------------------------------

# Feature Engineering for call data

# -----------------------------------------------------------------------------------------------------------------------

# 1. Features from 'phone_randomized'

# calculating number of unique contacts each person has had interaction with
call_by_person = calldf.groupby(['person_id_random']).phone_randomized.nunique().to_frame('unique_contacts_count')

# calculating the entropy of calls for each person
contact_by_person = calldf[calldf['direction'] != 'missed'].groupby(
    ['person_id_random', 'phone_randomized']).size().to_frame('unique_phone_count').reset_index()
sum_contact_by_person = calldf[calldf['direction'] != 'missed'].groupby(['person_id_random']).size().to_frame(
    'total_count').reset_index()
merged_counts_df = pd.merge(contact_by_person, sum_contact_by_person, on='person_id_random', how='left')
merged_counts_df['entropy'] = -(merged_counts_df['unique_phone_count'] / merged_counts_df['total_count']) * np.log2(
    merged_counts_df['unique_phone_count'] / merged_counts_df['total_count'])
call_by_person['entropy'] = (merged_counts_df.groupby(['person_id_random']).entropy.sum()).round(2)

# calculating number of unique area codes each person has had interaction with
calldf['area_code'] = calldf.phone_randomized.apply(lambda x: str(x)[:3])
call_by_person['unique_areacode_count'] = calldf.groupby(['person_id_random']).area_code.nunique()


# -----------------------------------------------------------------------------------------------------------------------

# 2. Features from 'contact_name_redacted'

# Define categories for the 'contact_name_redacted' column


def typeofcontact(x):
    if x == 'not_in_contacts':
        return 'not_in_contacts'
    elif not (re.split('\S+', x)[1]):
        return 'one_word'
    else:
        return 'other_types'


calldf['type_of_contact'] = calldf.contact_name_redacted.apply(typeofcontact)
types_counts = calldf.groupby(['person_id_random', 'type_of_contact']).size().unstack()
types_counts['types_sum'] = (types_counts['not_in_contacts'].fillna(0) + types_counts['one_word'].fillna(0) +
                             types_counts['other_types'].fillna(0))
call_by_person['percent_one_word'] = ((types_counts['one_word'] / types_counts['types_sum']) * 100).round(2)
call_by_person['percent_not_in_contacts'] = ((types_counts['not_in_contacts'] / types_counts['types_sum']) * 100).round(
    2)

# -----------------------------------------------------------------------------------------------------------------------

# 3. Features from 'direction'

# calculating the percent of different call types
direction_counts = calldf.groupby(['person_id_random', 'direction']).size().unstack()
call_by_person['call_count'] = (direction_counts['incoming'].fillna(0) + direction_counts['outgoing'].fillna(0) +
                                direction_counts['missed'].fillna(0) + direction_counts['unknown'].fillna(0))
call_by_person['percent_incoming'] = ((direction_counts['incoming'] / call_by_person['call_count']) * 100).round(2)
call_by_person['percent_outgoing'] = ((direction_counts['outgoing'] / call_by_person['call_count']) * 100).round(2)
call_by_person['percent_missed'] = ((direction_counts['missed'] / call_by_person['call_count']) * 100).round(2)

# calculating 'contacts_to_interactions ratio'
call_by_person['contacts_to_interactions_ratio'] = (
    call_by_person['unique_contacts_count'] / call_by_person['call_count']).round(2)

# -----------------------------------------------------------------------------------------------------------------------

# 4. Features from 'duration'

call_by_person['call_duration_mean_minute'] = (
    calldf[calldf['duration'] != 0].groupby(['person_id_random']).duration.mean()).dropna().astype(int)
call_by_person['call_duration_std_minute'] = (
    calldf[calldf['duration'] != 0].groupby(['person_id_random']).duration.std()).dropna().astype(int)

# -----------------------------------------------------------------------------------------------------------------------

# 5. Features from 'local_timestamp'

total_duration = calldf.groupby(['person_id_random']).duration.sum()
# calculating percentage of call durations at nights
nightcall_duration = calldf[calldf['local_timestamp'].dt.hour.isin([0, 1, 2, 3, 4, 5, 6, 22, 23])].groupby(
    ['person_id_random']).duration.sum()
call_by_person['percent_nightcalls_duration'] = ((nightcall_duration / total_duration) * 100).round(2)

# calculating percentage of call durations in normal business hours in weekdays
businesshours_call_duration = calldf[
    calldf['local_timestamp'].dt.hour.isin([9, 10, 11, 12, 13, 14, 15, 16]) & calldf['local_timestamp'].dt.weekday.isin(
        [0, 1, 2, 3, 4])].groupby(['person_id_random']).duration.sum()
call_by_person['percent_businesshours_calls_duration'] = ((businesshours_call_duration / total_duration) * 100).round(2)

# calculating the mean and stdev of inter-event time for call activities
for personID, group in calldf.groupby(['person_id_random']):
    normal_times = group.loc[~(group.local_timestamp.isnull()), 'local_timestamp'].sort_values()
    shifted_times = normal_times.shift(1)  # Shift the timestamps down to calculate the time differences
    delta = (normal_times - shifted_times).iloc[1:]
    minutes_lag = delta.apply(lambda x: x.total_seconds() / 60)
    if not (pd.isnull(minutes_lag.mean()) or pd.isnull(minutes_lag.std())):
        call_by_person.loc[personID, 'inter_event_duration_mean_minute'] = int(minutes_lag.mean())
        call_by_person.loc[personID, 'inter_event_duration_std_minute'] = int(minutes_lag.std())
    else:
        call_by_person.loc[personID, 'inter_event_duration_mean_minute'] = 0
        call_by_person.loc[personID, 'inter_event_duration_std_minute'] = 0

# -----------------------------------------------------------------------------------------------------------------------

# Merge loan and call dataframes to build the final one

# -----------------------------------------------------------------------------------------------------------------------

call_loan_df = pd.merge(call_by_person, loandf[['days_from_signup_to_disbursement', 'paid_first_loan']],
                        left_index=True, right_index=True, how='left')

# The final dataframe with all the new features is stored in "call_by_person"
print(checking_na(call_loan_df))
print('\n')

call_loan_df.to_csv("./call_loan.csv")
