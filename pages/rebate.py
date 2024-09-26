import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import streamlit as st

@st.cache_data()
def load_transform_data():
    ## Load data and transform data
    # Initialize Google sheet API call
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    creds = Credentials.from_service_account_info(st.secrets['Google_API_Credentials'], scopes=scope)
    client = gspread.authorize(creds)

    # Connect to google sheet
    sheet_id = st.secrets['sheet_id']
    sh = client.open_by_key(sheet_id)
    # values_list = sh.get_worksheet(0)

    # Extract worksheets
    wks_trans = sh.worksheet('Transactions')
    wks_cat = sh.worksheet('H_Categories')
    
    df_trans = pd.DataFrame(wks_trans.get_all_records())
    df_cat = pd.DataFrame(wks_cat.get_all_records())
    
    time = datetime.now().strftime("%b %d, %H:%M")
      
    # Transform data
    df_main = df_trans.merge(df_cat,how='left',on='cat')
    df_cat = df_cat[['cat','sub_type','type']]
    
    # Remove empty dates
    df_main = df_main[df_main['date'] != '']

    # Convert to numeric
    df_main['gross'] = pd.to_numeric(df_main['gross'], errors = 'coerce').round(2).astype(float)

    # # Convert to date
    df_main['date'] = pd.to_datetime(df_main['date'],dayfirst=True)
    # Filter out future transaction
    df_main = df_main[df_main['date'] <= pd.to_datetime(dt.date.today())]

    # Make txt lower case
    df_main[['person','shared','sas']] = df_main[['person','shared','sas']].apply(lambda x: x.astype(str).str.lower())

    # Add net amount
    df_main['net'] = np.where(df_main['shared'] =='x', df_main['gross']/2,df_main['gross']).round(2).astype(float)

    # Add payed by
    df_main['paid_by'] = df_main['person'] + df_main['shared']

    # Add year & month
    df_main['year'] = df_main['date'].dt.year
    df_main['month'] = df_main['date'].dt.month
    df_main['day_txt'] = df_main['date'].dt.strftime('%A')
    df_main['month_txt'] = df_main['date'].dt.strftime('%B')       
    df_main['date'] = df_main['date'].dt.strftime('%d-%m-%Y')
    
    return df_main, time

@st.cache_data
def time_frames(df_main):
    today = datetime.now().date()
    current_month = today.month
    past_month = 12 if current_month == 1 else current_month - 1
    current_year = today.year
    past_year = current_year-1
    past_month_year = int(current_year) -1 if current_month == 1 else current_year

    df_current = df_main[(df_main['month'] == int(current_month)) & (df_main['year'] == int(current_year))]
    df_past = df_main[(df_main['month'] == past_month) & (df_main['year'] == past_month_year)]
    
    return df_current, df_past, current_month, past_month, past_month_year, current_year, past_year

def rebate_choose_period():
    left_widget,right_widget, r = st.columns([1,1,2])
    
    selected_year = left_widget.selectbox(
        'Select year:',
        np.arange(current_year,2014,-1)
    )
    selected_month = right_widget.selectbox(
        'Select month:',
        np.arange(1,13,1),
        index=current_month - 1
    )
    return selected_year,selected_month

@st.cache_data
def rebate_metric(selected_year,selected_month):
    df_rebate = df_main[(df_main['year'] == selected_year) & (df_main['month'] == selected_month)]
    
    shared_exp_current = df_rebate[df_rebate['shared'] == 'x']['gross'].sum()
    shared_exp_current_s = df_rebate[df_rebate['paid_by'] == 'sx']['gross'].sum()
    shared_exp_current_d = df_rebate[df_rebate['paid_by'] == 'dx']['gross'].sum()
    simone_exp_current_card = df_rebate[(df_rebate['paid_by'] == 's') & (df_rebate['sas'] == 'x')]['gross'].sum()
    denise_exp_current_card = df_rebate[(df_rebate['paid_by'] == 'd') & (df_rebate['sas'] == 'x')]['gross'].sum()
    shared_balance = shared_exp_current/2 + abs(shared_exp_current_s) - abs(simone_exp_current_card)
    
    df_rebate = df_rebate[['date','paid_by','net','description','cat','sub_type']]
    
    left_page,right_page = st.columns(2)
    
    with left_page:
        st.dataframe(df_rebate,hide_index=True)
    
    with right_page:
        _ , metric_left, metric_right = st.columns([1,2,2])
        with metric_left:
            st.metric(
                label='Total spent',
                value=f'{shared_exp_current:.2f} kr'
            )
            st.metric(
                label='Denise paid',
                value=f'{shared_exp_current_d:.2f} kr'
            )
            st.metric(
                label='Simone exp with Denise card',
                value=f'{simone_exp_current_card:.2f} kr'
            )
        with metric_right:
            st.metric(
                label='Personal share',
                value=f'{(shared_exp_current / 2):.2f} kr'
            )
            st.metric(
                label='Simone paid',
                value=f'{shared_exp_current_s:.2f} kr'
            )
            st.metric(
                label='Simone owes',
                value=f'{shared_balance:.2f} kr'
            )

st.title('Monthly rebate')
df_main, time= load_transform_data()
df_current, df_past, current_month, past_month, past_month_year, current_year, past_year = time_frames(df_main)
# Calculate month range from current month
selected_year,selected_month = rebate_choose_period()
rebate_metric(selected_year,selected_month)