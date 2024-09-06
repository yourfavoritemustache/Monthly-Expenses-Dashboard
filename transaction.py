
import gspread
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import plotly.express as px

from google.oauth2.service_account import Credentials
from datetime import datetime
from streamlit_plotly_events import plotly_events
from typing import Dict, Set
from millify import millify

@st.cache_data()
def load_data():
    ## Load data & apply basics transformations
    # Google sheet API
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file('credentials.json', scopes=scopes)
    client = gspread.authorize(creds)

    # Connect to google sheet
    sheet_id = "1MwslTvC_v5DHJuXnglELvWNnc7JrE65oP1_uVTaslb4"
    sh = client.open_by_key(sheet_id)
    values_list = sh.get_worksheet(0)

    # Extract worksheets
    wks_trans = sh.worksheet('Transactions')
    df_trans = pd.DataFrame(wks_trans.get_all_records())
    wks_cat = sh.worksheet('H_Categories')
    df_cat = pd.DataFrame(wks_cat.get_all_records())
    df_cat = df_cat[['cat','sub_type','type']]
    df_main = df_trans.merge(df_cat,how='left',on='cat')
    
    # Remove empty dates
    df_main = df_main[df_main['date'] != '']

    # Convert to numeric
    df_main['gross'] = pd.to_numeric(df_main['gross'], errors = 'coerce').astype(int)

    # Convert to date
    df_main['date'] = pd.to_datetime(df_main['date'],dayfirst=True)
    # Filter out future transaction
    df_main = df_main[df_main['date'] <= pd.to_datetime(dt.date.today())]

    # Make txt lower case
    df_main[['person','shared','sas']] = df_main[['person','shared','sas']].apply(lambda x: x.astype(str).str.lower())

    # Add net amount
    df_main['net'] = np.where(df_main['shared'] =='x', df_main['gross']/2,df_main['gross'])

    # Add payed by
    df_main['paid_by'] = df_main['person'] + df_main['shared']

    # Add year & month
    df_main['year'] = df_main['date'].dt.year
    df_main['month'] = df_main['date'].dt.month
    df_main['day_txt'] = df_main['date'].dt.strftime('%A')
    df_main['month_txt'] = df_main['date'].dt.strftime('%B')       
    df_main['date'] = df_main['date'].dt.strftime('%d-%m-%Y')
    
    return df_main

@st.cache_data()
def last_updated():
    time = datetime.now().strftime("%b %d, %H:%M")
    return time

def reload_data():
    with st.sidebar:
        if st.button("Reload Data",type="primary",key='reload_data'):
            load_data.clear()
            load_data()
            mess = last_updated()
            st.write(f'Last updated on: {mess}')

def tag_list(df):
    ## Create a list of tags ordered by date
    tags = df[['tag','date']]
    tags.loc[:,'date'] = pd.to_datetime(tags['date'],dayfirst=True)
    tags = tags.groupby('tag',as_index=False)['date'].max().sort_values(by='date',ascending=False)
    tags = list(tags.tag)
    return tags

def current_period(df,y,m,p,t):
    df_current = df[(df['year'].isin(y)) & 
                    (df['month'].isin(m)) &
                    (df['paid_by'].isin(p)) &
                    (df['tag'].isin(t))] 
    return df_current
    
def past_period(df,y,m,p):
    # there are 3 options:
    # a. More then one year is selected
    # b. Only one month is selected
    # c. All the months in a year are selected
    
    if len(y) > 1:
        df_past = df
    elif len(m) == 1:
        past_month = 12 if m[0] == 1 else m[0] - 1
        past_year = y[0] -1 if m[0] == 1 else y[0]
        
        df_past = df[(df['month'] == past_month) & 
                     (df['year'] == past_year) & 
                     (df['paid_by'].isin(p))
                    ]
    else:
        past_year = y[0] -1
        # Get list of current period months and applies it to past period
        past_month = df[df['month'].isin(m)]['month'].unique().tolist()
        
        df_past = df[(df['month'].isin(past_month)) & 
                     (df['year'] == past_year) & 
                     (df['paid_by'].isin(p))
                    ]
    
    return df_past

def select_variables(df):
    ## Stores the inputs
    year,month,person,tag,r = st.columns([1,1,1,1,2])
    
    selected_year = year.selectbox(
        'Select year:',
        ['Select All',*np.arange(datetime.now().year,2014,-1)],
        index=1
    )
    selected_month = month.selectbox(
        'Select month:',
       ['Select All',*range(1,13,1)],
        index=(datetime.now().month)
    )
    selected_person = person.selectbox(
        'Select person:',
        ['Denise','Simone'],
        index=1
    )
    tags = tag_list(df)
    selected_tag = tag.selectbox(
        'Select tag:',
        tags,
        index=0
    )
    # Series to if statement to factor in the "select all" option
    if selected_year == 'Select All':
        years = [*np.arange(datetime.now().year,2013,-1)]
    else:
        years = [selected_year]
    #
    if selected_month == 'Select All':
        months = [*range(1,13,1)]
    else:
        months = [selected_month]
    #
    if selected_person == 'Simone':
        person = ['s','sx','dx']
    else:
        person = ['d','dx','sx']
    #   
    if selected_tag == '':
        tags_list = tag_list(df)
    else:
        tags_list = [selected_tag]
    
    return years, months, person, tags_list

def period_card(df_current, df_past):
    df_card_current = df_current.groupby('type',as_index=False)['net'].sum()
    df_card_current['net'] = df_card_current['net'].abs()
    
    df_card_past = df_past.groupby('type',as_index=False)['net'].sum()
    df_card_past['net'] = df_card_past['net'].abs()
    
    df_card = df_card_current.merge(df_card_past,on='type',how='outer',suffixes=['_current','_past'])
    df_card = df_card.fillna(0)
    df_card['diff'] = df_card.net_current - df_card.net_past
    
    # Income exceptions    
    try:
        income = int(df_card.loc[df_card['type']=='Income']['net_current'].iloc[0])
    except (ValueError, TypeError, IndexError):
        income = 0
    # Expenses exceptions 
    try:
        expenses = int(df_card.loc[df_card['type']=='Expenses']['net_current'].iloc[0])
    except (ValueError, TypeError, IndexError):
        expenses = 0
    # Savings exceptions 
    try:
        savings = int(df_card.loc[df_card['type']=='Savings']['net_current'].iloc[0])
    except (ValueError, TypeError, IndexError):
        savings = 0
    # Savings delta
    try:
        savings_delta = int(df_card.loc[df_card['type']=='Savings']['diff'].iloc[0])
    except (ValueError, TypeError, IndexError):
        savings_delta = 0
    # Investing exceptions 
    try:
        investing = int(df_card.loc[df_card['type']=='Investing']['net_current'].iloc[0])
    except (ValueError, TypeError, IndexError):
        investing = 0
    # Investing delta
    try:
        investing_delta = int(df_card.loc[df_card['type']=='Investing']['diff'].iloc[0])
    except (ValueError, TypeError, IndexError):
        investing_delta = 0
    
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric(
            label='Income',
            value=millify(income,precision=1,drop_nulls=True),
            delta=millify(
                int(df_card.loc[df_card['type']=='Income']['diff'].iloc[0]),
                precision=1,
                drop_nulls=True)
        )
    with col3:  
        st.metric(
            label='Savings',
            value=millify(savings,precision=1,drop_nulls=True),
            delta=millify(savings_delta,precision=1,drop_nulls=True)
        )
    with col2:
        st.metric(
            label='Expenses',
            value=millify(expenses,precision=1,drop_nulls=True),
            delta=millify(
                int(df_card.loc[df_card['type']=='Expenses']['diff'].iloc[0]),
                precision=1,
                drop_nulls=True)
            )
    with col4:
        st.metric(
            label='Investing',
            value=millify(investing,precision=1,drop_nulls=True),
            delta=millify(investing_delta,precision=1,drop_nulls=True)
            )

def build_cat_fig(df_current,df_past,df_avg):
    df_current = df_current[(df_current['type'] != 'Income') & (df_current['type'] != 'Savings')]
    df_current = df_current.groupby('cat',as_index=False)['net'].sum()
    df_current['net'] = df_current['net'].abs() 
    df_current = df_current.sort_values('net',ascending=True)
    
    df_past = df_past.groupby('cat',as_index=False)['net'].sum()
    df_past['net'] = df_past['net'].abs()
    
    df = df_current.merge(
                df_past,
                how='left', 
                on='cat',
                suffixes=('_current','_past')
                ).merge(
                    df_avg,
                    how='left',
                    on='cat'
                ).fillna(0)
                
    df['diff'] = (df['net_current']-df['net_past'])
    
    chart_height = df.shape[0]*80
    
    cat_fig = px.bar(
        df,
        x=['average','net_current'],
        y='cat',
        orientation='h',
        title=f'Expenses break-down: [add here the current period]',
        text='diff',
        hover_name='cat',
        # hover_data={
        #     'net_current':False,
        #     'cat':False
        # },
        text_auto=True,
        labels={
            'net_current':'Amount',
            'cat':'Category',
            'diff':'Diff vs prev period'
        },
        height=chart_height,
        color_discrete_map={'net_current':'#264653','average':'#969696'}
    )    
    cat_fig.update_layout(
        xaxis_tickprefix='kr ',
        xaxis_tickformat=',.0f',
        yaxis_title=None,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#264653",
        ),
        barmode='group'
    )
    cat_fig.update_traces(
        hovertemplate = '<i>Diff vs prev period<i>: kr %{text}<extra></extra>'
    )
    
    return cat_fig

def build_sav_rate(df,df_main,p):
    df['date'] = pd.to_datetime(df['date'],dayfirst=True)
    df_main['date'] = pd.to_datetime(df_main['date'],dayfirst=True)
    
    if (len(df.year.unique().tolist()) > 1) | (len(df.month.unique().tolist()) > 1):
        min_date = df.date.min()
        max_date = df.date.max()
        df_filtered = df.loc[(df['date'] >= min_date) &
                             (df['date'] <= max_date)]
    else:
        max_date = df.date.max()
        min_date = (max_date - dt.timedelta(days=365)).replace(day=1)
        df_filtered = df_main.loc[(df_main['date'] >= min_date) &
                                  (df_main['date'] <= max_date) &
                                  (df_main['paid_by'].isin(p))
                                  ]
    
    sav_rate = df_filtered[(df_filtered['cat'] != 'Savings') & (df_filtered['type'] != 'Investing')]
    sav_rate = sav_rate.pivot_table(index=['year','month'],values='net',columns='type',aggfunc='sum',fill_value=0)
    sav_rate['diff'] = sav_rate.Income - sav_rate.Expenses.apply(lambda x: abs(x))
    sav_rate['diff_%'] = np.where(
                                    sav_rate.Income == 0,
                                    0,
                                    np.where(
                                        sav_rate.Expenses/sav_rate.Income < 0,
                                        (sav_rate.Expenses/sav_rate.Income + 1)*100,
                                        (sav_rate.Expenses/sav_rate.Income - 1)*100
                                        )
                                )    
    sav_rate = sav_rate.reset_index()
    sav_rate['day'] = 1
    sav_rate['date'] = pd.to_datetime(sav_rate[['year', 'month','day']])
    average_perc = sav_rate['diff_%'].mean()
    average = sav_rate['diff'].mean()

    sav_rate_fig = px.line(
                        sav_rate,
                        x='date',
                        y='diff_%',
                        line_shape='spline',
                        markers=True,
                        text='diff_%',
                        custom_data='diff',
                        labels={
                            'diff':'Cash EoM'},
                        hover_data={
                            'date':False,
                            'diff_%':False,
                            'diff':True
                        },
                        title='Cash EoM',
                        height=400,
                        width=600
                    )
    sav_rate_fig.update_xaxes(
                        dtick="M1",
                        tickformat="%b\n%Y"
                    )
    sav_rate_fig.update_traces(
        line= dict(
            color='#264653'
        ),
        texttemplate='%{text:.1f}%',
        textposition="top center",
        hovertemplate='Cash EoM:</b> %{customdata[0]:,.0f}kr<extra></extra>',
    )
    sav_rate_fig.update_layout(
        xaxis_title=None,
        yaxis_title='% of saving',
        yaxis=dict(
            showgrid=False,
            )
        )
    sav_rate_fig.add_hline(
        y=average_perc,
        line_color= '#264653',
        line_width=1,
        line_dash='dot',
        annotation_text=f'12M avg: {average:.0f}kr', 
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="grey"
        )
    return sav_rate_fig

def categorize_avg(row):
    if row['cat'] in ['Authorities','Bar&cocktails','Car','Crypto','Entertainment','Furniture&home','Groceries','Hotels','Income','Misc','Present','Housing&bills']:
        return row['median']
    elif row['cat'] in ['Café&restaurant','Cloathing&Beauty','Flights','Health','Investing','Rent','SKAT','Salary','Savings','Sport','Subscriptions','Transportation']:
        return row['mean']
    else:
        return 0

def mean_monthly_values(df_main,y,m,p):
    df = df_main[
        (df_main['year'] == (pd.to_datetime(dt.date.today()).year-1)) & ##### if condition per considerare ultimi 6 mesi?
        (df_main['person'].isin(p))
        ]
    df = df.groupby(['month','cat'],as_index=False)['net'].sum()
    df = df.groupby('cat',as_index=False).agg(
        sum=('net','sum'),
        mean=('net','mean'),
        median=('net','median')
    )    
    df['average'] = df.apply(categorize_avg, axis=1).abs()
    
    df = df[['cat','average']]
    
    return df

def render_ui(df,y,m,p,t):
    df_current = current_period(df,y,m,p,t)
    df_past = past_period(df,y,m,p)
    df_avg = mean_monthly_values(df,y,m,p)
    cat_fig= build_cat_fig(df_current,df_past,df_avg)
    sav_rate_fig = build_sav_rate(df_current,df,p)   
    df_current['date'] = pd.to_datetime(df_current['date']).dt.strftime('%m/%d/%Y')
    
    col1,col2 = st.columns(2)
    with col1:
        period_card(df_current,df_past)
        with st.expander('Click to see transaction\'s list'):
            st.dataframe(df_current[['date','net','description','cat','sub_type','type']],hide_index=True)

        st.plotly_chart(sav_rate_fig,use_container_width=True)
    with col2:
        st.plotly_chart(cat_fig)
    
def main():
    # This load and transform the data @cached
    df_main = load_data()
    
    # This reload the google sheet data
    reload_data()
    
    st.title('Monthly budget dashboard')
    
    # This filter df based on widget inputs
    years, months, person, tags_list = select_variables(df_main)
    # with st.expander('Click to expand'):
    #     a,b,c,d = st.columns(4)
    #     with a:
    #         years, 
    #     with b:
    #         months,
    #     with c:
    #         person, 
    #     with d:
    #         tags_list
    
    render_ui(df_main, years, months, person, tags_list)    

if __name__ == '__main__':
    st.set_page_config(layout='wide')
    main()