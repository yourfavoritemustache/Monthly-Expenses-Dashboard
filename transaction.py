
import gspread
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from google.oauth2.service_account import Credentials
from datetime import datetime
from streamlit_plotly_events import plotly_events
from typing import Dict, Set

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

def initialize_state():
    ## Loops through the different session states to initialize it, 
    ## it adds a suffix to the variables that stores the fig click event
    ## The counter is used to later reset the filter callbacks
    for q in ['type']:
        if f"{q}_query" not in st.session_state:
            st.session_state[f"{q}_query"] = set()

    if "counter" not in st.session_state:
        st.session_state.counter = 0

def reset_state_callback():
    ## Resets all filters and increments counter in Streamlit Session State
    st.session_state.counter = 1 + st.session_state.counter
    for q in ['type']:
        st.session_state[f"{q}_query"] = set()

def query_data(df):
    ## Apply filters in Streamlit Session State to filter the input DataFrame
    df["selected"] = True
    
    for q in ['type']:
        if st.session_state[f"{q}_query"]:
            df.loc[~df[q].isin(st.session_state[f"{q}_query"]), "selected"] = False

    return df[df['selected']==True]

@st.cache_data
def time_frames(df):
    ## Create current & past timeframes
    today = datetime.now().date()
    current_month = today.month
    past_month = 12 if current_month == 1 else current_month - 1
    current_year = today.year
    past_year = current_year-1
    past_month_year = int(current_year) -1 if current_month == 1 else current_year

    df_current = df[(df['month'] == current_month) & (df['year'] == current_year)]
    df_past = df[(df['month'] == past_month) & (df['year'] == past_month_year)]
    
    return df_current, df_past, current_month, past_month, past_month_year, current_year, past_year

def select_variables_and_filter(df):
    ## Stores the input to later filter the df accordingly
    year,month,person,r = st.columns([1,1,1,2])
    
    selected_year = year.selectbox(
        'Select year:',
        np.arange(datetime.now().year,2014,-1)
    )
    selected_month = month.selectbox(
        'Select month:',
        np.arange(1,13,1),
        index=datetime.now().month - 1
    )
    selected_person = person.selectbox(
        'Select person:',
        ['Denise','Simone'],
        index=1
    )

    if selected_person == 'Simone':
        person = ['s','sx','dx']
    else:
        person = ['d','dx','sx']
    
    df_filtered = df[(df['year'] == selected_year) & 
                     (df['month'] == selected_month) &
                     (df['paid_by'].isin(person))]
    return df_filtered

def build_wat_fig(df):
    ## Transform the df for the fig creation
    df = df.groupby('type',as_index=False)['net'].sum()
    df = df[df['type']!='Savings']
    # Add total row
    total_row = {
        'type':'Final balance',
        'net':df['net'].sum()
    }
    # Add new row to df
    df_merged = pd.concat([
        df,
        pd.DataFrame([total_row])
    ])
    # Plot chart
    wat_fig = px.bar(
        df_merged,
        x='type',
        y='net',
        color='type',
        color_discrete_map={ #https://coolors.co/264653-2a9d8f-e9c46a-f4a261-e76f51
            'Income':'#2A9D8F',
            'Expenses':'#E76F51',
            'Investing':'#F4A261',
            'Final balance':'#264653'
            },
        text_auto=True,
    )
    wat_fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return wat_fig

def build_cat_fig(df):
    df_current, df_past, current_month, past_month, past_month_year, current_year, past_year = time_frames(df)
    
    df = df[(df['type'] != 'Income') & (df['type'] != 'Savings')]
    df = df.groupby('cat',as_index=False)['net'].sum()
    df['net'] = df['net'].abs()
    df = df.sort_values('net',ascending=True)
    
    cat_fig = px.bar(
        df,
        x='net',
        y='cat',
        orientation='h',
        title='Expenses break-down',
        text_auto=True,
        labels={
            'net':'Amount',
            'cat':'Category'
        },
        height=600,
        # color='net'
    )
    cat_fig.update_traces(
        marker_color='#264653',
        # textposition="outside",
    )
    cat_fig.update_layout(
        xaxis_tickprefix='kr ',
        xaxis_tickformat=',.0f',
        yaxis_title=None
    )
    return cat_fig

def render_plotly_ui(df):
    ## This function 'print' the fig & df and place them in the ui
    ## Additionally collects the fig inputs and extract them into session state variables
    wat_fig = build_wat_fig(df)
    cat_fig = build_cat_fig(df)
    
    l,r = st.columns(2)
    with l:
        wat_fig_selected = plotly_events(
        wat_fig,
        click_event=True,
        key=f'type_{st.session_state.counter}'
        )
        st.plotly_chart(cat_fig)
        current_query = {}
        current_query['type_query'] = {el['x'] for el in wat_fig_selected}
    with r:
        st.dataframe(
            df[['date','net','description','cat','sub_type','type']],
            hide_index=True,
            # width=800
            )

    return current_query

def update_state(current_query: Dict[str, Set]):
    ## The function is designed to manage and update the state of a Streamlit application based on incoming query parameters.
    ## The function expects a dictionary where keys are strings and values are sets
    rerun = False
    for q in ['type']:
        # Check whether the fig selection is different compared to the initial state
        if current_query[f"{q}_query"] - st.session_state[f"{q}_query"]:
            # Set the session state equal to the current fig selection
            st.session_state[f"{q}_query"] = current_query[f"{q}_query"]
            # Used to trigger a rerun
            rerun = True
    if rerun:
        st.rerun() 

def main():
    # This load and transform the data @cached
    df_main = load_data()
    st.title('Monthly budget dashboard')
    # This filter df based on widget inputs
    df_filtered_period = select_variables_and_filter(df_main)
    # Thanks to callbacks and session state this will update df according to fig selection
    df_filtered = query_data(df_filtered_period)
    # Place fig and df in the ui
    current_query = render_plotly_ui(df_filtered)
    # Update the session state according to fig selection and triggers rerun to feed the query_data function
    update_state(current_query)
    st.button("Reset filters", on_click=reset_state_callback)

if __name__ == '__main__':
    st.set_page_config(layout='wide')
    initialize_state()
    main()




## fix selected timeframe and adjust it to return the timeframe based on the selection adn split the filtering