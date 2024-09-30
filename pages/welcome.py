import streamlit as st

st.header('Welcome')

### URLs ###
url_streamlit = 'https://docs.streamlit.io/'
url_github = 'https://github.com/yourfavoritemustache/Monthly-Expenses-Dashboard'
url_image = 'https://raw.githubusercontent.com/yourfavoritemustache/Monthly-Expenses-Dashboard/refs/heads/main/assets/me_coding.jpg'
### Text ###
st.markdown('''Welcome to my monthly personal dashboard, I have created this website using the incredible [streamlit library](%s).'''%url_streamlit)
st.markdown('''For more technical information visit my [Github repo](%s).'''%url_github)
st.markdown('''In order to access the dashboard open the side menu (by clicking the arrow in the top left corner) and login using :red[guest] for username and password.''')
st.markdown('''The dataset contains fake numbers and transactions and is therefore not representative of my personal finances. However the metrics and graph are the same as the ones I use to asses my financial overview''')
st.caption('Below a picture of me writing some app code during summer holiday :laughing:')
st.image(url_image, width=350)