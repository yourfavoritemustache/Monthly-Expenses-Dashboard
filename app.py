import streamlit as st
import streamlit_authenticator as stauth

def pag_layout(username,authentication_status):
    guest_dashboard = st.Page(
        "pages/transaction_guest.py",
        title="Guest dashboard",
        icon=":material/dashboard:",
    )
    rebate = st.Page(
        "pages/rebate.py",
        title="Rebate",
        icon=":material/person_add:",
    )
    dashboard = st.Page(
        "pages/transactions.py",
        title="Transactions",
        icon=":material/person_add:",
    )
    welcome = st.Page(
        "pages/welcome.py",
        title="Welcome",
        icon=":material/person_add:",
    )

    private_pages = [rebate, dashboard]
    guest_pages = [guest_dashboard]
    public_pages = [welcome]

    page_dict = {}
    if authentication_status:
        if username == None:
            page_dict["Public"] = public_pages
        elif username == 'guest':
            page_dict["Guest"] = guest_pages
        else:
            page_dict['Private'] = private_pages
    else:
        page_dict["Public"] = public_pages
        
    pg = st.navigation(page_dict)
    pg.run()

def hash_passwords(passwords):
    ## Function to hash passwords
    return stauth.Hasher(passwords).generate()

def login():
    # User credentials
    names = ['Astengoli', 'Guest']
    usernames = st.secrets['usernames']
    passwords = st.secrets['passwords']
    
    # Hash passwords
    hashed_passwords = hash_passwords(passwords)

    # Create credentials dictionary
    credentials = {
        'usernames': {
            usernames[0]: {'name': names[0], 'password': hashed_passwords[0]},
            usernames[1]: {'name': names[1], 'password': hashed_passwords[1]}
        }
    }
    # Create authenticator object
    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name='expenses_dashboard',
        cookie_key = 'yourfavoritesignaturek3y',
        cookie_expiry_days=1
    )
    # Implement login widget
    name, authentication_status, username = authenticator.login('sidebar')
    
    return authenticator, name, authentication_status, username

if __name__ == '__main__':
    st.set_page_config(layout='wide')
    authenticator, name, authentication_status, username = login()
    # Handle authentication status
    if authentication_status == None:
        with st.sidebar:
            st.write('For Guest access use "guest" for both username and passwords')
        pag_layout(username,authentication_status)
    elif authentication_status == False:
        with st.sidebar:
            st.write('For Guest access use "guest" for both username and passwords')
            st.error('Username or password is incorrect')
        pag_layout(username,authentication_status)
    else:
        pag_layout(username,authentication_status)
        authenticator.logout('Log out','sidebar')
