import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from dotenv import dotenv_values


# Set the layout mode
st.set_page_config(
    page_title="View Data",
    layout='wide',
    page_icon='',
)

#PULLING VARIABLES from an environment
environment_variables = dotenv_values('.env')
 
database = environment_variables.get("database_name")
server = environment_variables.get("server_name")
username = environment_variables.get("user")
password = environment_variables.get("password")
 
# defining a connection string for connecting to our SQL server datatbase
connection_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
 
# establish a database connection using the 'pyodbc' library
connection = pyodbc.connect(connection_string)

query = 'SELECT * FROM dbo.LP2_Telco_churn_first_3000'

data1 = pd.read_sql(query,connection)


st.title('Data from source')

st.subheader('All Data')
uploaded_file = data1


df = uploaded_file

df.fillna(method='bfill', inplace=True)

st.write(df)

st.subheader('View Data Type')
p=st.selectbox('', ['Select Data Type','Numerical','Categorical'])

n=[]
c=[]
for i in df.columns:
    if df[i].dtype=='O' or df[i].dtype=='bool':
        c+=[i]
    else:
        n+=[i]
if p=='Numerical':
    p1=st.selectbox('',['Select Numerical Column to view','All Numerical']+n)
    if p1=='All Numerical':
        st.write(df.loc[:,n])
    elif p1!='Select Numerical Column to view':
        st.write(df[p1])

elif p=='Categorical':
    p2=st.selectbox('',['Select Categorical Column to view','All Categorical']+c)
    if p2=='All Categorical':
        st.write(df.loc[:,c])
    elif p2!='Select Categorical Column to view':
        st.write(df[p2])





