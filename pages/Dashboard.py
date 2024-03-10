import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import pyodbc
from pages.Data import df

# Set the layout mode
st.set_page_config(
    page_title="Dashboard",
    layout='wide',
    page_icon='',
)

st.write('### Which variable type do you want to visualize (Numerical or Categorical)')

vtp=st.selectbox('', ['Select Variable Type','Numerical','Categorical'])

st.set_option('deprecation.showPyplotGlobalUse', False)
    # Distribution of numerical values
df.fillna(method='bfill', inplace=True)

n=['Select Variable']
c=['Select Variable']
for i in df.columns:
    if df[i].dtype=='O' or df[i].dtype=='bool':
        c+=[i]
    else:
        n+=[i]
if vtp=='Numerical':
    vt1=st.selectbox('Variable Name', n)
    if vt1!='Select Variable':
        feature=vt1
        st.write(f'### {feature.capitalize()}')
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')
        st.pyplot()
    
elif vtp=='Categorical':
    vt2=st.selectbox('Variable Name',  c)
    if vt2!='Select Variable':
        feature=vt2
        df[feature].fillna("Unknown", inplace=True)
        plt.figure(figsize=(8, 6))
        sns.countplot(x=feature, data=df.iloc[:,1:], hue='Churn', palette='viridis')
        plt.title(f'Count plot of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Count')
        st.pyplot()

#st.subheader('Distribution of numerical features')
