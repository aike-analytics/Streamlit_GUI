import streamlit as st
import pandas as pd

def history():
    st.title('Predictions History')
    
    
    # Load previous predictions dataframe
    previous_predictions = pd.read_csv('cleaned_data.csv')

    # Display previous predictions
    st.write('Cleaned Data:')
    st.dataframe(previous_predictions)
    
    # Call the data function directly
if __name__ == '__main__':
    history()