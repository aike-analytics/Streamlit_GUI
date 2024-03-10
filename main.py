import streamlit as st
from PIL import Image


# Set the layout mode
st.set_page_config(
     layout='wide',
     initial_sidebar_state='auto'
     )

#use home for navigation link
st.title("ChurnPulse App", anchor="main") 
st.subheader("Stay ahead of customer churn with real-time predictive analytics.")


# Open the image
img = Image.open("logo.jpg")
st.image(img, width=None, use_column_width=True)

try:
  img = Image.open("logo.jpg")
except FileNotFoundError:
  # Display a placeholder or error message
  st.write("Logo image not found.")
else:
  # Display the image if found
  st.image(img)





def home():
       st.title('Customer Churn Prediction - See Tomorrow Today, Drive Business Forward!')
    
       st.write('This data-driven App Offers predictive insights to help businesses retain customers effectively, ensuring sustained growth and profitability.')
    
    

    
    



if __name__ == '__main__':
    home()

# Adding Login details
    
# Define the correct username and password
correct_username = "your_username"
correct_password = "your_password"

# Collect username and password inputs from the user
username_input = st.text_input('Username')
password_input = st.text_input('Password', type='password')

# Check if the login button is clicked
if st.button('Login'):
    # Check if the provided username and password match the correct ones
    if username_input == correct_username and password_input == correct_password:
        st.success('Logged in as {}'.format(username_input))
    else:
        st.error('Invalid username or password. Please try again.')


# Add links to home page
st.write('Links:')
st.markdown('[GitHub](https://github.com/dashboard)')
st.markdown('[LinkedIn](www.linkedin.com/in/isaac-fumey-27b75774)')
    