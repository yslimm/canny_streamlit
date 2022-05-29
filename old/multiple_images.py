# https://cafe-mickey.com/python/streamlit-5/

import streamlit as st
from PIL import Image

st.title('Multiple Images')

#Layout

col1, col2, col3 = st.columns(3)
print(col1)

with col1:
    st.header('image 1')
    st.image('images/img (1).jpeg')

with col2:
    st.header('image 2')
    st.image('images/img (2).jpeg')

with col2:
    st.header('image 3')
    st.image('images/img (3).jpeg')


