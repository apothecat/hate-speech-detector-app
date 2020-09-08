import streamlit as st
import pandas as pd
import joblib
# from PIL import Image

#Loading final trained  model 
# "rb" mode opens the file in binary format for reading
model= open("hsmod.pkl", "rb")
hsmod_clf=joblib.load(model)
st.title("Hate Speech Detection App")

user_input = st.text_area("Paste text here")


