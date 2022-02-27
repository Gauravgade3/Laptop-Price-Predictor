import streamlit as st
import pickle
import numpy as np


hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


# Set title
st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand',df['Manufacturer'].unique())

# type of laptop
type = st.selectbox('Type',df['Category'].unique())



# OS and weight
col1, col2 = st.columns(2)
with col1:
    os = st.selectbox('Operating System', df['Operating System'].unique())

with col2:
    weight = st.number_input('Weight of the Laptop')


# ram and gpu
col3, col4 = st.columns(2)
with col3:
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

with col4:
    gpu = st.selectbox('GPU', df['Gpu'].unique())




# touchscreen and ips display
col5, col6 = st.columns(2)
with col5:
    touchscreen = st.checkbox("Touchscreen")
with col6:
    ips = st.checkbox("IPS Display")


cpu = st.selectbox('CPU', df['Cpu'].unique())


# hdd and sdd
col7, col8 = st.columns(2)
with col7:
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
with col8:
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])




# screen size and resolution
col7, col8 = st.columns(2)
with col7:
    screen_size = st.selectbox('Screen Size (in Inches)', [10.1, 11.3, 11.6, 12.0, 12.3, 12.5, 13.0, 13.5, 13.9, 14.0, 14.1, 15.0, 15.4, 15.6, 17.0, 17.3, 18.4])
with col8:
    resolution = st.selectbox('Screen Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                               '2560x1440', '2304x1440'])




# predict button
if st.button("Predict Price"):
    ppi=None
    if touchscreen:
        touchscreen=1
    else:
        touchscreen=0
    if ips:
        ips=1
    else:
        ips=0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company,type,os,weight,ram,gpu,touchscreen,ips,cpu,hdd,ssd,ppi])
    query = query.reshape(1,12)

    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))





