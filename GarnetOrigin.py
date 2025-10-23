import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.title(':blue[Discrimination of Garnet Origins based on Machine Learning :earth_asia:]')
st.markdown('This web is designed to discriminate different origins (:blue[Igneous, Metamorphic and Peritectic]) of garnet by major and trace elements.')
st.caption('Author: Guo Jie and Wang Haozheng, from Southwest Petroleum University')

image = Image.open('Garnet_Image.jpg')
st.image(image, caption='Garnets in Gneiss')

st.header('1. Input your data')
model = st.radio("Make predictions based on：", ["Major Elements", "Trace Elements"])

global data
@st.cache_data
def to_template_df(model):
    output = BytesIO()
    
    input_major_excel = pd.DataFrame(columns=['Grain_no', 'Sample', 'SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Sum'])
    input_trace_excel = pd.DataFrame(columns=['Grain_no', 'Sample', 'Y', 'Zr', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Lu'])
    
    if model == "Major Elements":
        df = input_major_excel
    else:
        df = input_trace_excel
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    template_df = output.getvalue()
    return template_df

Template_excel = to_template_df(model)

st.download_button(
    label = "Download Template",
    data = Template_excel,
    file_name = model + "_Input_Template.xlsx",
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()

st.header('2. Upload your data')

uploaded_file = st.file_uploader("Garnet composition dataset:", type=['xlsx', 'csv', 'xls'], accept_multiple_files=False)

if uploaded_file is not None:
    if uploaded_file.name.split('.')[-1] == 'xlsx':
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    st.dataframe(data)

st.divider()

st.header('3. Get your results')

with open('Scaler_major_model.pkl', 'rb') as f:
    scaler_major_model = pickle.load(f)

with open('XGBoost_major_model.pkl', 'rb') as f:
    xgboost_major_model = pickle.load(f)

with open('Scaler_trace_model.pkl', 'rb') as f:
    scaler_trace_model = pickle.load(f)

with open('XGBoost_trace_model.pkl', 'rb') as f:
    xgboost_trace_model = pickle.load(f)

@st.cache_data
def to_result_df(model):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')

    result_df = output.getvalue()
    return result_df

if st.button('Make predictions') and uploaded_file is not None:
    data.fillna(0.001, inplace=True)
    if model == "Major Elements":
        scaled_data = scaler_major_model.transform(data.query('97.50 < Sum < 102.50').iloc[:,2:-1])
        mask = (data['Sum'] > 97.50) & (data['Sum'] < 102.50)
        data.loc[mask, 'prediction'] = xgboost_major_model.predict(scaled_data)
    else:
        scaled_data = scaler_trace_model.transform(np.log1p(data.iloc[:,2:]))
        data['prediction'] = xgboost_trace_model.predict(scaled_data)

    data['prediction'].replace({0:'Igneous', 1:'Metamorphic', 2:'Peritectic'}, inplace=True)

    result_df = to_result_df(model)

    st.dataframe(data)

    st.download_button(
        label = "Download",
        data = result_df,
        file_name = uploaded_file.name + "_Results.xlsx",
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    fig = plt.figure(dpi=600)
    sns.countplot(data=data, x='prediction', hue='Sample')
    plt.title('Distribution of Predictions')
    st.pyplot(fig)
else:
    st.text('Please input your data.')


