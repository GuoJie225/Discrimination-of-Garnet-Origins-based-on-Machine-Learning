import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

FONT_PATH = os.path.join(os.path.dirname(__file__), 'simhei.ttf')

from matplotlib import font_manager
font_manager.fontManager.addfont(FONT_PATH)
prop = font_manager.FontProperties(fname=FONT_PATH)

plt.rcParams['font.sans-serif'] = [prop.get_name()]

st.set_page_config(
    page_title="ML-based Garnet Origins Discrimination",
    layout="wide",  
    initial_sidebar_state="auto"
)

st.title(':blue[基于机器学习的石榴石成因分类 :earth_asia:]')
st.markdown('利用石榴石主量或微量元素区分石榴石不同成因（:blue[岩浆、变质或转熔]）')
st.markdown('Discriminate garnet origins (:blue[Igneous, Metamorphic or Peritectic]) with major or trace elements.')
st.caption('作者：郭杰；王浩铮；刘恒；冯林峰；张红；翟明国；李炳春')
st.caption('Author: Jie Guo, Haozheng Wang, Heng Liu, Linfeng Feng, Hong Zhang, Mingguo Zhai and Byung Choon Lee')

st.header('1. Input your data')
model = st.radio("Make predictions based on：", ["Major Elements", "Trace Elements"])

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

    
@st.cache_data
def to_template_df(model):
    output = BytesIO()
    
    input_major_excel = pd.DataFrame(columns=['Grain_no', 'Sample', 'SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Sum'])
    input_trace_excel = pd.DataFrame(columns=['Grain_no', 'Sample', 'Zr', 'Eu', 'Tb', 'Ce', '(Gd/Yb)N', 'Dy', 'Sm'])
    
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
    key="download_template_button"
)

st.divider()

st.header('2. Upload your data')

uploaded_file = st.file_uploader("Garnet composition dataset:", type=['xlsx', 'csv'], accept_multiple_files=False)

if uploaded_file is not None:
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        if uploaded_file.name.split('.')[-1] == 'xlsx':
            st.session_state.data = pd.read_excel(uploaded_file)
        else:
            st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.prediction_made = False

if st.session_state.data is not None:
    st.dataframe(st.session_state.data)

st.divider()

st.header('3. Get your results')

with open('Scaler_major_model.pkl', 'rb') as f:
    scaler_major_model = pickle.load(f)

with open('XGBoost_major_model.pkl', 'rb') as f:
    xgboost_major_model = pickle.load(f)

with open('Scaler_trace_model.pkl', 'rb') as f:
    scaler_trace_model = pickle.load(f)

with open('Adaboost_trace_model.pkl', 'rb') as f:
    Adaboost_trace_model = pickle.load(f)

def to_result_df(data, model):
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
    
    result_df = output.getvalue()
    return result_df

if st.button('Make predictions') and st.session_state.uploaded_file is not None:
    data = st.session_state.data
    data.fillna(0.001, inplace=True)

    if model == "Major Elements":
        if 'Sum' in data.columns:
            scaled_data = scaler_major_model.transform(data.query('97.50 < Sum < 102.50').iloc[:, 2:10])
            mask = (data['Sum'] > 97.50) & (data['Sum'] < 102.50)
            data.loc[mask, 'prediction'] = xgboost_major_model.predict(scaled_data)
        else:
            st.error("Data should include the 'Sum' column")
    else:
        scaled_data = scaler_trace_model.transform(data.iloc[:,2:9])
        data.loc[:, 'prediction'] = Adaboost_trace_model.predict(scaled_data)
    
    data.loc[:, 'prediction'].replace({0:'Igneous', 1:'Metamorphic', 2:'Peritectic'}, inplace=True)

    st.session_state.data = data
    st.session_state.prediction_made = True

    st.dataframe(data)

    result_df = to_result_df(data, model)
    st.download_button(
        label="Download",
        data=result_df,
        file_name=st.session_state.uploaded_file.name.split('.')[0] + "_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results_button_1"
    )

    samples = st.session_state.data['Sample'].unique()
    n = len(samples)

    ncols = 4                        
    nrows = math.ceil(n / ncols)     

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=900)
    axes = axes.flatten()            

    for i, sample in enumerate(samples):
        ax = axes[i]
        sub = st.session_state.data[st.session_state.data['Sample'] == sample]
        counts = sub['prediction'].value_counts()
        wedges, _, _ = ax.pie(counts, labels=None, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}, radius=0.75)
        ax.set_title(f'Sample: {sample}', fontsize=14)
        ax.legend(wedges, counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=12)
        ax.axis('equal')

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

elif st.session_state.uploaded_file is None:
    st.markdown(':red[Please input your data.]')
