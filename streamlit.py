import streamlit as st
from tempfile import NamedTemporaryFile
from pathlib import Path
import os
import shutil

from model.result import Result
from extract.feature_extraction_classification import ExtractorClassification


#extractor = ExtractorClassification(r'../etc/config.yaml')
extractor = ExtractorClassification(r'C:/Users/Mimnat/Documents/DAS/etc/config.yaml')

st.title('Detection anomaly sound')

with st.sidebar.form(key='Form1'):
    uploaded_file = st.file_uploader("Choose an audio file")
    submitted1 = st.form_submit_button(label='Load')

machine = st.selectbox("machine selected", extractor.machines)
if machine:
    id_m = st.selectbox("ID machine selected", extractor.id_machines[machine])

if submitted1 or uploaded_file:
    if uploaded_file:
        st.header('Sound')
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format='audio/wav')

        st.header('Prediction')
        result = Result(machine, id_m, extractor)
        suffix = Path(uploaded_file.name).suffix
        try:
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(uploaded_file, tmp)

                out = result.predict(tmp.name)
                if out == 0:
                    st.text(f"Audio '{uploaded_file.name}' sounds normal for machine {machine}"
                            f" {id_m}.")
                else:
                    st.text(f"Audio '{uploaded_file.name}' sounds anormal for machine {machine} "
                            f"{id_m}.")
        finally:
            tmp.close()
            os.unlink(tmp.name)
