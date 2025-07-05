import streamlit as st
import os
import boto3
from transformers import pipeline
import torch

s3_prefix = 'ml-models/tinybert-sentiment-analysis/'
local_path = 'tinybert-sentiment-analysis'
bucket_name = 'mlops-jjmv'

s3 = boto3.client('s3', region_name='us-west-2')

def download_dir(local_path,s3_prefix):
   
    paginator = s3.get_paginator('list_objects_v2')
    
    os.makedirs(local_path, exist_ok=True)

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for obj in result['Contents']:
                s3_key = obj['Key']
                
                # Caminho local relativo ao prefixo
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file = os.path.join(local_path, relative_path)

                # Cria diretórios se necessário
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                print(f"Baixando: {s3_key} -> {local_file}")
                s3.download_file(bucket_name, s3_key, local_file)



st.title("Machine Learning Molde Deployment at the Server!")

button = st.button("Download Moldel")

if button:
    with st.spinner("Downloading... Please wait!"):
        download_dir(local_path, s3_prefix)
text = st.text_area("Enter Your Review")

predict = st.button("Predict")



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment', device=device)


if predict:
    with st.spinner("Predicting..."):
        output = classifier(text)

        st.write(output)
        # st.info(output)