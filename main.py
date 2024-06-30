import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def extract_data(url):
        try:
                print('extracting')
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                regex = re.compile('.*comment.*')
                results = soup.find_all('p',{'class':regex})
                reviews = [result.text for result in results]
                df =  pd.DataFrame(np.array(reviews), columns=['reviews'])
                return df
        except Exception as e:
               st.error(f"Error extracting data: {e}")
               return None

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# token = tokenizer.encode('hello this is a good day', return_tensors='pt')
# result = model(token)
# print(result.logits)

def sentiment_analysis(review):
        try:
                max_length = tokenizer.model_max_length
                token = tokenizer.encode(review, max_length=max_length, truncation=True, return_tensors='pt')
                result = model(token)
                return int(torch.argmax(result.logits))+1
        except Exception as e:
                st.error(f"Error performing sentiment analysis: {e}")
                return None


#Streamlit Code
st.title("Yelp Comment Sentiment Analysis")


if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""


url = st.text_input("Enter the URL:", key="input_text")

def clear_input():
        print('clear url')
        st.session_state["input_text"] = ""


if url:
    st.write(f"The entered URL is: {url}")
    # data = extract_data('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
    st.write("Generating Sentiments...")

    data =  extract_data(url)
        # print(data['reviews'])
    if not data.empty:
        print(data)
        review1 = data['reviews']
        # emoji = ['🤢','🥴','🙃','🙂','😍']
        list_data= []
        for review in review1:
                rev_senti = sentiment_analysis(review)
                # emoji_senti = emoji[rev_senti - 1]
                list_data.append({ "review": review, "sentiment": rev_senti })    
                # print(list_data)
                
        df = pd.DataFrame(list_data)
        
        st.write("User Sentiment Rating:", df )
        clear_input()