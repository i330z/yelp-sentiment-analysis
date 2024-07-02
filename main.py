import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from urllib.parse import urlparse

@st.cache_resource()
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def extract_data(url):
    try:
        with st.spinner('Extracting Data...'):
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            regex = re.compile('.*comment.*')
            results = soup.find_all('p', {'class': regex})
            reviews = [result.text for result in results]
            df = pd.DataFrame(np.array(reviews), columns=['reviews'])
            return df
    except requests.RequestException as e:
         st.error(f"Error fetching {e}")
    except Exception as e:
        st.error(f"Error extracting data: {e}")
    return None


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        'nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained(
        'nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model


tokenizer, model = load_model()
# token = tokenizer.encode('hello this is a good day', return_tensors='pt')
# result = model(token)
# print(result.logits)


def sentiment_analysis(reviews):
    try:
        max_length = tokenizer.model_max_length
        tokens = tokenizer(reviews, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        results = model(**tokens)
        return [int(torch.argmax(logits))+1 for logits in results.logits]
        # result = model(token)
        # return int(torch.argmax(result.logits))+1
    except Exception as e:
        st.error(f"Error performing sentiment analysis: {e}")
        return None


# Streamlit Code
st.title("Yelp Comment Sentiment Analysis")


if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""


url = st.text_input("Enter the URL:", key="input_text")


if url:
    if not is_valid_url(url):
        st.error('Enter a valid URL:')
    else:
        st.write(f"The entered URL is:{url}")
        data = extract_data(url)
        if data is not None and not data.empty:
          st.write("generating sentiments...")
          review1 = data['reviews'].tolist()
          print(type(review1))
          sentiments = sentiment_analysis(review1)
          if sentiments:
              emoji = ['🤢','🥴','🙃','🙂','😍']
              list_data = [{"review": review, "sentiment": sentiment, "emoji": emoji[sentiment-1]} for review, sentiment in zip(review1, sentiments)]
              df = pd.DataFrame(list_data)
              st.write("User Sentiment Rating:")
              st.dataframe(df)
          else:
              st.error("Failed to generate sentiments")
        else:
            st.error("No data to extract from the URL")



    		# data = extract_data('https://www.yelp.com/biz/social-brew-cafe-pyrmont')


			
			# print(data['reviews'])
			# if not data.empty:
			# 		print(data)
			# 		review1 = data['reviews']
			# 		# emoji = ['🤢','🥴','🙃','🙂','😍']
			# 		list_data = []
			# 		for review in review1:
			# 				rev_senti = sentiment_analysis(review)
			# 				# emoji_senti = emoji[rev_senti - 1]
			# 				list_data.append({"review": review, "sentiment": rev_senti})
			# 				# print(list_data)

			# 		df = pd.DataFrame(list_data)

			# 		st.write("User Sentiment Rating:", df)
