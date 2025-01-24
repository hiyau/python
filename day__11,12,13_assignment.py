# -*- coding: utf-8 -*-
"""day _11,12,13_assignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S-L0AEyj8EhV8lrMLc3qvxsnGcsGRGNM
"""

import requests
from bs4 import BeautifulSoup

def fetch_webpage_title(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the webpage content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract and return the title
        title = soup.title.string if soup.title else "No title found"
        return title
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# URL to scrape
url = 'https://example.com'

# Fetch and print the title
webpage_title = fetch_webpage_title(url)
print(f"Title of the webpage: {webpage_title}")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, output_file):
    try:
        # Create a WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Save the WordCloud to a file
        wordcloud.to_file(output_file)

        # Display the WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Text to generate WordCloud from
text = 'data science machine learning artificial intelligence'

# Output file path
output_file = 'wordcloud.png'

# Generate and save the WordCloud
generate_wordcloud(text, output_file)

import spacy

def pos_tagging(sentence):
    try:
        # Load the SpaCy model
        nlp = spacy.load('en_core_web_sm')

        # Process the sentence
        doc = nlp(sentence)

        # Extract and print the part-of-speech tags
        for token in doc:
            print(f"{token.text}: {token.pos_} ({token.tag_})")
    except Exception as e:
        print(f"An error occurred: {e}")

# Sentence to process
sentence = 'NLP is amazing and fun to learn.'

# Perform POS tagging
pos_tagging(sentence)