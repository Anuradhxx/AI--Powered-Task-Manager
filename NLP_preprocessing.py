import nltk
import pandas as pd


nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocess_text(text):
    tokens = word_tokenize(str(text)) 
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)


df = pd.read_csv("Cleaned_data.csv")  


df['processed_description'] = df['Task Name'].apply(preprocess_text)

df.to_csv('data/preprocessed_data.csv', index=False)

print("NLP preprocessing complete! File saved as preprocessed_data.csv")