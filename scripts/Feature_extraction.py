import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer
import pickle
import os


nltk.download('punkt')


df = pd.read_csv("Cleaned_data.csv") 
df['Task Name'] = df['Task Name'].fillna("")


print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Task Name'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())


df_tfidf = pd.concat([df, tfidf_df], axis=1)
df_tfidf.to_csv("output_tfidf_features.csv", index=False)
print("TF-IDF saved to output_tfidf_features.csv")


print("Training Word2Vec...")
df['tokens'] = df['Task Name'].apply(word_tokenize)
w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1)

def get_avg_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return [0]*model.vector_size

df['w2v_vector'] = df['tokens'].apply(lambda x: get_avg_vector(x, w2v_model))
w2v_df = pd.DataFrame(df['w2v_vector'].tolist(), columns=[f"w2v_{i}" for i in range(w2v_model.vector_size)])


df_w2v = pd.concat([df.drop(columns=['w2v_vector']), w2v_df], axis=1)
df_w2v.to_csv("output_word2vec_features.csv", index=False)
print("Word2Vec saved to output_word2vec_features.csv")


print("Generating BERT embeddings...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = bert_model.encode(df['Task Name'].tolist())
bert_df = pd.DataFrame(bert_embeddings, columns=[f"bert_{i}" for i in range(bert_embeddings.shape[1])])


df_bert = pd.concat([df, bert_df], axis=1)
df_bert.to_csv("output_bert_features.csv", index=False)
print(" BERT saved to output_bert_features.csv")



os.makedirs("features", exist_ok=True)


with open("features/features_tfidf.pkl", "wb") as f:
    pickle.dump((tfidf_matrix, df['Task Status']), f)


with open("features/features_word2vec.pkl", "wb") as f:
    pickle.dump((w2v_df, df['Task Status']), f)


with open("features/features_bert.pkl", "wb") as f:
    pickle.dump((bert_embeddings, df['Task Status']), f)

print("All feature sets saved in 'features/' folder as .pkl files.")
