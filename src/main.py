import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
nltk_path = '../.venv/lib/nltk_data'
nltk.download('punkt_tab', nltk_path)
nltk.download('averaged_perceptron_tagger_eng', nltk_path)
nltk.download('maxent_ne_chunker_tab', nltk_path)
nltk.download('words', nltk_path)
nltk.download('vader_lexicon', nltk_path)

def polarity_scores_roberta(text):
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2]),
    }
    return scores_dict

def polarity_scores(df):
    sia = SentimentIntensityAnalyzer()
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            statement = row['statement']
            id = row['id']
            vader_result = sia.polarity_scores(statement)
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value

            roberta_result = polarity_scores_roberta(statement)
            both = {**vader_result_rename, **roberta_result}
            res[id] = both
        except RuntimeError:
            print(f'Falha com a entrada de id {id}')
        except Exception as e:
            print(f'Falha desconhecida com o id {id}: {e}')

    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'id'})
    results_df = results_df.merge(df, how='left')
    return results_df


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    try:
        train_df = pd.read_csv('../data/train.csv')
        test_df = pd.read_csv('../data/test.csv')
    except FileNotFoundError:
        print("Make sure 'train.csv' and 'test.csv' are in the same directory.")
        exit()

    train_df['statement'].fillna('', inplace=True)
    test_df['statement'].fillna('', inplace=True)

    train_df['cleaned_statement'] = train_df['statement'].apply(clean_text)
    test_df['cleaned_statement'] = test_df['statement'].apply(clean_text)

    label_encoder = LabelEncoder()
    train_df['status_encoded'] = label_encoder.fit_transform(train_df['status'])

    tfidf_vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')

    X_train = tfidf_vectorizer.fit_transform(train_df['cleaned_statement'])
    X_test = tfidf_vectorizer.transform(test_df['cleaned_statement'])
    y_train = train_df['status_encoded']

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    predictions_encoded = model.predict(X_test)

    predictions = label_encoder.inverse_transform(predictions_encoded)

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'status': predictions
    })

    submission_df.to_csv('../data/submission.csv', index=False)