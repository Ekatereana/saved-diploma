# ========== (c) JP Hwang 2020-03-17  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import json
import re
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from uk_stemmer import UkStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objs as go

nltk.download('punkt')

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)
author_dict = {}
AUTHOR_COUNDER = 0
CHAT_FILE_PATH = './data/processed-messages-v1.json'
JAVA_KPI_FILE_PATH = './data/processed-messages-kpi-java.json'


def load_stopwords():
    with open('sources/stopwords_ua.txt', encoding='utf-8') as file:
        stopwords_ua = file.read().splitlines()
    with open('sources/stopwords_ru.txt', encoding='utf-8') as file:
        stopwords_ru = file.read().splitlines()
    return set(stopwords_ua).union(set(stopwords_ru))


stop_words = load_stopwords()

# DATA PREPROCESSING

def get_media_mapping(media_type):
    if media_type == "sticker" :return 1
    elif media_type == "annimation" : return 2
    elif media_type == "video_message": return 3
    if media_type == "voice_message" : return 4
    elif media_type == "video_file" : return 4
    else: return 1000

def get_message_class(media_type, photo_attached):
    """returns class identifier of the message
    0 - plain message, only text
    1 - sticker
    2 - annimation
    3 - video_message
    4 - voice_message
    5 - video_file
    6 - photo
    1000 - other
    """
    if bool(photo_attached): return 6
    if not bool(media_type): return 0
    return get_media_mapping(media_type)

def owns_by(user):
    
    """returns author class identifier of the message
    """
    global AUTHOR_COUNDER
    if user not in author_dict: 
        author_dict[user] = AUTHOR_COUNDER
        AUTHOR_COUNDER = AUTHOR_COUNDER + 1
    
    return author_dict[user] 

def preprocess_data(file_path):

    with open(file_path, encoding='UTF-8') as data_file:
        data = json.load(data_file)
        comp_df = pd.DataFrame(data['data'])

    comp_df = comp_df[["id", "type", "from", "text", "media_type", "photo", "date"]].fillna("")

    # Apply the transformations and fill NaN values with 0
    comp_df["owns_by"] = comp_df.apply(lambda x: owns_by(x["from"]), axis=1)
    comp_df["message_class"] = comp_df.apply(lambda x: get_message_class(x["media_type"], x["photo"]), axis=1)

    # Fill NaN values with 0 in specific columns
    columns_to_fill = ["id", "type", "owns_by", "text", "message_class"]
    comp_df[columns_to_fill] = comp_df[columns_to_fill].fillna(0)
    comp_df['text'] = comp_df['text'].astype(str)
    return comp_df

# BAG OF WORDS AND CLASSIFICATION

def ua_tokenizer(text,ua_stemmer=True,stop_words=[]):
    """ Tokenizer for Ukrainian language, returns only alphabetic tokens. 
    
    Keyword arguments:
    text -- text for tokenize 
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
        
    """
    tokenized_list=[]
    text=re.sub(r"""['’"`�]""", '', text)
    text=re.sub(r"""[A-z]""", ' ', text)
    text=re.sub(r"""([0-9])([\u0400-\u04FF]|[A-z])""", ' ', text)
    text=re.sub(r"""([\u0400-\u04FF]|[A-z])([0-9])""", ' ', text)
    text=re.sub(r"""[\-.,:+*/_]""", ' ', text)
    for word in nltk.word_tokenize(text): 
        if word.isalpha():
            word=word.lower() 
            if ua_stemmer is True:      
                word=UkStemmer().stem_word(word)
            if word not in stop_words:
                tokenized_list.append(word) 
    return tokenized_list


def ngrams_info(series,n=1,most_common=50,ua_stemmer=True,stop_words=stop_words):
    """ ngrams_info - Show detailed information about string pandas.Series column. 
    
    Keyword arguments:
    series -- pandas.Series object
    most_common -- show most common words(default 50)
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
        
    """
    print (n,'- grams')
    print ('ua_stemmer:',ua_stemmer)
    words= series.astype(str).str.cat(sep=' ')
    print ('Кількість символів: ',len(words))
    words=nltk.ngrams(ua_tokenizer(words,ua_stemmer=ua_stemmer,stop_words=stop_words),n)
    words=nltk.FreqDist(words)
    common_tokens = words.most_common(most_common)
    print ('Кількість токенів: ',words.N())
    print ('Кількість унікальних токенів: ',words.B())
    # print ('Найбільш уживані токени: ', common_tokens)
    # words.plot (most_common, cumulative = True)
    words_df = pd.DataFrame(list(common_tokens), columns=["ngram", "count"])
    # words_df = words_df.sort_values(by="count", ascending=False)
    return words_df

def build_ngrams_per_author(pre_df, author, n):


    # Call ngrams_info for each unique 'owns_by' value
    
    author_df = pre_df[pre_df['owns_by'] == author]
    ngrams_df = ngrams_info(author_df['text'], n=n)
     
    
    return ngrams_df

def bag_of_words(document_tokens,word_features):
        """ Return the dict of bag_of_words. 

        Keyword arguments:
        document_tokens -- list of tokens
        word_features -- list of features

        """
        
        features={}
        for word in word_features:
            
            features['contains({})'.format(word)]=(word[0] in document_tokens)
        
        return features

def label_features(dataframe, X_column,y_column, stop_words=stop_words,ua_stemmer=True,most_common=1000, n=1):
    words=dataframe[X_column].str.cat(sep=' ')
    words=nltk.ngrams(ua_tokenizer(words,ua_stemmer=ua_stemmer,stop_words=stop_words),n=n)
    words=nltk.FreqDist(words)
    word_features=words.most_common(most_common)
    word_features=[words[0] for words in word_features]
    
    labeled_featuresets=[]
    for _,row in dataframe.iterrows():
        
        row[X_column]=nltk.ngrams(ua_tokenizer(row[X_column],ua_stemmer=ua_stemmer,stop_words=stop_words),n=n)
        row[X_column]=[words[0] for words in nltk.FreqDist(row[X_column])]        
        labeled_featuresets.append((bag_of_words(row[X_column],word_features=word_features), row[y_column]))  
    return labeled_featuresets

def nltk_classifiers(dataframe,X_column,y_column,classifier=nltk.NaiveBayesClassifier,n=1,stop_words=stop_words,ua_stemmer=True,most_common=1000): 
    
        
    labeled_featuresets = label_features(dataframe,X_column,y_column)
    train_set,test_set,_,_=train_test_split(labeled_featuresets,dataframe[y_column],stratify=dataframe[y_column],test_size=0.33)
    
    
    if classifier==nltk.MaxentClassifier:
        classifier=classifier.train(train_set, max_iter=5)
    else:
        classifier=classifier.train(train_set)         
    accuracy_train=nltk.classify.accuracy(classifier, train_set)
    accuracy=nltk.classify.accuracy(classifier, test_set)
    print('Точність класифікатора на навчальних даних:',accuracy_train)
    print('Точність класифікатора на тестових даних:',accuracy)
    y_true=[]
    y_pred=[]
    for test in test_set:
        y_true.append(test[1])
        y_pred.append(classifier.classify(test[0]))
    confmat=nltk.ConfusionMatrix(y_pred,y_true)
    print(confmat)
    return classifier       
                      
# CLASTERISATION

def wrap_on_ua_tokenizer(text):
    return ua_tokenizer(text,ua_stemmer=True,stop_words=stop_words)

ua_tone_dict = pd.read_csv('https://raw.githubusercontent.com/lang-uk/tone-dict-uk/master/tone-dict-uk.tsv', delimiter='\t', names=['word', 'score'], index_col=0)
ru_tone_dict = pd.read_csv('https://raw.githubusercontent.com/text-machine-lab/sentimental/master/sentimental/word_list/russian.csv', delimiter=',', index_col=0)
union_tone_dict = pd.concat([ua_tone_dict, ru_tone_dict])
stemmed_index = union_tone_dict.index.map(UkStemmer().stem_word)
union_tone_dict = pd.DataFrame( union_tone_dict.values, index = stemmed_index )

def calculate_sentiment(words):
    tone_words = [word for word in words if word in union_tone_dict.index]
    if len(words) == 0: 
        return 0
    return sum([union_tone_dict.loc[word][0] if word in union_tone_dict.index else 0 for word in words])/len(words)

def get_topic_words(vectorizer, svd, n_top_words):
    words = vectorizer.get_feature_names_out()
    topics = []
    for component in svd.components_:
        top_words_idx = np.argsort(component)[::-1][:n_top_words]
        top_words = [words[i] for i in top_words_idx]
        topics.append(top_words)
    return topics

def get_topic_pretty_print(topics):
    topic_list = []
    for i, topic in enumerate(topics):
        topic_list.append(f'Topic {i}: {", ".join(topic)}\n')
        print(f'Topic {i}: {", ".join(topic)}\n')
    return topic_list

#  DATA TESTING

def classification_test(comp_df):
    classifiers=[nltk.NaiveBayesClassifier,nltk.MaxentClassifier,nltk.DecisionTreeClassifier]
    y_column = 'owns_by'
    for classifier in classifiers:
        for n in (1,5):
            print ('Класифікатор -',classifier)
            print ('Порядок n -',n)           
            print ('Класифікатор за колонкою -',y_column) 
            model=nltk_classifiers(comp_df,X_column='text',y_column=y_column,classifier=classifier, n=n)
            if classifier==nltk.NaiveBayesClassifier:
                print ('Найважливіші токени для класифікації за колонкою -',y_column)
                model.show_most_informative_features(10) 

def extract_topics(comp_df, components, topic_features):
    comp_df['tokens'] = comp_df.text.apply(wrap_on_ua_tokenizer)
    comp_df['tone'] = comp_df.tokens.apply(calculate_sentiment)
    comp_df['clean_text'] = comp_df.tokens.str.join(' ')

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(comp_df.clean_text)
    svd_vectorizer = TruncatedSVD(n_components=components, random_state=42)
    X_lsa = svd_vectorizer.fit_transform(X)

    topics = get_topic_words(tfidf_vectorizer, svd_vectorizer, topic_features)
    comp_df['topic'] = X_lsa.argmax(axis=1)
    return comp_df, get_topic_pretty_print(topics)


def main():

    comp_df = preprocess_data(CHAT_FILE_PATH)
    classification_test(comp_df)

    # extract_topics(comp_df)

    
if __name__ == "__main__":
    main()
