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

import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import json
import re
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from uk_stemmer import UkStemmer

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)

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

def owns_by(user, text):
    
    """returns class identifier of the message
    0 - mention java, spring, typescript, мальчики, орунькаю
    1 - rust, аніме, ахахахаха, сед, кричу
    2 - docker, kuber, microservice, докер, кубер, eбо
    """
    java = ["java", "spring", "typescript", "мальчики", "орунькаю"]
    rust = ["rust", "раст" "аніме", "аха", "сед", "кричу", "ex"]
    docker = ["docker", "kuber", "microservice", "докер", "кубер", "eбо"]
    if any(substring in text for substring in java): return 0
    if any(substring in text for substring in rust): return 1
    if any(substring in text for substring in docker): return 2
    
    if 'Gri' in user: return 0
    if 'Ши' in user: return 1
    if 'Ni' in user: return 2

import nltk

def preprocess_data():
    file_path = './data/processed-messages.json'

    with open(file_path, encoding='UTF-8') as data_file:
        data = json.load(data_file)
        comp_df = pd.DataFrame(data['data'])

    comp_df = comp_df[["id", "type", "from", "text", "media_type", "photo", "date"]].fillna("")

    # Apply the transformations and fill NaN values with 0
    comp_df["owns_by"] = comp_df.apply(lambda x: owns_by(x["from"], x["text"]), axis=1)
    comp_df["message_class"] = comp_df.apply(lambda x: get_message_class(x["media_type"], x["photo"]), axis=1)

    # Fill NaN values with 0 in specific columns
    columns_to_fill = ["id", "type", "owns_by", "text", "message_class"]
    comp_df[columns_to_fill] = comp_df[columns_to_fill].fillna(0)
    comp_df['text'] = comp_df['text'].astype(str)
    return comp_df

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

stop_words = ["я", "ты", "мы", "а", "і", "у", "там", "як", "как", "то", "не", "в", "что", "ну", "це", "що", "там", "для", "с", "да", "но", "по", "на", "и", "так"]

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
    print ('Найбільш уживані токени: ', common_tokens)
    # words.plot (most_common, cumulative = True)
    words_df = pd.DataFrame(list(common_tokens), columns=["ngram", "count"])
    # words_df = words_df.sort_values(by="count", ascending=False)
    return words_df

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
                      
def main():

    
    comp_df = preprocess_data()

    classifiers=[nltk.NaiveBayesClassifier]
    y_column = 'owns_by'
    for classifier in classifiers:
        for n in (1,3):
            print ('Класифікатор -',classifier)
            print ('Порядок n -',n)           
            print ('Класифікатор за колонкою -',y_column) 
            model=nltk_classifiers(comp_df,X_column='text',y_column=y_column,classifier=classifier, n=n)
            if classifier==nltk.NaiveBayesClassifier:
                print ('Найважливіші токени для класифікації за колонкою -',y_column)
                model.show_most_informative_features(10) 

    



if __name__ == "__main__":
    main()
