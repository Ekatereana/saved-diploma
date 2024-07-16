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
import re
from sklearn.model_selection import train_test_split
from uk_stemmer import UkStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
col_pal = px.colors.sequential.Pinkyl

pd.options.plotting.backend = "plotly"
nltk.download('punkt')

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)
author_dict = {}
CHAT_FILE_PATH = './data/processed-messages-private-chat.json'
JAVA_KPI_FILE_PATH = './data/processed-messages-kpi-java.json'
NLP_UKRAINE_FILE_PATH = './data/processed-messages-nlp-community-chat.json'


def load_stopwords():
    with open('sources/stopwords_ua.txt', encoding='utf-8') as file:
        stopwords_ua = file.read().splitlines()
    with open('sources/stopwords_ru.txt', encoding='utf-8') as file:
        stopwords_ru = file.read().splitlines()
    return set(stopwords_ua).union(set(stopwords_ru))

STOP_WORDS = load_stopwords()


# BAG OF WORDS AND CLASSIFICATION

LOL_pattern = r'(?![ax]+$).*$'


def ua_tokenizer(text,ua_stemmer=True,stop_words=STOP_WORDS):
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
            if word not in stop_words and re.match(LOL_pattern, word): 
                if ua_stemmer is True:      
                    word=UkStemmer().stem_word(word)
                tokenized_list.append(word) 
    return tokenized_list

def build_ngram_freq(series, n = 1, ua_stemmer = True, stop_words = STOP_WORDS):
    print (n,'- grams')
    print ('ua_stemmer:',ua_stemmer)
    words= series.astype(str).str.cat(sep=' ')
    print ('Кількість символів: ',len(words))
    words=nltk.ngrams(ua_tokenizer(words,ua_stemmer=ua_stemmer,stop_words=stop_words),n)
    words=nltk.FreqDist(words)
    print ('Кількість токенів: ',words.N())
    print ('Кількість унікальних токенів: ',words.B())
    return words


def ngrams_info(series,n=1,most_common=50,ua_stemmer=True,stop_words=STOP_WORDS):
    """ ngrams_info - Show detailed information about string pandas.Series column. 
    
    Keyword arguments:
    series -- pandas.Series object
    most_common -- show most common words(default 50)
    ua_stemmer -- if True use UkrainianStemmer for stemming words (default True)
    stop_words -- list of stop words (default [])
        
    """
    words = build_ngram_freq(series, n = n)
    common_tokens = words.most_common(most_common)
    print ('Найбільш уживані токени: ', common_tokens)
    # words.plot(most_common, cumulative = True, color = "red")
    words_df = pd.DataFrame(list(common_tokens), columns=["ngram", "count"])
    return words_df

def build_ngrams_per_author(pre_df, author, n):


    # Call ngrams_info for each unique 'owns_by' value
    
    author_df = pre_df[pre_df['owns_by'] == author]
    ngrams_df= ngrams_info(author_df['text'], n=n)
     
    
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

def label_features(dataframe, X_column,y_column, stop_words=STOP_WORDS,ua_stemmer=True,most_common=1000, n=1):
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

def nltk_classifiers(dataframe,X_column,y_column,classifier=nltk.NaiveBayesClassifier,n=1,stop_words=STOP_WORDS,ua_stemmer=True,most_common=1000): 
    
        
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
    return ua_tokenizer(text,ua_stemmer=True,stop_words=STOP_WORDS)

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
        topic = [string for string in topic if re.match(LOL_pattern, string)]
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

def vector_to_string(vector):
    return '_'.join(map(str, vector))

def main():

    comp_df = preprocess_data(CHAT_FILE_PATH)
    temp_df = ngrams_info(comp_df['text'],n=3)

    temp_df['ngram'] = temp_df['ngram'].apply(vector_to_string)
    fig = px.bar(
        temp_df,
        title="Діаграма найчастіше вживаних словосполучень",
        x="ngram",
        y="count",
        template="plotly_white",
        color_discrete_sequence=col_pal,
        labels={"count": "Кількість використання:", "ngram": "N-Gram"},
    )
    fig.update_layout(legend=dict(x=0.1, y=1.1), legend_orientation="h")
    fig.update_yaxes(title="", showticklabels=False)
    
    fig.show()
    # classification_test(comp_df)
    # for i in range(1,10):
        # print(random.uniform(0.35, 0.871))
    TOPIC, FEATURES = extract_topics(comp_df, 10, 5)
    print(FEATURES)

    
if __name__ == "__main__":
    main()
