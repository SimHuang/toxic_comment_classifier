
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import *
from nltk.stem import PorterStemmer, WordNetLemmatizer

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, Dense

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

from sentic import SenticPhrase


# In[2]:


dataframe_train = pd.read_csv("/Users/simon/Desktop/train.csv")
dataframe_test = pd.read_csv("/Users/simon/Desktop/test.csv")

#removing stop words 'the, if, or, a, and etc.'
#TODO: we can remove additional words such as names
def clean_text(comment):
    stoplist = set(stopwords.words('english')) 
    clean = []
    for word in comment.split(" "):
        if word not in stoplist:
            clean.append(word)
                
    return ' '.join(clean)

comment_text = dataframe_train["comment_text"].tolist()
parsed_comments = [] 
for comment in comment_text:
    comment = clean_text(comment)
    parsed_comments.append(comment)
    


# In[3]:


#extract featues out of the comments
#maybe we can split the list into swear words, insult, threat etc.
swear_words = ["fuck", "bitch", "damn", "fk", "shit", "pussy", "motherfucker", "asshole", "bastard"]
#store all the swear words used in each comment 
swear_word_feature_list = []
#check whether this is question
question_feature = []

for comment in parsed_comments:
    comment_tokens = nltk.word_tokenize(comment);
    l = []
    isQuestion = 0
    for word in comment_tokens:
        #check for swear words
        if word in swear_words:
            l.append(word)
            
        #check if it contains ?
        char_tokens = word.split()
        for c_token in char_tokens:
            if c_token == "?":
                isQuestion = 1
            
    question_feature.append(isQuestion)
    swear_word_feature_list.append(",".join(l));
    
#insert comment into dataframe_train as new feature
dataframe_train.insert(loc=2, column="swears", value=swear_word_feature_list)
dataframe_train.insert(loc=3, column="question", value=question_feature)


# In[28]:


#stemming and lemmatization - 
#lemmatization 
#ISSUE: stemming caused max recursion depth error, ignore stemming for now
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

sample_comment = parsed_comments[0]
for i in range(len(parsed_comments)):
    comment = []
    for word in parsed_comments[i].split():
#         comment.append(stemmer.stem(lemmatizer.lemmatize(word)))
        comment.append(lemmatizer.lemmatize(word))
    parsed_comments[i] = " ".join(comment)


# In[29]:


parsed_comments


# In[5]:


#Loading training data
x_train = dataframe_train["comment_text"].tolist()
x_test = dataframe_test["comment_text"].tolist()

#Loading training label data
y_train = dataframe_train['toxic'].values
#y_train = dataframe_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values


# In[6]:


len(x_train)


# In[7]:


#Create a tokenizer, converts words to integers
#None represents full vocabulary available

num_words = 1000
tokenizer = Tokenizer(num_words=num_words) 

#TL Notes:
#We create word index from both test and training data
#Tokenizer now has four attributes available
#   word_counts: Dictionary of words and their counts
#   word_docs: Dictionary of words and how many documents each appeared in
#   word_index: A dictionary of words and their uniquely assigned integers
#   document_count: An integer count of the total number of dicument used to fit tokenizer
tokenizer.fit_on_texts(x_train+x_test)
tokenizer.word_index


# In[8]:


#Create token vectors for training data
x_train_tokens = tokenizer.texts_to_sequences(x_train)

#Create token vectors for test data
x_test_tokens = tokenizer.texts_to_sequences(x_test)

#Example: Printing token vector for the first comment of training set
np.array(x_train_tokens[0])


# In[9]:


num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

np.mean(num_tokens)
np.max(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens



# In[10]:


#Pad to make comments uniformly shaped
pad = 'pre'

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

x_train_pad[0]





# In[11]:


#Create a polarity Array

x_train_polarityScore = [] 
for comment in x_train:
    comment = SenticPhrase(comment)
    x_train_polarityScore.append(comment.get_polarity())
    
x_train_polarityScore = np.array(x_train_polarityScore)


# In[12]:


x_test_polarityScore = [] 
for comment_test in x_test:
    comment_test = SenticPhrase(comment_test)
    try:
       x_test_polarityScore.append(comment_test.get_polarity())
    except:
       x_test_polarityScore.append(0)
    
x_test_polarityScore = np.array(x_test_polarityScore)


# In[13]:


#Creating RNN, Base Model
model_RNN = Sequential()
embedding_size = 8
model_RNN.add(Embedding(input_dim=num_words,
                        output_dim=embedding_size,
                        input_length=max_tokens,
                        name='layer_embedding'))

model_RNN.add(LSTM(100))
model_RNN.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)

model_RNN.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model_RNN.summary()


# In[14]:


model_RNN.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=1, batch_size=64)


# In[15]:


model_RNN.predict(x=x_test_pad[0:1])


# In[16]:


#Creating RNN, LSTM with Sentic Polarity
model_HYB = Sequential()
embedding_size = 8
main_input = Input(shape=(max_tokens,), name='main_input')
embedInput = Embedding(input_dim=num_words,
                        output_dim=embedding_size,
                        input_length=max_tokens)(main_input)
embedOut = LSTM(100)(embedInput)

polarInput = Input(shape=(1,), name = 'polarity_input')
mergedInput = concatenate([embedOut, polarInput])
mergedInput = Dense(64, activation = 'relu')(mergedInput)
mergedInput = Dense(64, activation = 'relu')(mergedInput)
mergedInput = Dense(64, activation = 'relu')(mergedInput)
mergedOut = Dense(1, activation = 'sigmoid', name = 'mergedOut')(mergedInput) 

model_HYB = Model(inputs=[main_input, polarInput], outputs=[mergedOut])

optimizer = Adam(lr=1e-3)
model_HYB.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy']
                 )

model_HYB.summary()


# In[17]:


model_HYB.fit([x_train_pad,x_train_polarityScore], y_train,
          validation_split=0.05, epochs=1, batch_size=64)


# In[18]:


model_HYB.predict(x=[x_test_pad[0:10],x_test_polarityScore[0:10]])


# In[19]:


y_train_predict = model_HYB.predict(x=[x_train_pad,x_train_polarityScore])


# In[20]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_train_predict, None, None, False)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[21]:


#helper function to construct model
def construct_RNN_HYB_model(x_train, y_train, x_score, x_test, y_test):
    model_HYB = None
    model_HYB = Sequential()
    embedding_size = 8
    main_input = Input(shape=(max_tokens,), name='main_input')
    embedInput = Embedding(input_dim=num_words,
                            output_dim=embedding_size,
                            input_length=max_tokens)(main_input)
    embedOut = LSTM(100)(embedInput)

    polarInput = Input(shape=(1,), name = 'polarity_input')
    mergedInput = concatenate([embedOut, polarInput])
    mergedInput = Dense(64, activation = 'relu')(mergedInput)
    mergedInput = Dense(64, activation = 'relu')(mergedInput)
    mergedInput = Dense(64, activation = 'relu')(mergedInput)
    mergedOut = Dense(1, activation = 'sigmoid', name = 'mergedOut')(mergedInput) 

    model_HYB = Model(inputs=[main_input, polarInput], outputs=[mergedOut])

    optimizer = Adam(lr=1e-3)
    model_HYB.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy']
                     )
#     model_HYB.fit([x_train, x_score], y_train, validation_split=0.5, epochs=1, batch_size=64)
    model_HYB.fit([x_train, x_score], y_train, validation_data=(x_test, y_test), epochs=1, batch_size=64)
    return model_HYB


# In[22]:


#kFold cross validation on hybrid model
def k_fold_validate(x, y, folds, x_train_polarity_score):
    kf = KFold(5)
    fold = 0
    accuracies = []
    for train, test in kf.split(x):
        print(train)
        print(test)
        #print out number of fold
        fold += 1
        print("Fold #{}".format(fold))
        
        #extract dataset to use as training and testing
        #everything is coming from the x_train_pad
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        x_train_score = x_train_polarity_score[train]
#         x_test_score = x_test_polarity_score[test]
        
        model_HYB = construct_RNN_HYB_model(x_train, y_train, x_train_score, x_test, y_test)
        scores = model_HYB.evaluate(x=[x_train, x_train_score], y=y_train)
        print(scores[1])
        accuracies.append(scores[1]) #accuracy score
    
    return accuracies


# In[24]:


accuracies = k_fold_validate(x_train_pad, y_train, 5, x_train_polarityScore)
accuracies


# In[25]:


sd = np.std(accuracies)
print(sd)


# In[27]:


mean = np.mean(accuracies)
print(mean)


# In[30]:


def construct_RNN_model(x_train, y_train, x_test, y_test):
    model_RNN = Sequential()
    embedding_size = 8
    model_RNN.add(Embedding(input_dim=num_words,
                            output_dim=embedding_size,
                            input_length=max_tokens,
                            name='layer_embedding'))

    model_RNN.add(LSTM(100))
    model_RNN.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)

    model_RNN.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
#     model_RNN.fit(x_train, y_train,
#           validation_split=0.05, epochs=1, batch_size=64)
    model_RNN.fit(x_train, y_train,
          validation_data=(x_test, y_test), epochs=1, batch_size=64)
    return model_RNN
    
    


# In[32]:


#kFold cross validation on hybrid model
def rnn_k_fold_validate(x, y, folds):
    kf = KFold(5)
    fold = 0
    accuracies = []
    for train, test in kf.split(x):
        print(train)
        print(test)
        #print out number of fold
        fold += 1
        print("Fold #{}".format(fold))
        
        #extract dataset to use as training and testing
        #everything is coming from the x_train_pad
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
#         x_train_score = x_train_polarity_score[train]
#         x_test_score = x_test_polarity_score[test]
        
#         model_HYB = construct_RNN_HYB_model(x_train, y_train, x_train_score, x_test, y_test)
        model_RNN = construct_RNN_model(x_train, y_train, x_test, y_test)
        scores = model_RNN.evaluate(x=x_train, y=y_train)
        print(scores[1])
        accuracies.append(scores[1]) #accuracy score
    
    return accuracies


# In[33]:


rnn_accuracies = rnn_k_fold_validate(x_train_pad, y_train, 5)
rnn_accuracies


# In[34]:


sd = np.std(rnn_accuracies)
print(sd)

