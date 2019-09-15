
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import spacy
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import re
import os
import time

contractions = {
"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not",
"couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
"haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did",
"how'd'y": "how do you","how'll": "how will","how's": "how is","i'd": "I would","i'd've": "I would have","i'll": "I shall","i'll've": "I shall have",
"i'm": "I am","i've": "I have","isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have",
"mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she had","she'd've": "she would have",
"she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have",
"so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there would","there'd've": "there would have",
"there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will", "they'll've": "they will have","they're": "they are",
"they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
"we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is",
"what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have","who'll": "who will",
"who'll've": "who will have","who's": "who is","who've": "who have","why's": "why has","why've": "why have","will've": "will have","won't": "would not",
"won't've": "would not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would",
"y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would","you'd've": "you would have","you'll": "you will",
"you'll've": "you will have","you're": "you are","you've": "you have","uiu": "united international university"
}

def detect_language(text):
    language = None
    try:
        from langdetect import detect
        language = detect(text)
    except:
        language = "error"
    return language

def clean(textData):
    msg_dlt = '\"message\"'
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)  # if more then 2 time an alphabet repeat then reduce them. for example heeeeee will be hee
    fbData = []
    otherLangData = []
    flag =0
    allPost=''
    all_str = ''
    for line in textData.splitlines():
        match = re.search(msg_dlt, line)
        if match:
            all_str = line
            all_str = re.sub('\"message\":', '', all_str)  # remove all "message\:"
            all_str = all_str.lower()  # convert text to lower-case
            all_str = re.sub(' [ ]+', ' ', all_str)  # remove unnecessary space
            all_str = all_str.replace('\\n', ' ')  # remove all \n
            all_str = re.sub('\?', ' ', all_str)  # remove all ? without last position of a line
            all_str = re.sub('\" *\",', '', all_str)  # remove all empty string " "
            all_str = re.sub(r'http\S+', '', all_str)  # remove all Url" "
            all_str = re.sub(r'#[ ]*([^\s]+)', r'\1', all_str)  # remove the # in #hashtag
            all_str = all_str.replace('\"', ' ')  # remove all "
            all_str = all_str.replace(',', ' ')  # remove all ,
            all_str = re.sub('\.[\.]+', ' ', all_str)  # remove ...
            all_str = re.sub('\\\\', '', all_str)  # remove backslash
            all_str = re.sub('�', '\'', all_str)
            all_str = re.sub(' i ', ' I ', all_str)

            # replace short word into full word E.g.. i m shazzad will be i am shazzad
            all_str = re.sub(r' r ', ' are ', all_str)
            all_str = re.sub(r' m ', ' am ', all_str)
            all_str = re.sub(r' u ', ' you ', all_str)
            all_str = re.sub(r' b ', ' be ', all_str)
            all_str = re.sub(r' n8 ', ' night ', all_str)
            all_str = re.sub(r' gn8 ', ' good night ', all_str)
            all_str = re.sub(r' r8 ', ' right ', all_str)
            all_str = re.sub(r' hv ', ' have ', all_str)
            all_str = re.sub(r' bt ', ' but ', all_str)
            all_str = re.sub(r' ur ', ' your ', all_str)
            all_str = re.sub(r' n ', ' and ', all_str)
            all_str = re.sub(r' bro ', ' brother ', all_str)
            all_str = re.sub(r' tha ', ' the ', all_str)
            all_str = re.sub(r' it(z)+ ', ' it\'s ', all_str)
            all_str = re.sub(r' ai ', ' artificial intelligent ', all_str)
            all_str = re.sub(r' uni ', ' university ', all_str)
            all_str = pattern.sub(r"\1\1", all_str)  # remove more then 2 characters in a row.. like amiiii will be amii
            fbData.append(all_str)
    for j in range(0, len(fbData)):
        fbData[j] = fbData[j].strip()  # remove all extra spaces
        for word in fbData[j].split():  # convert i'm = i am
            if word.lower() in contractions:
                fbData[j] = fbData[j].replace(word, contractions[word.lower()])
        if len(fbData[j]) > 2:
            language = detect_language(fbData[j])
        else:
            language = "error"
        if language == "en" or language == "error":
            if flag ==0:
                allPost = fbData[j]
                flag =1
            else:
                allPost = allPost + '\n' + fbData[j]
        else:
            fbData[j] = fbData[j] + "---->" + language
            otherLangData.append(fbData[j])
    allPost = list(filter(None, allPost))  # remove empty string
    otherLangData = list(filter(None, otherLangData))  # remove empty string

    print("other language Data: ")
    print(otherLangData)
    return allPost

app = Flask(__name__)

global nlp,graph,di_model,pss_model,sess,elmo

tf_config = os.environ.get('TF_CONFIG')
sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()
set_session(sess)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
print("spacy load complete")
di_model = load_model('di.h5')
print("di model load complete")
pss_model = load_model('pss.h5')
print("pss model load complete")
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
graph = tf.get_default_graph()



@app.route('/', methods = ['GET','POST'] )
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        _name = request.form['name']
        _email = request.form['email']
        _data = request.form['data']

        cleanData = clean(_data)
        print("first part of clean complete")
        df = pd.DataFrame(columns=['Facebook_Data'])
        df = df.append({'Facebook_Data': cleanData}, ignore_index=True)

    #pre-processing

        # remove punctuation marks
        punctuation = '!"#$%&*/;=?@[\\]`{|}~'
        df['clean_fb_data'] = df['Facebook_Data'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
        # remove whitespaces
        df['clean_fb_data'] = df['clean_fb_data'].apply(lambda x: ' '.join(x.split()))

        # function to lemmatize text
        def lemmatization(texts):
            output = []
            for i in texts:
                s = [token.lemma_ for token in nlp(i)]
                output.append(' '.join(s))
            return output

        df['clean_fb_data'] = lemmatization(df['clean_fb_data'])

    #elmo word embedding
        X = df["clean_fb_data"]

        def elmo_vectors2(x):
            embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                return sess.run(embeddings)


        # Extract ELMo embeddings
        with graph.as_default():
            set_session(sess)
            elmo_train_X = elmo_vectors2(X)
        print("train shape: ",elmo_train_X.shape)
    #load di.h5
        with graph.as_default():
            set_session(sess)
            prediction_di = di_model.predict(x=elmo_train_X)
        prediction_probability_di = np.amax(prediction_di[0])
        prediction_index_di = (np.where(prediction_di[0] == np.amax(prediction_di[0])))[0][0]
        if prediction_index_di == 0:
            predicted_di = 0 + (prediction_probability_di * 0.25)
        elif prediction_index_di == 1:
            predicted_di = 0.251 + (prediction_probability_di * 0.498)
        else:
            predicted_di = 0.75 + (prediction_probability_di * 0.25)
        di_percent = np.round(predicted_di*100)
        print("dipression percent:",di_percent)


        return render_template('result.html', di=di_percent)



@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run()
