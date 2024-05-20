import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

#first 100,000 rows are extracted for machine learning
df = pd.read_csv("Reviews.csv", nrows = 100000)
df.drop_duplicates(subset = ['Text'], inplace = True)
df.dropna(axis = 0, inplace = True)

#separate text and summary
input_data = df.loc[:,'Text']
target_data = df.loc[:,'Summary']
target_data.replace('', np.nan, inplace=True)


# PREPROCESSING ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#store input_data and target_data after cleaning and preprocessing
clean_input = []
clean_target = []

#build vocabulary from input_data and target_data by storing words
input_vocab = []
target_vocab = []

#contractions and stop words
contractions = pickle.load(open("contractions.pkl", "rb"))['contractions']
stop_words = set(stopwords.words('english'))


def cleantext(texts, src):
  #tokenize text by removing html tags and convert to lowercase
  words = word_tokenize(BeautifulSoup(texts, "lxml").text.lower())

  #remove words with non-alphabetic characters, numbers, or are too short, and expand contractions
  #stem the words to root word if needed and filter stop words
  filtered_words = []
  stemmer = LancasterStemmer()
  for word in words:
    if word.isalpha() and len(word) >= 3:
      if word in contractions:
        word = contractions[word]
      if word not in stop_words:
        if src == "inputs":
          filtered_words.append(stemmer.stem(word))
        else:
          filtered_words.append(word)
  return filtered_words


#put all texts and their corresponding summaries into pairs
for input_text, target_text in zip(input_data, target_data):
  #make input text concise and add words to vocabulary
  iwords = cleantext(input_text, "inputs")
  clean_input += [' '.join(iwords)]
  input_vocab += iwords

  #make target text concise and add words to vocabulary
  #sos indicates start and eos indicates end
  twords = cleantext("sos " + target_text + " eos", "target")
  clean_target += [' '.join(twords)]
  target_vocab += twords


#sort vocabularies in alphabetical order and get rid of duplicates
input_vocab = sorted(list(set(input_vocab)))
target_vocab = sorted(list(set(target_vocab)))
input_word_count = len(input_vocab)
target_word_count = len(target_vocab)

#use most common text length to determine reasonable length for text and summaries
max_input_len = mode([len(i) for i in clean_input])
max_target_len = mode([len(i) for i in clean_target])

print("Number of input words: ", input_word_count)
print("Number of target words: ", target_word_count)
print("Maximum input length: ", max_input_len)
print("Maximum target length: ", max_target_len)


# BUILDING THE MODEL --------------------------------------------------------------------------------------------------------------------------------------------------------------


#use 80% of data to train the model and 20% of data to assess model performance
input_train, input_test, output_train, output_test = train_test_split(clean_input, clean_target, test_size = 0.2, random_state = 0)

'''
convert texts into sequence of integers using the integer dictionary

example:
  A = ['hello world', 'machine learning is fun']

  tokenize all elements of A and make dictionary having key as tokens and value as number:
  
  B = {
    'hello': 1,
    'world': 2,
    'machine': 3,
    'learning': 4,
    'is': 5,
    'fun': 6
  }

  transform A into indexes, call it C:

  C = [[1, 2], [3, 4, 5, 6]]

ensure sentences have uniform length to process in parallel by padding with 0s
'''

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_train)
input_train = input_tokenizer.texts_to_sequences(input_train)
encoder_input_data = pad_sequences(input_train, maxlen = max_input_len, padding = 'post')

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(output_train)
output_train = target_tokenizer.texts_to_sequences(output_train)
decoder_data = pad_sequences(output_train, maxlen = max_target_len, padding = 'post')

decoder_input_data = decoder_data[:, :-1] #remove eos
decoder_target_data = decoder_data.reshape(len(decoder_data), max_target_len, 1)[:, 1:] #remove sos


'''
encoder: reads and understands input text
decoder: generates the summary based on the understanding from the encoder

example of embedding:
  assume we have the following input sequences:
    input_sequences = [[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]]

  the embedding layer maps each word index to a dense vector based on the word's definition
  lets assume the embedding_matrix is:
    embedding_matrix = {
      2: [0.1, 0.2, 0.3],
      3: [0.4, 0.5, 0.6],
      4: [0.7, 0.8, 0.9],
      5: [0.01, 0.02, 0.03],
      6: [0.04, 0.05, 0.06],
      7: [0.07, 0.08, 0.09],
      8: [0.1, 0.11, 0.12],
      9: [0.13, 0.14, 0.15],
      0: [0.0, 0.0, 0.0]  (Padding)
  }

  then the output tensor, en_embedding is:
    en_embedding = [
      [  # First sequence
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.0, 0.0, 0.0],  (Padding)
        [0.0, 0.0, 0.0]   (Padding)
      ],
      [  # Second sequence
        [0.01, 0.02, 0.03],
        [0.04, 0.05, 0.06],
        [0.07, 0.08, 0.09],
        [0.1, 0.11, 0.12],
        [0.13, 0.14, 0.15]
      ]
    ]

    batch_size = 2 (2 matrices)
    max_in_len = 5 (each matrix has 5 vectors)
    latent_dim = 3 (each vector has 3 elements)
    remember that each vector in this case represents a word
    
each LSTM encoder layer will attempt to summarize the input sequence and give contextual information
and will pass only the summary to the next layer
  outputs: summary
  h (hidden state): contextual information
  c (cell state): contextual information

the next encoder layer will use the passed summary to attempt to generate a better summary and give
better contextual information

en_states contains the contextual information from the final summary generation attempt

the LSTM decoder layer will use this final contextual information and generate a final summary that
will represent the summary of the beginning input text
'''

K.clear_session()
latent_dim = 500

encoder_inputs = Input(shape = (max_input_len,))
encoder_embedding = Embedding(input_word_count + 1, latent_dim)(encoder_inputs)
outputs1, h1, c1 = LSTM(latent_dim, return_state = True, return_sequences = True)(encoder_embedding)
outputs2, h2, c2 = LSTM(latent_dim, return_state = True, return_sequences = True)(outputs1)
outputs3, h3, c3 = LSTM(latent_dim, return_state = True, return_sequences = True)(outputs2)
encoder_states = [h3, c3]

decoder_inputs = Input(shape = (None,))
decoder_embedding = Embedding(target_word_count + 1, latent_dim)(decoder_inputs)
decoder_outputs, *_ = LSTM(latent_dim, return_state = True, return_sequences = True)(decoder_embedding, initial_state = encoder_states) 


'''
the attention layer takes the decoder's current output and encoder's output and uses the word from the
decoder and the summary from the encoder to decide which parts of the beginning input text are important.

by merging the context vector with the decoder's output, the model will make more informed predictions.

example:
  input: "AI applications can be found in various fields, including healthcare, education, and finance."
  
  decoder output: let's say the decoder is generating the word "field"
  encoder output: the encoder has processed the entire paragraph and has outputs representing different
  parts of the paragraph.

  attention layer looks at both the decoder's current state (word before field) and the encoder's
  outputs. from this, it decides which parts of the input paragraph are most relavent for generating
  the next word, "field".

  basically, the attention layer is using what the decoder outputs to infer what parts of the input
  and encoder output made the decoder generate the word. these parts of the input are saved in the
  context vector.

the dense layer transforms this merged vector (context vector and decoder output) into probabilities
for each word in the target vocabulary, enabling the model to choose the most likely next word in
the summary.
'''

attention = Attention()
context_vector = attention([decoder_outputs, outputs3])
merge = Concatenate(axis = -1, name = 'concat_layer1')([decoder_outputs, context_vector])
decoder_outputs = Dense(target_word_count + 1, activation = 'softmax')(merge)


# TRAIN MODEL ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#combine input, target, and predicted sequences (encoder and decoder) to create the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

#creates visual representation of the model's architecture
plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)


'''
compiling the model will configure the learning process of the model

optimizer (RMSprop): efficiently trains model by dynamically adjusting learning rate
loss function (scc): efficiently handles the integer-encoded word indices
metrics (accuracy): monitors how often model prediction is accurate during training

model processes data in batches of 512 samples, repeated 10 times, 10% of data being used for validation

'''

model.compile(optimizer = "rmsprop", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = 512, epochs = 10, validation_split = 0.1,)
model.save("s2s")


# INFERENCE MODEL -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


'''
encoder inference: processes a new input sequence and outputs the final hidden and cell states, which
will be used to initialize the decoder for generating the target sequence

during inference, the encoder model reads and processes the input text and provides the important parts
of the input sequence to the decoder.
'''

latent_dim = 500
model = models.load_model("s2s")
encoder_outputs, hidden, cell = model.layers[6].output
encoder_states = [hidden, cell]

#intialize model with input as the inference encoder model and output as the last LSTM layer
encoder_model = Model(model.input[0], [encoder_outputs] + encoder_states)


"""
decoder inference: generates the summary from the processed input sequence by using the passed final
states from the encoder and the decoder's previous outputs.

during inference, the decoder model generates each word in the summary one at a time, using the
attention mechanism to focus on relevant parts of the input sequence.
"""

#define intial hidden and cell states for decoder
decoder_hidden = Input(shape = (latent_dim,))
decoder_cell = Input(shape = (latent_dim,))
encoder_hidden_output = Input(shape = (max_input_len, latent_dim))

#input, lstm, and embedding layers
decoded_inputs = model.input[1]
decoded_lstm = model.layers[7]
decoded_embedding = model.layers[5](decoded_inputs)

#process embedded decoder inputs to generate output sequence and updated states
decoder_outputs2, h2, c2 = decoded_lstm(decoded_embedding, initial_state = [decoder_hidden, decoder_cell])


'''
attention layer: decoder deduces important parts of the input text in context vector
dense layer: converts combined context and decoder output into probabilities for word prediction
'''

attention = model.layers[8]
attention_output2 = attention([decoder_outputs2, encoder_hidden_output])
merge2 = Concatenate(axis = -1)([decoder_outputs2, attention_output2])

dense = model.layers[10]
decoder_outputs2 = dense(merge2)

#intializes inference model
predictor = Model([decoded_inputs] + [encoder_hidden_output, decoder_hidden, decoder_cell], [decoder_outputs2] + [h2, c2])

#create dictionary with key as index and value as words, map word indices to words and vice versa
index_to_word = target_tokenizer.index_word
word_to_index = target_tokenizer.word_index
index_to_word[0] = ' '

#using trained model, decode an input sentence into a target sequence one word at a time
def summarize(input_seq):
  #intialize a target sequence
  target_seqence = np.zeros((1, 1))
  target_seqence[0, 0] = word_to_index['sos']

  summary = ""
  while True:
    #get predicted word probabilities and updated states
    output_words, *_ = predictor.predict([target_seqence] + [encoder_model.predict(input_seq)])
    
    word_index = np.argmax(output_words[0, -1, :]) #finds index of most probable next word
    text_word = index_to_word[word_index] #converts word index to actual word
    summary += text_word + " " #append predicted word to final sentence

    #check if predicted word is end-of-sequence or length reaches max length
    if len(summary) > max_target_len or text_word == "eos":
      break
        
    #set first element of target sequence to next word to process
    target_seqence = np.zeros((1, 1))
    target_seqence[0, 0] = word_index
  
  return summary


# FINAL RESULT -------------------------------------------------------------------------------------------------------------------------------------------------------------------


#user enters a text review
review = input("Enter: ")
print("Review: ", review)

#preprocess text review
review = cleantext(review, "inputs")
review = ' '.join(review)

#convert cleaned text review into sequence of indices and pad sequences
indices = input_tokenizer.texts_to_sequences([review])
indices = pad_sequences(indices,  maxlen = max_input_len, padding = 'post')

#generate a summary for review
summary = summarize(indices.reshape(1, max_input_len))
if 'eos' in summary:
  summary = summary.replace('eos', '')
print("\nPredicted summary: ", summary)