


from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from pickle import load
import random
get_ipython().magic('matplotlib inline')









# load doc into memory
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y



def uniform_sentences(sentences_X,sentences_Y,length=50):

	outputs_X = []
	outputs_Y = []
	m = len(sentences_X)
	for i in range(m):

		word_list_X = sentences_X[i].split()
		word_list_Y = sentences_Y[i].split()
		if((len(word_list_X) > length) and (len(word_list_Y) > length)):

			word_list_X = word_list_X[:length]
			word_list_Y = word_list_Y[:length]
			outputs_X.append(word_list_X)
			outputs_Y.append(word_list_Y)

	return outputs_X, outputs_Y



def chuck_nonglove(sentences_X,sentences_Y,word_to_vec_map):

	outputs_X = []
	outputs_Y = []
	m = len(sentences_X)
	for i in range(m):

		chuck = False
		for word in sentences_X[i]:
			if word not in word_to_vec_map.keys():
				chuck = True
				break
		if(not chuck):
			outputs_X.append(sentences_X[i])
			outputs_Y.append(sentences_Y[i])

	return outputs_X, outputs_Y


def to_glove_vectors(sentences_X,word_to_vec_map):
    outputs = np.zeros((sentences_X.shape[0],sentences_X.shape[1],50))

    for i in range(len(sentences_X)):
        for j in range(len(sentences_X[i])):
            outputs[i,j,:] = word_to_vec_map[sentences_X[i,j]]

    return outputs


def create_french_dict(french_sentences):
	french_dict = []
	for sentence in french_sentences:
		for word in sentence:
			if(word not in french_dict):
				french_dict.append(word)

	return french_dict




word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
X = load_clean_sentences('english_vocab.pkl')
Y = load_clean_sentences('french_vocab.pkl')
X, Y = uniform_sentences(X,Y)
X, Y = chuck_nonglove(X,Y,word_to_vec_map)
X = to_glove_vectors(np.array(X),word_to_vec_map)
Y = np.array(Y)

#french_dict = create_french_dict(Y)

# with open('french_dict.dat', 'w') as f:
#     for item in french_dict:
#         f.write("%s\n" % item)


with open('french_dict.dat', 'r') as f:
	french_dict = f.readlines()
	french_dict = [w.strip('\n') for w in french_dict]


french_word_to_id = dict(zip(french_dict,range(len(french_dict))))

french_id_to_word = dict(zip(range(len(french_dict)),french_dict))

french_vocab_size = len(french_dict)

def one_hot_Y(Y,french_word_to_id):
	Y_oh = np.zeros((Y.shape[0],Y.shape[1],french_vocab_size))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y_oh[i,j,french_word_to_id[Y[i,j]]] = 1

	return Y_oh


Y_oh = one_hot_Y(Y[:10000],french_word_to_id)
X = X[:10000]

X.shape
Y_oh.shape

Tx = 50
Ty = 50
#######################################################################################################################################################################################
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation("softmax", name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)




def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """


    s_prev = repeator(s_prev)

    concat = concatenator([a,s_prev])

    e = densor1(concat)

    energies = densor2(e)

    alphas = activator(energies)

    context = dotor([alphas,a])


    return context



n_a = 64
n_s = 256
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(french_vocab_size, activation="softmax")




def model(Tx, Ty, n_a, n_s, glove_size, french_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    glove_size -- size of the glove embeddings
    french_vocab_size -- size of the python dictionary "french_dict"

    Returns:
    model -- Keras model instance
    """


    X = Input(shape=(Tx, glove_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0


    outputs = []




    a = Bidirectional(LSTM(n_a,return_sequences=True))(X)


    for t in range(Ty):


        context = one_step_attention(a,s)


        s, _, c = post_activation_LSTM_cell(context,initial_state=[s,c])


        out = output_layer(s)


        outputs.append(out)


    model = Model(inputs=[X,s0,c0],outputs=outputs)



    return model




model = model(Tx, Ty, n_a, n_s, 50, french_vocab_size)

model.summary()


opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])



s0 = np.zeros((10000, n_s))
c0 = np.zeros((10000, n_s))
outputs = list(Y_oh[:10000].swapaxes(0,1))




model.fit([X[:10000], s0, c0], outputs, epochs=100, batch_size=500)




# preds = model.predict([X[:2],s0,c0])
#
# preds = np.array(preds)
#
# #preds[:,i,:].shape = (50,42606)
#
def preds_to_sen(preds):
	assert(preds.shape == (50,42606))

	outputs=[]
	for w in preds:
		outputs.append(french_id_to_word[np.argmax(w)])


	return " ".join(outputs)






#    prediction = model.predict([source, s0, c0])
#    prediction = np.argmax(prediction, axis = -1)








#attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "", num = 7, n_s = 64)
