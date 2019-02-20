# Attention Model for Neural Machine Translation: English to French

## Under Construction

Neural Machine Translation (NMT) refers to the use of neural networks, to translate one form of information into other.

Some common examples include language translation, speech to text and image captioning.

Most common neural network architectures used for machine translation are Recurrent Neural Networks (RNNs), and its variations including GRUs and LSTMs. This task falls under the sequence-to-sequence category, and traditionally, the go to model for such a task used to be encoder-decoder models.

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/encoder_decoder.png)


Encoder-decoder models consist of two parts, wherein the first part, i.e encoder, takes as input a sequence, which has to be translated and encodes the probability distribution of the input sequence, conditioned on the order of the sequence.

The second part (decoder), outputs this probability conditioned on its previous outputs. For example, in the picture above, the first two outputs of the decoder are :

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/y1.png)

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/y2.png)


When a big document is fed into an encoder-decoder, the encoder takes in the entire document before producing an output. But this is not the way a human would do it. **For example, a translator would not learn an entire English novel, before translating it into French.**

Humans do it much more efficiently by focusing only parts of sentences, to translate. This is the intuition behind **attention models**. Although even this framework takes in the entire body of document, it focuses only on parts of the document during translation. This results in much more accurate translations.






<common architectures: encoder-decoder>

<B-RNN working>

<problem with common architectures>


<attention modelling working>


<demonstration>


<attention matrix>
