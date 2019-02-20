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

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/german.jpg)

## Attention models

Attention models (due to [Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio](https://arxiv.org/abs/1409.0473)) work by stacking up one more layer of RNNs (LSTM/GRU) over the input layer, the so called "attention layer". The number of times this RNN is unrolled is equal to the number of output units (in this case, the number of words in the translated text). Each unit in this layer is connected to all the units of the input layer, and receives a "context" variable denoted by ***C***. In the figure below, the input layer is a **Bi-directional LSTM**. Simple uni-directional LSTMs can also be used but to capture connections amongst words better, B-RNNs are recommended, but they come at an increased computational cost.

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/attention.png)

Context ***C*** receives input from all the nodes of the input layer, weighted by the matrix alpha. The alpha matrix captures the importance that the t<sup>th</sup> output should give to the t'<sup>th</sup> input. Formally,

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/context.png)

Where T<sub>x</sub> is the number of words in the input.

Intuitively, the importance that the t-th output should give to the t'-th input should depend on the t'-th input and the activations of the nodes in the attention layer that came before the t-th output. Also, for each output t, we define the alphas to be between 0 and 1. Since each output t should get attended, we constrain the sum of alphas to be 1. Given this, the softmax function is a good candidate for defining alphas. But since we don't know what's the exact function that governs this dynamic, we can let it be learnt! Therefore, we can write :

![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/attention_def.png)

Where the vectors ***e*** are some representations of the input nodes and previous hidden states of the attention layer. Since we don't know what these representations look like, we can let them be learnt by constructing a small feedforward neural network which takes as input the activations of the input layer and previous hidden state, and outputs the vectors ***e***


![alt text](https://raw.githubusercontent.com/sarangzambare/nmt_attention/master/png/mini_nn.png)

With this architecture, the network learns the alpha matrix, and learns to focus on the right words while translating.


## Demonstration : English to French translation, with attention.

To demonstrate neural machine translation, I worked with the task of language translation.

I used the [briefings of the Europian Parliament](http://www.statmt.org/europarl/), which consists of more than **200,000** sentences in parallel text format with parallel texts for French and English. I preprocessed the text to :

1. Only keep sentences which are 50 words or longer, and truncate them to 50 words for uniformity
2. Removed all sentences which contained words not in the GloVe word embeddings
3. Removed all punctuation marks, including html tags (<>)
4. Converted everything to lower-case.






## References

1. [Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

<common architectures: encoder-decoder>

<B-RNN working>

<problem with common architectures>


<attention modelling working>


<demonstration>


<attention matrix>
