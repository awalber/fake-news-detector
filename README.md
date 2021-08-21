# Fake News Detection

This project, hosted by [Kaggle](https://www.kaggle.com/c/fake-news) involved building a model to classify sections of text into reliable or unreliable news. I chose to build an LSTM (Long Short-Term Memory) Neural Network to perform the classifications due to its ability to process sequential data while preserving the time aspect. The predictions calculated on the testing data produced an accuracy value of 95.8%.

## Advantages of an LSTM Neural Network
The LSTM was chosen because it uses the RNN (Recurrent Neural Network) architecture while circumventing the issue of a vanishing or exploding gradient. RNNs do an excellent job at analyzing information in a way that preserves the temporal aspect of the data. However, with long sequences of data, the gradient can quickly approach zero or infinity. To better explain this phenomenon, the following image has been referenced from [Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network). In this image, X is the input sequence, h is the hidden layer of the RNN, and o is the output of feeding X through the model. Specifically for this problem, X is a batch of padded, embedded inputs representing the news article whose truth should be predicted. Since the Model utilizes the temporal meaning behind the text, the output from each timestep (starting with t<sub>0</sub>) is fed into the proceeding timestep to form chronological "layers". The significance of this is that each of the model's weights also dependent on the preceding weight in the RNN, which makes the gradient multiplicative. Thus:

<p align="center">
    <img src="/imgs/eq_1.png" | width=300>
</p>
<p align="center">
    <img src="/imgs/Recurrent_neural_network_unfold.png" | width=650>
</p>

The LSTM retains the RNN architecture while mitigating the multiplicative gradient by utilizing "forget gates", which regulates the amount of information passed forward through the model. The short-term memory aspect of the model supports the generation of speech/text because usually the meaning of the word or sentence is directly influenced by other words or sentences in its vicinity. Strictly speaking, the first word in an article most likely has minimal influence on the actual meaning of the article five hundred words deep.