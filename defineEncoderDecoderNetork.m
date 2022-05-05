%Create the encoder and decoder networks using the languageTranslationLayers function, attached to this example as a supporting file. To access this function, open the example as a live script. Specify an embedding dimension of 128, and 128 hidden units in the LSTM layers.
embeddingDimension = 128;
numHiddenUnits = 128;

[lgraphEncoder,lgraphDecoder] = languageTranslationLayers(embeddingDimension,numHiddenUnits,numWordsSpanish,numWordsEnglish);
%To train the network in a custom training loop, convert the encoder and decoder networks to dlnetwork objects.
netEncoder = dlnetwork(lgraphEncoder);
netDecoder = dlnetwork(lgraphDecoder);
%The decoder has multiple outputs including the context output of the attention layer, which is also passed to another layer. Specify the network outputs using the OutputNames property of the decoder dlnetwork object.
netDecoder.OutputNames = ["softmax" "context" "lstm2/hidden" "lstm2/cell"];