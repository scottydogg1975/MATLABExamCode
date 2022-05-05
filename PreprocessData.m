%Tokenize 
documentsSpanish = preprocessText(dataTrain.Source);

%Create a wordEncoding object that maps tokens to a numeric index and vice versa using a vocabulary.
encSpanish = wordEncoding(documentsSpanish);

%Convert the target data to sequences using the same steps.
documentsEnglish = preprocessText(dataTrain.Target);
encEnglish = wordEncoding(documentsEnglish);

%View the vocabulary sizes of the source and target encodings.
numWordsSpanish = encSpanish.NumWords

numWordsEnglish = encEnglish.NumWords