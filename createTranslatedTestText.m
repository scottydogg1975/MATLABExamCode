%To evaluate the quality of the translations, use the BiLingual Evaluation Understudy (BLEU) scoring algorithm
strTranslatedTest = translateText(netEncoder,netDecoder,encSpanish,encEnglish,dataTest.Source);
