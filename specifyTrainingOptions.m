%Specify Training Options 
miniBatchSize = 64;
numEpochs = 15;
learnRate = 0.005;

%Initialize the options for Adam optimization.
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

%Train using gradually decaying values of ϵ for scheduled sampling. Start with a value of ϵ=0.5 and linearly decay to end with a value of ϵ=0. For more information about scheduled sampling, see the Decoder Predictions Function section of the example.
epsilonStart = 0.5;
epsilonEnd = 0;

%Train using SortaGrad [3], which is a strategy to improve training of ragged sequences by training for one epoch with the sequences sorted by sequence then shuffling once per epoch thereafter.
%Sort the training sequences by sequence length.
sequenceLengths = doclength(documentsSpanish);
[~,idx] = sort(sequenceLengths);
documentsSpanish = documentsSpanish(idx);
documentsEnglish = documentsEnglish(idx);