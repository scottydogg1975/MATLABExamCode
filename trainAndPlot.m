%Train Model
%Create array data stores
adsSource = arrayDatastore(documentsSpanish);
adsTarget = arrayDatastore(documentsEnglish);
cds = combine(adsSource,adsTarget);

%Create minibatch queue
mbq = minibatchqueue(cds,4, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,Y) preprocessMiniBatch(X,Y,encSpanish,encEnglish), ...
    MiniBatchFormat=["CTB" "CTB" "CTB" "CTB"], ...
    PartialMiniBatch="discard");

%Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));

xlabel("Iteration")
ylabel("Loss")
ylim([0 inf])
grid on

%For the encoder and decoder networks, initialize the values for Adam optimization.
trailingAvgEncoder = [];
trailingAvgSqEncoder = [];
trailingAvgDecder = [];
trailingAvgSqDecoder = [];

%Create an array of ϵ values for scheduled sampling.
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
numIterations = numIterationsPerEpoch * numEpochs;
epsilon = linspace(epsilonStart,epsilonEnd,numIterations);

%Train the model. For each iteration:

%Read a mini-batch of data from the mini-batch queue.
%Compute the model loss and gradients.
%Update the encoder and decoder networks using the adamupdate function.
%Update the training progress plot and display an example translation using the ind2str function, attached to this example as a supporting file. To access this function, open this example as a live script.
%If the iteration yields the lowest training loss, then save the network.
%At the end of each epoch, shuffle the mini-batch queue.

iteration = 0;
start = tic;
lossMin = inf;
reset(mbq)

% Loop over epochs.
for epoch = 1:numEpochs

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T,maskT,decoderInput] = next(mbq);
        
        % Compute loss and gradients.
        [loss,gradientsEncoder,gradientsDecoder,YPred] = dlfeval(@modelLoss,netEncoder,netDecoder,X,T,maskT,decoderInput,epsilon(iteration));
        
        % Update network learnable parameters using adamupdate.
        [netEncoder, trailingAvgEncoder, trailingAvgSqEncoder] = adamupdate(netEncoder,gradientsEncoder,trailingAvgEncoder,trailingAvgSqEncoder, ...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        [netDecoder, trailingAvgDecder, trailingAvgSqDecoder] = adamupdate(netDecoder,gradientsDecoder,trailingAvgDecder,trailingAvgSqDecoder, ...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);

        % Generate translation for plot.
        if iteration == 1 || mod(iteration,10) == 0
            strSpanish = ind2str(X(:,1,:),encSpanish);
            strEnglish = ind2str(T(:,1,:),encEnglish,Mask=maskT);
            strTranslated = ind2str(YPred(:,1,:),encEnglish);
        end

        % Display training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(gather(extractdata(loss)));
        addpoints(lineLossTrain,iteration,loss)
        title( ...
            "Epoch: " + epoch + ", Elapsed: " + string(D) + newline + ...
            "Source: " + strSpanish + newline + ...
            "Target: " + strEnglish + newline + ...
            "Training Translation: " + strTranslated)

        drawnow
        
        % Save best network.
        if loss < lossMin
            lossMin = loss;
            netBest.netEncoder = netEncoder;
            netBest.netDecoder = netDecoder;
            netBest.loss = loss;
            netBest.iteration = iteration;
            netBest.D = D;
        end
    end

    % Shuffle.
    shuffle(mbq);
end