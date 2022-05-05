function strTranslated = translateText(netEncoder,netDecoder,encSpanish,encEnglish,strSpanish,args)

% Parse input arguments.
arguments
    netEncoder
    netDecoder
    encSpanish
    encEnglish
    strSpanish
    
    args.BeamIndex = 3;
end

beamIndex = args.BeamIndex;

% Preprocess text.
documentsSpanish = preprocessText(strSpanish);
X = preprocessPredictors(documentsSpanish,encSpanish);
X = dlarray(X,"CTB");

% Loop over observations.
numObservations = numel(strSpanish);
strTranslated = strings(numObservations,1);             
for n = 1:numObservations
    
    % Translate text.
    strTranslated(n) = beamSearch(X(:,n,:),netEncoder,netDecoder,encEnglish,BeamIndex=beamIndex);
end

end