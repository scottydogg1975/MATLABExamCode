%evaluate the quality of the translations using the BLEU similarity score
candidates = preprocessTextArgs(strTranslatedTest,StartToken="",StopToken="");
references = preprocessTextArgs(dataTest.Target,StartToken="",StopToken="");

minLength = min([doclength(candidates); doclength(references)])

if minLength < 4
    ngramWeights = ones(1,minLength) / minLength;
else
    ngramWeights = [0.25 0.25 0.25 0.25];
end

for i = 1:numObservationsTest
    score(i) = bleuEvaluationScore(candidates(i),references(i),NgramWeights=ngramWeights);
end

figure
histogram(score);
title("BLEU Evaluation Scores")
xlabel("Score")
ylabel("Frequency")