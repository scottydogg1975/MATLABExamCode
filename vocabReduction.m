%Reduce Vocabulary
idx = size(data,1) - floor(discardProp*size(data,1)) + 1;
data(idx:end,:) = [];

%View number of remaining data
size(data,1)