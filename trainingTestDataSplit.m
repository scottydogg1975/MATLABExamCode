%Split data into traning and test partitions containing 90% and 10%
%respectivly

trainingProp = 0.9;
idx = randperm(size(data,1),floor(trainingProp*size(data,1)));
dataTrain = data(idx,:);
dataTest = data;
dataTest(idx,:) = [];

%first few rows of training data 

head(dataTrain)

%View num of training observations

numObservationsTrain = size(dataTrain,1)