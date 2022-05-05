%create table that contains sentence pairs
filename = fullfile(dataFolder,"spa.txt");

opts = delimitedTextImportOptions(...
    Delimiter="\t", ...
    VariableNames=["Target" "Source" "License"], ...
    SelectedVariableNames=["Source" "Target"], ...
    VariableTypes=["string" "string" "string"], ...
    Encoding="UTF-8");

data = readtable(filename, opts);
head(data)