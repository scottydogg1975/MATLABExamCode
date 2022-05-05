%Number from 0-1 for discarding percentage of data down the line
discardProp = 0.70;

downloadFolder = tempdir;
url = "http://www.manythings.org/anki/spa-eng.zip";
filename = fullfile(downloadFolder,"spa-eng.zip");
dataFolder = fullfile(downloadFolder,"spa-eng");

if ~exist(dataFolder,"dir")
    fprintf("Downloading English-Spanish Tab-delimited Bilingual Sentence Pairs data set (5.2 MB)... ")
    websave(filename,url);
    unzip(filename,dataFolder);
    fprintf("Done.\n")
end