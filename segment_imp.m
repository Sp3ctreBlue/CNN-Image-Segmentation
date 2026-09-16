close all;
clear;
clc;

% data paths and loading image store
imgDir   = fullfile('cw_data','images');
labelDir = fullfile('cw_data','segmentation');

imds = imageDatastore(imgDir);

% class definition
classNames = ["background","crop","weed"];

labelIDs = [
    0   0   0;      % background(dirt)
    0 255   0;      % crop
    255 0   0       % weed
];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

% split data 80/20
numFiles = numel(imds.Files);
rng(1);

idx = randperm(numFiles);

numTrain = round(0.8*numFiles);

trainIdx = idx(1:numTrain);
valIdx   = idx(numTrain+1:end);

imdsTrain = subset(imds,trainIdx);
pxdsTrain = subset(pxds,trainIdx);

imdsVal = subset(imds,valIdx);
pxdsVal = subset(pxds,valIdx);

inputSize = [240 320 3];
numClasses = 3;

% augmentation for training
augmenter = imageDataAugmenter( ...
    'RandXReflection',true,...
    'RandRotation',[-5 5],...
    'RandXTranslation',[-10 10],...
    'RandYTranslation',[-10 10]);

trainingData = pixelLabelImageDatastore( ...
    imdsTrain,pxdsTrain,...
    'OutputSize',[240 320],...
    'DataAugmentation',augmenter);

validationData = pixelLabelImageDatastore( ...
    imdsVal,pxdsVal,...
    'OutputSize',[240 320]);

% network
layers = [

imageInputLayer(inputSize)

convolution2dLayer(3,32,'Padding','same')
reluLayer

convolution2dLayer(3,32,'Padding','same')
reluLayer

convolution2dLayer(3,64,'Padding','same')
reluLayer

convolution2dLayer(3,64,'Padding','same')
reluLayer

convolution2dLayer(1,numClasses)

softmaxLayer
pixelClassificationLayer
];

% training options
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-3,...
    'MaxEpochs',30,...
    'MiniBatchSize',4,...
    'Shuffle','every-epoch',...
    'ValidationData',validationData,...
    'VerboseFrequency',10,...
    'Plots','training-progress');

% TRAIN
%net = trainNetwork(trainingData,layers,opts);

%save('segmentnet_imp.mat','net');
 load('segmentnet_imp.mat');

pxdsResults = semanticseg(imdsVal,net,'MiniBatchSize',1);

% evaluation
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);

disp(metrics.ClassMetrics(2:3,:))   % crop & weed
disp(metrics.DataSetMetrics)

% confusion matrix
figure;
confusionchart(metrics.ConfusionMatrix.Variables,classNames);
title('Improved Model Confusion Matrix');