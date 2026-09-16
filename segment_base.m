close all;
clear;
clc;

% data paths and loading image store
imgDir = fullfile('cw_data','images');
labelDir = fullfile('cw_data','segmentation');

imds = imageDatastore(imgDir);

% class definition
classNames = ["background","crop","weed"];

% RGB label IDs:
% background = black
% crop = green
% weed = red

labelIDs = [
    0   0   0;      % background(dirt)
    0 255   0;      % crop
    255 0   0       % weed
];

% ground truth labels
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

% splitting data (80/20)
% 40 training images, 10 validation images
numFiles = numel(imds.Files);
rng(1);

shuffled = randperm(numFiles);

numTrain = round(0.8 * numFiles);

trainIdx = shuffled(1:numTrain);
valIdx   = shuffled(numTrain+1:end);

imdsTrain = subset(imds,trainIdx);
pxdsTrain = subset(pxds,trainIdx);

imdsVal = subset(imds,valIdx);
pxdsVal = subset(pxds,valIdx);

% combine the data
trainingData = pixelLabelImageDatastore(imdsTrain,pxdsTrain,'OutputSize',[240 320]);
validationData = pixelLabelImageDatastore(imdsVal,pxdsVal,'OutputSize',[240 320]);

% network input size
inputSize = [240 320 3];

numClasses = 3;
numFilters = 16;

% main network
layers = [

imageInputLayer(inputSize)

convolution2dLayer(3,16,'Padding','same')
reluLayer

convolution2dLayer(3,16,'Padding','same')
reluLayer

convolution2dLayer(1,numClasses)

softmaxLayer
pixelClassificationLayer
];

% training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',15, ...
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationData, ...
    'VerboseFrequency',10, ...
    'Plots','training-progress');

% TRAIN NETWORK
%net = trainNetwork(trainingData,layers,opts);


% SAVE/LOAD MODEL
%save('segmentnet_base.mat','net');
load('segmentnet_base.mat');

% RUN SEGMENTATION ON VALIDATION SET
pxdsResults = semanticseg(imdsVal,net,'MiniBatchSize',1);

% CHECK IMAGE SIZE VS PREDICTION SIZE
I = readimage(imdsVal,1);
Cpred = readimage(pxdsResults,1);

size(I)
size(Cpred)

% CREATE RESIZED VALIDATION LABELS
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal,'OutputSize',[240 320]);

% calc metrics and evaluation
pred = readimage(pxdsResults,1);
truth = readimage(pximdsVal.PixelLabelDatastore,1);

metrics = evaluateSemanticSegmentation(pxdsResults,pximdsVal.PixelLabelDatastore);

classTable = metrics.ClassMetrics;
disp(classTable(2:3,:))
disp(metrics.DataSetMetrics)

% sample result
Itest = readimage(imdsVal,1);
Cpred = readimage(pxdsResults,1);

figure;
imshow(labeloverlay(Itest,Cpred));
title('Predicted Segmentation');

% confusion matrix
confMat = metrics.ConfusionMatrix.Variables;
figure;
confusionchart(confMat,classNames);
title('Confusion Matrix');