function perdictedLabel = getPrediction(testImg, Xtrain, Ytrain, net)
% getPrediction - Compares a single image to a set of images and returns a
% predicted label
% input:
%       testImg:  a single test image to compare to a Set of images
%       Xtrain: a Set of images to compare to a single image
%       Ytrain: Labels of the set of images
%       net: network layers of trained weights for prediction
% process:
%        Pre-process single image and set of images
%        passes single image through the network to get a predicted
%        encoding
%        loops through each image in set and passes it through the network
%        getting a predicted encoding.
%        Each encoded has the measured distance using euclidean distance
%        and added to an array called distance[]
%        the element in distance with the lowest score is returned as
%        the predicted label
% return: predicted label

distances = []; 
delta = 1e-6;

%process testImg
XTest = dlarray(single(testImg),"SSCB");
XTest = gpuArray(XTest);
FTest = predict(net,XTest);

% loop through all train images
% better if they were in a table as to not pass through network each time
for i=1:size(Xtrain,4)
    trainImg = Xtrain(:,:,:,i);
    processedTrainImg = pre_processImage(trainImg);
    % process training images 
    trainImg = dlarray(single( processedTrainImg),"SSCB");
    trainImg = gpuArray(trainImg);
    FTrain = predict(net,trainImg);
    
    % Find Euclidean distance metric
    dist = sqrt(sum((FTest - FTrain).^2,1) + delta);
    distances(i) = dist; 
    

end
%loss = contrastiveLoss(F1,F2,pairLabel,margin);
labelIndx=find(distances==min(distances)); 
perdictedLabel = Ytrain(labelIndx(1),:);

end