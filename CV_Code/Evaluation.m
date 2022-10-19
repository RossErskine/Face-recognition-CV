clear all;
close all;
trainPath='.\FaceDatabase\stg2_Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here

%% Retrive training and testing images with augmented images included
[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images




%% Now we need to do facial recognition: Baseline Method
%tic;
%   outputID=FaceRecognition(trainImgSet, trainPersonID, testPath);
%runTime=toc
%
%load testLabel
%correctP=0;
%for i=1:size(testLabel,1)
%    if strcmp(outputID(i,:),testLabel(i,:))
%        correctP=correctP+1;
%    end
%end
%recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy

%% Method developed by you
% 
 tic;
   outputIDNew=FaceRecognitionNew(trainImgSet, trainPersonID, testPath);
 methodNewTime=toc
% 
 load testLabel
 correctP=0;
 for i=1:size(testLabel,1)
     if strcmp(outputIDNew(i,:),testLabel(i,:))
         correctP=correctP+1;
     end
 end
 recAccuracyNew=correctP/size(testLabel,1)*100  %Recognition accuracy
 predLabels = convertStringsToChars(outputIDNew);
 predLabels= char(predLabels); 
 confusionchart(testLabel,predLabels)

