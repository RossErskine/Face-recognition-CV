function outputIDNew=FaceRecognitionNew(trainImgSet, trainPersonID, testPath)% include testPath later
%% Face recognition 
% Input: 
% Processes:
% Return:
%



%% --------------------------------------------- CNN model --------------------------------- %%
layers = CNN_layers(); % get layer graph and model options
%4layers = simplified_VGG_layers();
%layers = alexNetCNN();
net = dlnetwork(layers); % enables custom training 

%% ----------------------------------- Satge one training ---------------------------------- %%
% comment out if not wanting to train
%net = stg1_training(net);
load ('Net2.mat', 'Net2');
net = Net2;


%% ---------------------------------- Stage two training ----------------------------------- %%
% comment out if not wanting to train
%net = stg2_training(net,trainImgSet, trainPersonID);
load ('stg2_train.mat', 'stg2_net');
net = stg2_net;

%% -------------------------------  testing model --------------------------------- %%
% load test set
outputIDNew=[];
testImgNames=dir([testPath,'*.jpg']);
ptrain = pre_processImage(trainImgSet); 

for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:).name]);%load one of the test images
    testImg = pre_processImage(testImg);

    outputIDNew = [outputIDNew; getPrediction(testImg, ptrain, trainPersonID, net)];

end


