function outputIDNew=FaceRecognitionNew(trainImgSet, trainPersonID, testPath)% include testPath later
%% Face recognition 
% Input: 
% Processes:
% Return:
%
%% Globals 
numIterations = 300;
miniBatchSize = 10;
learningRate = 1e-4;
trailingAvg = [];
trailingAvgSq = [];
gradDecay = 0.9;
gradDecaySq = 0.99;
margin = 0.3;%The margin parameter is used for constraint: if two images 
% in a pair are dissimilar, then their distance should be at least margin,
% or a loss will be incurred.
%% pre-processing
% TODO PREPROCESSING (grayscale, normaliastion)

%% create augmented images 
[Xtrain, Ytrain] = createAugmentedImages(trainImgSet, trainPersonID);

%% cross-validation
% TODO POSSIBLY

%% train model
[layer_graph, options] = CNN_layers(); % get layer graph and model options
net = dlnetwork(layer_graph); % enables custom training 

% plot training loss
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iterations")
ylabel("Loss")
grid on

start = tic;

% Loop over mini-batches.
for iteration = 1:numIterations

    % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabels] = getSiameseBatch(Xtrain, Ytrain,miniBatchSize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch) for image data
    X1 = dlarray(single(X1),"SSCB");
    X2 = dlarray(single(X2),"SSCB");

    % If training on a GPU, then convert data to gpuArray.
    
    X1 = gpuArray(X1);
    X2 = gpuArray(X2);
    

    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    % function listed at the end of the example.
    [loss,gradients] = dlfeval(@modelLoss,net,X1,X2,pairLabels,margin);

    % Update the Siamese network parameters.
    [net.Learnables,trailingAvg,trailingAvgSq] = ...
        adamupdate(net.Learnables,gradients, ...
        trailingAvg,trailingAvgSq,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    loss = extractdata(loss); %extracts data from dlarray in double
    addpoints(lineLossTrain,iteration,loss)
    title("Elapsed: " + string(D))

    drawnow
end

%% test model
% load test set
outputIDNew=[];
testImgNames=dir([testPath,'*.jpg']);

for i=1:size(testImgNames,1)
    % TODO PREPROCESSING (grayscale, normaliastion)
    testImg=imread([testPath, testImgNames(i,:).name]);%load one of the test images
    XTest = dlarray(single(testImg),"SSCB");
    XTest = gpuArray(XTest);
    FTest = predict(net,XTest);
    labelIndx=find(FTest==max(FTest));    % TODO SORT PREDICTION VALUES 
    outputID=[outputID; trainPersonID(labelIndx(1),:)];

end
