function net = stg2_training(net,trainImgSet, trainPersonID )

%% Global varaibles (should be in capital letters)
numIterations = 3000;
miniBatchSize = 32;
numEpochs = 10;
learningRate = 1e-5;
trailingAvg = [];
trailingAvgSq = [];
gradDecay = 0.9;
gradDecaySq = 0.99;
margin = 0.3;%The margin parameter is used for constraint: if two images 
% in a pair are dissimilar, then their distance should be at least margin,
% or a loss will be incurred.

%% create augmented images 
%processedImgs = pre_processImage(trainImgSet);
[Xtrain, Ytrain] = createAugmentedImages(trainImgSet, trainPersonID);


% plot training loss
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iterations")
ylabel("Loss")
grid on

start = tic;


is_imds = false;
% Loop over mini-batches.   
for iterations = 1:15000

    % Extract mini-batch of image pairs and pair labels for training 
    [X1,X2,pairLabels] = getSiameseBatch(Xtrain, Ytrain,miniBatchSize,is_imds);
   
    
    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    [loss,gradients] = dlfeval(@modelLoss,net,X1,X2,pairLabels,margin);
    

    % Update the Siamese network parameters.
    [net.Learnables,trailingAvg,trailingAvgSq] = ...
        adamupdate(net.Learnables,gradients, ...
        trailingAvg,trailingAvgSq,iterations,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    loss = extractdata(loss); %extracts data from dlarray in double
    addpoints(lineLossTrain,iterations,loss)
    %addpoints(lineLossVal,i,loss)
    title("Elapsed: " + string(D))

    drawnow
    
end

stg2_net = net;
save ('stg2_train.mat', 'stg2_net');