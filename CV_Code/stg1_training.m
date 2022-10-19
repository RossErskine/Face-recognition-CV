function net = stg1_training(net)

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


%% ---------------------------------------------  Stage one training --------------------------- %%

stg1_trainPath='.\FaceDatabase\stg1_Train\'; 
imds = imageDatastore(fullfile(stg1_trainPath),...
"IncludeSubfolders",true,"FileExtensions",".jpg","LabelSource","foldernames"); % using datastore 




% plot training loss
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
marklossVal = animatedline('Marker','o');
ylim([0 inf])
xlabel("Iterations")
ylabel("Loss")
 
grid on

start = tic;
is_imds = true;
iterations=0;
for epoch=1:numEpochs
    [trainImds, valImds]= splitEachLabel(imds,0.9,'randomized');
    for i = 1:numIterations
            iterations = iterations +1;
            % Extract mini-batch of image pairs and pair labels for training 
            [X1,X2,pairLabels] = getSiameseBatch(trainImds,trainImds.Labels,miniBatchSize,is_imds);
            
            
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
    
    [V1,V2,pairLabels] = getSiameseBatch(valImds,valImds.Labels,miniBatchSize,is_imds);
           
    
    [loss, na_gradients] = dlfeval(@modelLoss,net,V1,V2,pairLabels,margin);
    lossVal = extractdata(loss);
    addpoints(marklossVal,iterations,lossVal)
    drawnow
    correct=0;
   

end
    
Net2 = net;
save ('Net2.mat', 'Net2');