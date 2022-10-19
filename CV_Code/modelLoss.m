function [loss,gradients] = modelLoss(net,X1,X2,pairLabel,margin)
% The modelLoss function calculates the contrastive loss between the
% paired images and returns the loss and the gradients of the loss with
% respect to the network learnable parameters
%
% Input: 
%      net- the network that contrastive loss is to be calcuated on
%      X1, X2 - Siamese batches of similar and disimilar images see
%               getSiameseBatch()
%      PairLables - labels of siamese batches
%      Margin - see contrastiveLoss()
%
% Process:
%      pre-process each set of siamese batches and turn them into
%      "SSCB" (spatial, spatial, channel, batch) for image data, and then
%      transform into gpu array for use of gpu, Pass siamese batches
%      through siamese network.
%      calculate contrasyive loss see contrastiveLoss()
%      calcualte gradient
% 
% Return: 
%      loss and gradient
%
% source: https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html
%


% Pass first half of image pairs forward through the network
% 
           
X1=pre_processImage(X1);
X1 = dlarray(single(X1),"SSCB");
X1 = gpuArray(X1);
F1 = forward(net,X1);

% Pass second set of image pairs forward through the network
X2=pre_processImage(X2);
X2 = dlarray(single(X2),"SSCB");
X2 = gpuArray(X2);
F2 = forward(net,X2);

% Calculate contrastive loss
loss = contrastiveLoss(F1,F2,pairLabel,margin);

% Calculate gradients of the loss with respect to the network learnable
% parameters
gradients = dlgradient(loss,net.Learnables);
loss = double(loss);
end
