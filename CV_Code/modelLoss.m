function [loss,gradients] = modelLoss(net,X1,X2,pairLabel,margin)
% The modelLoss function calculates the contrastive loss between the
% paired images and returns the loss and the gradients of the loss with
% respect to the network learnable parameters
% https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html

% Pass first half of image pairs forward through the network
F1 = forward(net,X1);
% Pass second set of image pairs forward through the network
F2 = forward(net,X2);

% Calculate contrastive loss
loss = contrastiveLoss(F1,F2,pairLabel,margin);
% Calculate binary cross-entropy loss.
%loss = binarycrossentropy(Y,pairLabels);


% Calculate gradients of the loss with respect to the network learnable
% parameters
gradients = dlgradient(loss,net.Learnables);
loss = double(loss);
end
