function [X1,X2,pairLabels] = getSiameseBatch(X,Y,miniBatchSize, is_imds)
% getSiameseBatch - returns a randomly selected batch of paired images.
% On average, this function produces a balanced set of similar and
% dissimilar pairs.
%
% Input:
%      X: A set of images or a ImageDataStore of images
%      Y Labels of image set
%      MiniBatchSize: sze of minibatch wanting returned example: 32,64,128
%      is_imds: a boolean variable indictating if set is an imageDataStore
%      or not
%
% Process:
%       produces an evenly weighted set of similar and dissimiar images
%       depending on batch size 
%       See getSimilarPair() and getDissimilarPair() for more information
%
% Returns:
%       two evenly distributed Batches of paired similar or dissimiar images 
%
% Source: https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html

pairLabels = zeros(1, miniBatchSize);
if is_imds == true 
    imgSize = size(readimage(X,1));
else
    imgSize = size(X(:,:,:,1));
end

X1 = zeros([imgSize 1 miniBatchSize]);
X2 = zeros([imgSize 1 miniBatchSize]);

for i = 1:miniBatchSize
    choice = rand(1);
    if choice < 0.5
        [pairIdx1, pairIdx2, pairLabels(i)] = getSimilarPair(Y);
    else
        [pairIdx1, pairIdx2, pairLabels(i)] = getDissimilarPair(Y);
    end
    if is_imds == true
        X1(:,:,:,i) = X.readimage(pairIdx1);
        X2(:,:,:,i) = X.readimage(pairIdx2);
    else
        X1(:,:,:,i) = X(:,:,:,pairIdx1);
        X2(:,:,:,i) = X(:,:,:,pairIdx2);
    end
end

X1=uint8(X1(:,:,:,1:miniBatchSize));
X2=uint8(X2(:,:,:,1:miniBatchSize));

end