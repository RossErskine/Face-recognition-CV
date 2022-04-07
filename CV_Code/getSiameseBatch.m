function [X1,X2,pairLabels] = getSiameseBatch(X,Y,miniBatchSize)
% getSiameseBatch returns a randomly selected batch of paired images.
% On average, this function produces a balanced set of similar and
% dissimilar pairs.
% from https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html
pairLabels = zeros(1, miniBatchSize);
imgSize = size(X(:,:,:,1));
X1 = zeros([imgSize 1 miniBatchSize]);
X2 = zeros([imgSize 1 miniBatchSize]);

for i = 1:miniBatchSize
    choice = rand(1);
    if choice < 0.5
        [pairIdx1, pairIdx2, pairLabels(i)] = getSimilarPair(Y);
    else
        [pairIdx1, pairIdx2, pairLabels(i)] = getDissimilarPair(Y);
    end
    X1(:,:,:,i) = X(:,:,:,pairIdx1);
    X2(:,:,:,i) = X(:,:,:,pairIdx2);
end
X1=uint8(X1(:,:,:,1:miniBatchSize));
X2=uint8(X2(:,:,:,1:miniBatchSize));
end