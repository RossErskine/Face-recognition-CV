function loss = contrastiveLoss(F1,F2,pairLabel,margin)
% The contrastiveLoss - function calculates the contrastive loss between
% the reduced features of the paired images
%
% Input: 
%     F1, F2 - A pair of Siamese image encodings that have been pass
%              through the siamese network
%     pairLabels - Siamese batches labels
%     margin - %The margin parameter is used for constraint: if two images 
%               in a pair are dissimilar, then their distance should be at least margin,
%               or a loss will be incurred.
%
% Process:
%     Find the Euclidean disatnec between each Siames image encoding
%     incure loss on similar images, incur loss on disimilar images if
%     within margin threshold others incure a zero loss
%     calcualte loss
%
% Return:
%     overal loss of siames batches
%
% source: https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html

% Define small value to prevent taking square root of 0
delta = 1e-6;

% Find Euclidean distance metric
distances = sqrt(sum((F1 - F2).^2,1) + delta);

% label(i) = 1 if features1(:,i) and features2(:,i) are features
% for similar images, and 0 otherwise
lossSimilar = pairLabel.*(distances.^2);

lossDissimilar = (1 - pairLabel).*(max(margin - distances, 0).^2);

loss = 0.5*sum(lossSimilar + lossDissimilar,"all");

end