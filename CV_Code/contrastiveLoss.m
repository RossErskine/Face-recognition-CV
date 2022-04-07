function loss = contrastiveLoss(F1,F2,pairLabel,margin)
% The contrastiveLoss function calculates the contrastive loss between
% the reduced features of the paired images
% https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html

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