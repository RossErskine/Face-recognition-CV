function  [pairIdx1,pairIdx2,pairLabel] = getDissimilarPair(classLabel)
% getDissimilarPair returns a random pair of indices for images
% that are in different classes and the dissimilar pair label = 0.
%
% Input: 
%     class Labels of image set
%
% Process: 
%      find all unique classes, chooses one of those classes randomly making sure there is
%      more than one in each class. then randomly choosing a pair of them.
%
% Return: 
%      A pair of same class iamges and there corresponding labels
%
% Source: https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html

% Find all unique classes.
classes = unique(classLabel);

% Choose two different classes randomly which will be used to get a dissimilar pair.
classesChoice = randperm(numel(classes), 2);

% Find the indices of all the observations from the first and second classes.
idxs1 = find(classLabel==classes(classesChoice(1)));
idxs2 = find(classLabel==classes(classesChoice(2)));


% Randomly choose one image from each class.
% the while loop checks the choice is in the cross-validation partition
pairIdx1Choice = randi(numel(idxs1));
pairIdx2Choice = randi(numel(idxs2));
    
pairIdx1 = idxs1(pairIdx1Choice);
pairIdx2 = idxs2(pairIdx2Choice);
pairLabel = 0;
        
    

end