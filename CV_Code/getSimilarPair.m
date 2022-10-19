function [pairIdx1,pairIdx2,pairLabel] = getSimilarPair(classLabel)
% getSimilarPair - returns a random pair of indices for images
% that are in the same class and the similar pair label = 1.
%
% Input:
%      Class labels of image set 
%
% Process: 
%      the while loop guarantees a pair. in the loop we find all unique
%      classes, chooses one of those classes randomly making sure there is
%      more than one in each class. then randomly choosing a pair of them.
% 
% Returns:
%      A pair of same class iamges and there corresponding labels
%
% from https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html

% while loop guarantees a pair
is_moreThanOne =false;
while ~is_moreThanOne
    % Find all unique classes.
    classes = unique(classLabel);
    
    % Choose a class randomly which will be used to get a similar pair.
    classChoice = randi(numel(classes));
    
    % Find the indices of all the observations from the chosen class.
    idxs = find(classLabel==classes(classChoice));
   
    if length(idxs)>1
        is_moreThanOne =true;
    end
end


% Randomly choose two different images from the chosen class.
pairIdxChoice = randperm(numel(idxs),2);
    
pairIdx1 = idxs(pairIdxChoice(1));
pairIdx2 = idxs(pairIdxChoice(2));
pairLabel =1;
       
        
end