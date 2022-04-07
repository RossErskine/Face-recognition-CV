function [augImgSet, augPersonID] = createAugmentedImages(trainImgSet, trainPersonID)
% createAugmentedImages - 
% 

imageSetSize = size(trainImgSet, 4); % used as fixed iterations
augImgSet = zeros(600,600,3,imageSetSize*3);
augPersonID=[];
k=1;
for i=1:imageSetSize
    % add original image
    augImgSet(:,:,:,k) = trainImgSet(:,:,:,i);
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)]; % converts to string so classes can be seperated
    k=k+1;

    % blur image
    sigma = 1+5*rand; 
    blurImg = imgaussfilt(trainImgSet(:,:,:,i),sigma);

    augImgSet(:,:,:,k) = blurImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;

    %  Flip image
    flippedImg = flip(trainImgSet(:,:,:,i), 2);
    augImgSet(:,:,:,k)= flippedImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;
    
end
augImgSet=uint8(augImgSet(:,:,:,1:k-1));