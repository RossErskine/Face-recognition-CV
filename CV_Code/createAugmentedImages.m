function [augImgSet, augPersonID] = createAugmentedImages(trainImgSet, trainPersonID)
% createAugmentedImages - Takes a set of images and returns a set of
% augmented images plus the original 
% Input: 
%      trainImgSet: a Set of images
%      trainPersonId: a set of labels for the set of images
%
% Process:
%      loops through the set of images, adds original to the new set and
%      that an augmented image of blurred, flipped, shear, noise and scaled
%      of the original and adds them to the new augmented set of images
%
% Returns:
%     an enlarged set of augmented plus original set of images
% 
% Source: https://uk.mathworks.com/help/deeplearning/ug/image-augmentation-using-image-processing-toolbox.html
%

imageSetSize = size(trainImgSet, 4); % used as fixed iterations
augImgSet = zeros(105, 105,3,imageSetSize*6);
augPersonID=[];
k=1;

for i=1:imageSetSize
    % add original image
    originalImg(:,:,:,1) = imresize(trainImgSet(:,:,:,i), [105, 105]);
    augImgSet(:,:,:,k) = originalImg(:,:,:,1);
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)]; % converts to string so classes can be seperated
    k=k+1;

    % blur image
    sigma = 1+5*rand; 
    blurImg = imgaussfilt(originalImg,sigma);
    augImgSet(:,:,:,k) = blurImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;

    %  Flip image
    flippedImg = flip(originalImg, 2);
    augImgSet(:,:,:,k)= flippedImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;

    % Shear image
    tform = randomAffine2d(XShear=[-30 30]); 
    outputView = affineOutputView(size(originalImg),tform); 
    shearImg = imwarp(originalImg,tform,OutputView=outputView);
    augImgSet(:,:,:,k)= shearImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;
    

    % noise filter
    noiseFilterImg = imnoise(originalImg,"gaussian");
    augImgSet(:,:,:,k)= noiseFilterImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;
    
    % scale 
    tform = randomAffine2d(Scale=[1.2,1.5]);
    outputView = affineOutputView(size(originalImg),tform);
    scaledImg = imwarp(originalImg,tform,OutputView=outputView);
    augImgSet(:,:,:,k)= scaledImg;
    augPersonID=[convertCharsToStrings(augPersonID); trainPersonID(i,:)];
    k=k+1;
end
augImgSet=uint8(augImgSet(:,:,:,1:k-1)); % 
