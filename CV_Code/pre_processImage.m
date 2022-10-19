function processedImageSet = pre_processImage(trainImgSet)
% pre_processImage - takes a set of images crops each image by enphasizing  
% on the face using cascade object detector, then resize and
% normalise each image
% input: 
%       a set of images
%
% Process:
%        loops through each image, using cascade object detection, finds
%        all faces in the image and returns a set of coordinates of each
%        image the first image in the set is the most central in the image,
%        that image is then cropped using coordinates found, if no face are
%        found return image un-cropped, resize the image to desired size
%        then normalised using min-max normalisation.
%       
%
% Returns:
%        a set of processed images
%
% Source: https://uk.mathworks.com/help/vision/ref/vision.cascadeobjectdetector-system-object.html

targetSize = [105, 105];

imageSetSize = size(trainImgSet,4);

processedImageSet = zeros(105,105, 3, imageSetSize);

% Viola Jones Cascade object detector
faceDetect = vision.CascadeObjectDetector;


for i=1:imageSetSize
    img = trainImgSet(:,:,:,i);
    bbox =faceDetect(img);
    if ~isempty(bbox)
        cropedImg = imcrop(img, bbox(1,:));
        rescaledImg = imresize(cropedImg ,targetSize);
        
    else
        rescaledImg = imresize(img ,targetSize);
    end
    
    processedImageSet(:,:,:,i) = im2double(rescaledImg); % min - max normalisation
    
    %processedImageSet(:,:,:,i) = rgb2gray(rescaledImg); % transform to grayscale
    %processedImageSet(:,:,:,i) = double(processedImageSet(:,:,:,i))/255'; 
    
end




