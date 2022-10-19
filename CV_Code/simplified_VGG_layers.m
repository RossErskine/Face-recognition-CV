function layers = simplified_VGG_layers()
% CNN_Layers - defines CNN layers 
% Input: none
% layers:
%
% Returns:  CNN layers 
%
% Source: https://www.researchgate.net/figure/Simplified-VGG-network-architecture-used-in-this-work-for-global-image-feature-ex_fig2_306186689


layers = [
    imageInputLayer([105 105 1],Normalization="none")

    convolution2dLayer(3,32, "WeightsInitializer",'he')
    reluLayer
    convolution2dLayer(3,32, "WeightsInitializer",'he')
    reluLayer
    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,32, "WeightsInitializer",'he')
    reluLayer
    convolution2dLayer(3,32, "WeightsInitializer",'he')
    reluLayer
    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,64, "WeightsInitializer",'he')
    reluLayer
    convolution2dLayer(3,128, "WeightsInitializer",'he')
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    
    fullyConnectedLayer(256, "WeightsInitializer",'he')
    reluLayer
    fullyConnectedLayer(1024, "WeightsInitializer",'he')
    reluLayer
    fullyConnectedLayer(2048, "WeightsInitializer",'he')];



end