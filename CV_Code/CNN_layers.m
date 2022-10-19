function layers = CNN_layers()
% CNN_Layers - defines CNN layers 
% Input: None 
% layers: 
%        Input layer: 105x105x3,
%        Conv layer 1: 105x105x64, 10x10 window, ReLu activation
%        Max-Pooling layer 1: 52x52x64, 2x2 window, 2x2 stride
%        Conv layer 2: 52x52x128, 7x7 window, Relu activation
%        Max-pooling layer 2: 26x26x128, 2x2 window, 2x2 stride
%        Conv layer 3: 26x26x128, 4x4 window, Relu activation
%        Max-pooling layer 3: 13x13x128, 2x2 window, 2x2 stride
%        Conv layer 4: 13x13x256, 5x5 window, Relu activation
%        Fully conected layer 1: output 4096
%
% Returns: CNN layers 
%
% Source: https://uk.mathworks.com/help/deeplearning/ug/train-a-siamese-network-to-compare-images.html
%

layers = [
    imageInputLayer([105 105 3],Normalization="none")
    convolution2dLayer(10,64,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(7,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(4,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(5,256,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    fullyConnectedLayer(4096,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")];

 