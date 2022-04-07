function [layers, options] = CNN_layers()
% CNN_Layers - defines CNN layers 
% Input: none
% layers:
%
% Returns Layer graph of CNN layers 

options = trainingOptions("sgdm");

layers = [
    imageInputLayer([600 600 3],Normalization="none")
    convolution2dLayer(10,64,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(7,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(4,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    
    fullyConnectedLayer(524,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")];

%layer_graph = layerGraph(layers); %TODO... for customising back
% propagation 