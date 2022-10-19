function layers = alexNetCNN()
% alexNetCNN - the archecure layers of alexnet CNN
% Input:
% output:


% pre-traind alexnet
net = alexnet;
% The last three layers of the pretrained network net are configured for
% 1000 classes. These three layers must be fine-tuned for the
% new classification problem. Extract all layers, except the last three,
% from the pretrained network.
layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer
    
    
    fullyConnectedLayer(1024,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)

    ];
