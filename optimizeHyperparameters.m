function [accuracy, net] = optimizeHyperparameters(trainingFeatures, trainingLabels, testFeatures, testLabels, numClasses)
    % Define the layers
    layers = [
        featureInputLayer(size(trainingFeatures, 2))
        fullyConnectedLayer(256)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(128)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(64)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(3) 
        softmaxLayer
        classificationLayer];
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 0.001, ...
        'ValidationData', {testFeatures, testLabels}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress');
    
    % Train the network on combined features
    net = trainNetwork(trainingFeatures, trainingLabels, layers, options);
    
    % Evaluate the network
    predictedLabels = classify(net, testFeatures);
    accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
end
