function [optimalParams] = genetic_algorithm(combinedTrainingFeatures, combinedTestFeatures, trainingLabelsSNP, testLabelsSNP, minTestRows, numClasses)
    % Define the parameter bounds
    paramBounds = [...
        16, 512;  % fullyConnectedLayer1Size
        16, 256;  % fullyConnectedLayer2Size
        16, 128;  % fullyConnectedLayer3Size
        0.001, 0.1  % InitialLearnRate
    ];

    % Genetic algorithm settings
    options = optimoptions('ga', ...
        'MaxGenerations', 50, ...
        'PopulationSize', 20, ...
        'FunctionTolerance', 1e-4, ...
        'UseParallel', true, ...
        'Display', 'iter');

    % Pass additional variables to the fitness function
    fitnessFunction = @(params) evaluateNetwork(params, combinedTrainingFeatures, combinedTestFeatures, trainingLabelsSNP, testLabelsSNP, minTestRows, numClasses);

    % Run genetic algorithm
    [optimalParams, ~] = ga(fitnessFunction, size(paramBounds, 1), [], [], [], [], paramBounds(:, 1), paramBounds(:, 2), [], options);
end

function fitness = evaluateNetwork(params, combinedTrainingFeatures, combinedTestFeatures, trainingLabelsSNP, testLabelsSNP, minTestRows, numClasses)
    % Extract the parameters
    fullyConnectedLayer1Size = round(params(1));
    fullyConnectedLayer2Size = round(params(2));
    fullyConnectedLayer3Size = round(params(3));
    initialLearnRate = params(4);

    % Define the layers
    layers = [
        featureInputLayer(size(combinedTrainingFeatures, 2))
        fullyConnectedLayer(fullyConnectedLayer1Size)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(fullyConnectedLayer2Size)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(fullyConnectedLayer3Size)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(3) % Output layer matching numClasses
        softmaxLayer
        classificationLayer
    ];

    % Training options
    options = trainingOptions('sgdm', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', initialLearnRate, ...
        'ValidationData', {combinedTestFeatures, testLabelsSNP(1:minTestRows)}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'none');

    % Train the network
    net = trainNetwork(combinedTrainingFeatures, trainingLabelsSNP(1:minTestRows), layers, options);

    % Evaluate the network
    predictedLabels = classify(net, combinedTestFeatures);
    accuracy = sum(predictedLabels == testLabelsSNP(1:minTestRows)) / numel(testLabelsSNP(1:minTestRows));

    % Fitness value is the negative accuracy (minimize error)
    fitness = -accuracy;
end
