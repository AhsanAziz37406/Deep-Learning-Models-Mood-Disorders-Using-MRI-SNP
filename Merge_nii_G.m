% Load preprocessed SNP data from CSV
dataFile = 'D:\psychiaric\new\All.csv'; % Update with your file path
dataTable = readtable(dataFile);

% Extract Unique IDs, Labels, and Features
uniqueIDs = dataTable{:, 1}; % Unique IDs (not used further)
labels = dataTable{:, 2}; % Labels
features = dataTable{:, 3:end}; % Numeric features

% Normalize features if needed (example: scale to [0, 1])
features = normalize(features);

% Convert labels to categorical type
labels = categorical(labels);

% Split data into training and test sets (80:20 ratio)
cv = cvpartition(labels, 'HoldOut', 0.2);
trainingFeaturesSNP = features(cv.training, :);
trainingLabelsSNP = labels(cv.training);
testFeaturesSNP = features(cv.test, :);
testLabelsSNP = labels(cv.test);

% Convert SNP features to tables
featureTableTrainSNP = array2table(trainingFeaturesSNP);
featureTableTestSNP = array2table(testFeaturesSNP);


net = inceptionv3;

% Define the root folder and subfolders
rootFolder = 'D:\psychiaric\new'; % Replace with your root folder path

% Define the categories (subfolders)
categories = {'Bipolar', 'Control', 'MDD'};

% Load data from each category folder
imds = imageDatastore(fullfile(rootFolder, 'Processed_data', categories), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.nii'}, ...
    'ReadFcn', @loadNiftiFile);

tbl = countEachLabel(imds);
minSetCount = min(tbl{:, 2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Prepare Training and Test Image Sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;
featureLayer = 'avg_pool';

trainingFeaturesNii = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeaturesNii = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');

% Convert .nii features to tables
featureTableTrainNii = array2table(trainingFeaturesNii');
featureTableTestNii = array2table(testFeaturesNii');


% Ensure the number of rows match between SNP and .nii features
minTrainRows = min(size(trainingFeaturesSNP, 1), size(trainingFeaturesNii, 2));
minTestRows = min(size(testFeaturesSNP, 1), size(testFeaturesNii, 2));

trainingFeaturesSNP = trainingFeaturesSNP(1:minTrainRows, :);
trainingFeaturesNii = trainingFeaturesNii(:, 1:minTrainRows)';

testFeaturesSNP = testFeaturesSNP(1:minTestRows, :);
testFeaturesNii = testFeaturesNii(:, 1:minTestRows)';

% Combine SNP and .nii features
combinedTrainingFeatures = [trainingFeaturesSNP, trainingFeaturesNii];
combinedTestFeatures = [testFeaturesSNP, testFeaturesNii];

% Normalize combined features
combinedTrainingFeatures = normalize(combinedTrainingFeatures);
combinedTestFeatures = normalize(combinedTestFeatures);

% Convert combined features to tables
featureTableTrain = array2table(combinedTrainingFeatures);
featureTableTest = array2table(combinedTestFeatures);

% Add labels to the combined feature tables
featureTableTrain.type = trainingLabelsSNP(1:minTrainRows);
featureTableTest.type = testLabelsSNP(1:minTestRows);

% Determine the number of classes
numClasses = numel(categories(trainingLabelsSNP));

%
% layers = [
%     featureInputLayer(size(combinedTrainingFeatures, 2))
% 
%     % Repeat blocks to create a deep network
%     repmat([
%         fullyConnectedLayer(256)
%         batchNormalizationLayer
%         reluLayer
%         dropoutLayer(0.5)
%     ], 25, 1) % Repeat the block 25 times to exceed 100 layers
% 
%     % Last few layers
%     fullyConnectedLayer(128)
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(64)
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(3)
%     softmaxLayer
%     classificationLayer];
% 
% % Training options with different optimization algorithm 
% validationPatience = 10;
% options = trainingOptions('sgdm', ...
%     'MaxEpochs', 50, ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', 0.1, ...
%     'ValidationData', {combinedTestFeatures, testLabelsSNP(1:minTestRows)}, ...
%     'ValidationFrequency', 30, ...
%     'Verbose', false, ...
%     'Plots', 'training-progress', ...
%     'Shuffle', 'every-epoch', ...
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.01, ...
%     'LearnRateDropPeriod', 10, ...
%     'ExecutionEnvironment', 'auto', ...
%     'ValidationPatience', validationPatience);
% 
% % Train the network
% net = trainNetwork(combinedTrainingFeatures, trainingLabelsSNP(1:minTrainRows), layers, options);

layers = [
    featureInputLayer(size(combinedTrainingFeatures, 2))
    
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

% Training options with different optimization algorithm (SGD with momentum)
options = trainingOptions('sgdm', ...  % Change to 'sgdm'
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'ValidationData', {combinedTestFeatures, testLabelsSNP(1:minTestRows)}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(combinedTrainingFeatures, trainingLabelsSNP(1:minTrainRows), layers, options);

% Evaluate the network
predictedLabels = classify(net, combinedTestFeatures);
accuracy = sum(predictedLabels == testLabelsSNP(1:minTestRows)) / numel(testLabelsSNP(1:minTestRows));
fprintf('Accuracy of the classifier on combined features: %.2f%%\n', accuracy * 100);

% Save the trained network and results
save('M_Nii_G_201_classification.mat', 'net', 'predictedLabels', 'testLabelsSNP', 'accuracy');

% Save combined features for future use
XTrain = combinedTrainingFeatures;
XTest = combinedTestFeatures;
YTrain = trainingLabelsSNP(1:minTrainRows);
YTest = testLabelsSNP(1:minTestRows);

save('M_Nii_G_D201.mat', 'XTrain', 'XTest', 'YTrain', 'YTest');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GA
% % Load preprocessed SNP data from CSV
% dataFile = 'D:\psychiaric\new\All.csv'; % Update with your file path
% dataTable = readtable(dataFile);
% 
% % Extract Unique IDs, Labels, and Features
% uniqueIDs = dataTable{:, 1}; % Unique IDs (not used further)
% labels = dataTable{:, 2}; % Labels
% features = dataTable{:, 3:end}; % Numeric features
% 
% % Normalize features if needed (example: scale to [0, 1])
% features = normalize(features);
% 
% % Convert labels to categorical type
% labels = categorical(labels);
% 
% % Split data into training and test sets (80:20 ratio)
% cv = cvpartition(labels, 'HoldOut', 0.2);
% trainingFeaturesSNP = features(cv.training, :);
% trainingLabelsSNP = labels(cv.training);
% testFeaturesSNP = features(cv.test, :);
% testLabelsSNP = labels(cv.test);
% 
% % Convert SNP features to tables
% featureTableTrainSNP = array2table(trainingFeaturesSNP);
% featureTableTestSNP = array2table(testFeaturesSNP);
% 
% net = densenet201;
% 
% % Define the root folder and subfolders
% rootFolder = 'D:\psychiaric\new'; % Replace with your root folder path
% 
% % Define the categories (subfolders)
% categories = {'Bipolar', 'Control', 'MDD'};
% 
% % Load data from each category folder
% imds = imageDatastore(fullfile(rootFolder, 'Processed_data', categories), ...
%     'LabelSource', 'foldernames', ...
%     'IncludeSubfolders', true, ...
%     'FileExtensions', {'.nii'}, ...
%     'ReadFcn', @loadNiftiFile);
% 
% tbl = countEachLabel(imds);
% minSetCount = min(tbl{:, 2});
% imds = splitEachLabel(imds, minSetCount, 'randomize');
% 
% % Prepare Training and Test Image Sets
% [trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
% 
% imageSize = net.Layers(1).InputSize;
% augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
% augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
% 
% trainingLabels = trainingSet.Labels;
% testLabels = testSet.Labels;
% featureLayer = 'avg_pool';
% 
% trainingFeaturesNii = activations(net, augmentedTrainingSet, featureLayer, ...
%     'MiniBatchSize', 64, 'OutputAs', 'columns');
% testFeaturesNii = activations(net, augmentedTestSet, featureLayer, ...
%     'MiniBatchSize', 64, 'OutputAs', 'columns');
% 
% % Convert .nii features to tables
% featureTableTrainNii = array2table(trainingFeaturesNii');
% featureTableTestNii = array2table(testFeaturesNii');
% 
% % Ensure the number of rows match between SNP and .nii features
% minTrainRows = min(size(trainingFeaturesSNP, 1), size(trainingFeaturesNii, 2));
% minTestRows = min(size(testFeaturesSNP, 1), size(testFeaturesNii, 2));
% 
% trainingFeaturesSNP = trainingFeaturesSNP(1:minTrainRows, :);
% trainingFeaturesNii = trainingFeaturesNii(:, 1:minTrainRows)';
% 
% testFeaturesSNP = testFeaturesSNP(1:minTestRows, :);
% testFeaturesNii = testFeaturesNii(:, 1:minTestRows)';
% 
% % Combine SNP and .nii features
% combinedTrainingFeatures = [trainingFeaturesSNP, trainingFeaturesNii];
% combinedTestFeatures = [testFeaturesSNP, testFeaturesNii];
% 
% % Normalize combined features
% combinedTrainingFeatures = normalize(combinedTrainingFeatures);
% combinedTestFeatures = normalize(combinedTestFeatures);
% 
% % Convert combined features to tables
% featureTableTrain = array2table(combinedTrainingFeatures);
% featureTableTest = array2table(combinedTestFeatures);
% 
% % Add labels to the combined feature tables
% featureTableTrain.type = trainingLabelsSNP(1:minTrainRows);
% featureTableTest.type = testLabelsSNP(1:minTestRows);
% 
% % Determine the number of classes
% numClasses = numel(categories(trainingLabelsSNP));
% 
% % Optimize hyperparameters using Genetic Algorithm
% optimalParams = genetic_algorithm();
% 
% % Extract optimized parameters
% fullyConnectedLayer1Size = round(optimalParams(1));
% fullyConnectedLayer2Size = round(optimalParams(2));
% fullyConnectedLayer3Size = round(optimalParams(3));
% initialLearnRate = optimalParams(4);
% 
% % Define the layers with optimized parameters
% layers = [
%     featureInputLayer(size(combinedTrainingFeatures, 2))
%     fullyConnectedLayer(fullyConnectedLayer1Size)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(fullyConnectedLayer2Size)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(fullyConnectedLayer3Size)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(3)
%     softmaxLayer
%     classificationLayer];
% 
% % Training options with optimized initial learn rate
% options = trainingOptions('sgdm', ...
%     'MaxEpochs', 50, ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', initialLearnRate, ...
%     'ValidationData', {combinedTestFeatures, testLabelsSNP(1:minTestRows)}, ...
%     'ValidationFrequency', 30, ...
%     'Verbose', false, ...
%     'Plots', 'training-progress');
% 
% % Train the network with optimized hyperparameters
% net = trainNetwork(combinedTrainingFeatures, trainingLabelsSNP(1:minTrainRows), layers, options);
% 
% % Evaluate the network
% predictedLabels = classify(net, combinedTestFeatures);
% accuracy = sum(predictedLabels == testLabelsSNP(1:minTestRows)) / numel(testLabelsSNP(1:minTestRows));
% fprintf('Accuracy of the classifier on combined features: %.2f%%\n', accuracy * 100);
% 
% 
% %%%%%%%%%%%%%%%%
% 
% 
% % Load preprocessed SNP data from CSV
% dataFile = 'D:\psychiaric\new\All.csv'; % Update with your file path
% dataTable = readtable(dataFile);
% 
% % Extract Unique IDs, Labels, and Features
% uniqueIDs = dataTable{:, 1}; % Unique IDs (not used further)
% labels = dataTable{:, 2}; % Labels
% features = dataTable{:, 3:end}; % Numeric features
% 
% % Normalize features if needed (example: scale to [0, 1])
% features = normalize(features);
% 
% % Convert labels to categorical type
% labels = categorical(labels);
% 
% % Split data into training and test sets (80:20 ratio)
% cv = cvpartition(labels, 'HoldOut', 0.2);
% trainingFeaturesSNP = features(cv.training, :);
% trainingLabelsSNP = labels(cv.training);
% testFeaturesSNP = features(cv.test, :);
% testLabelsSNP = labels(cv.test);
% 
% % Convert SNP features to tables (if needed)
% featureTableTrainSNP = array2table(trainingFeaturesSNP);
% featureTableTestSNP = array2table(testFeaturesSNP);
% 
% % Example neural network initialization (replace with your actual network setup)
% net = alexnet; % Example network
% 
% % Define the root folder and subfolders
% rootFolder = 'D:\psychiaric\new'; % Replace with your root folder path
% 
% % Define the categories (subfolders)
% categories = {'Bipolar', 'Control', 'MDD'};
% 
% % Example image data loading (replace with your actual data loading)
% imds = imageDatastore(fullfile(rootFolder, 'Processed_data', categories), ...
%     'LabelSource', 'foldernames', ...
%     'IncludeSubfolders', true, ...
%     'FileExtensions', {'.nii'}, ...
%     'ReadFcn', @loadNiftiFile);
% 
% tbl = countEachLabel(imds);
% minSetCount = min(tbl{:, 2});
% imds = splitEachLabel(imds, minSetCount, 'randomize');
% 
% % Prepare Training and Test Image Sets (example)
% [trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
% 
% imageSize = net.Layers(1).InputSize;
% augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
% augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
% 
% % Example feature extraction using activations (replace with your actual feature extraction)
% trainingFeaturesNii = activations(net, augmentedTrainingSet, 'pool5', ...
%     'MiniBatchSize', 64, 'OutputAs', 'columns');
% testFeaturesNii = activations(net, augmentedTestSet, 'pool5', ...
%     'MiniBatchSize', 64, 'OutputAs', 'columns');
% 
% % Convert .nii features to tables (if needed)
% featureTableTrainNii = array2table(trainingFeaturesNii');
% featureTableTestNii = array2table(testFeaturesNii');
% 
% % Ensure the number of rows match between SNP and .nii features
% minTrainRows = min(size(trainingFeaturesSNP, 1), size(trainingFeaturesNii, 2));
% minTestRows = min(size(testFeaturesSNP, 1), size(testFeaturesNii, 2));
% 
% trainingFeaturesSNP = trainingFeaturesSNP(1:minTrainRows, :);
% trainingFeaturesNii = trainingFeaturesNii(:, 1:minTrainRows)';
% 
% testFeaturesSNP = testFeaturesSNP(1:minTestRows, :);
% testFeaturesNii = testFeaturesNii(:, 1:minTestRows)';
% 
% % Combine SNP and .nii features
% combinedTrainingFeatures = [trainingFeaturesSNP, trainingFeaturesNii];
% combinedTestFeatures = [testFeaturesSNP, testFeaturesNii];
% 
% % Normalize combined features
% combinedTrainingFeatures = normalize(combinedTrainingFeatures);
% combinedTestFeatures = normalize(combinedTestFeatures);
% 
% % Convert combined features to tables (if needed)
% featureTableTrain = array2table(combinedTrainingFeatures);
% featureTableTest = array2table(combinedTestFeatures);
% 
% % Add labels to the combined feature tables
% featureTableTrain.type = trainingLabelsSNP(1:minTrainRows);
% featureTableTest.type = testLabelsSNP(1:minTestRows);
% 
% % Determine the number of classes
% numClasses = numel(categories(trainingLabelsSNP));
% 
% % Optimize hyperparameters using Genetic Algorithm
% optimalParams = genetic_algorithm(combinedTrainingFeatures, combinedTestFeatures, trainingLabelsSNP, testLabelsSNP, minTestRows, numClasses);
% 
% % Extract optimized parameters
% fullyConnectedLayer1Size = round(optimalParams(1));
% fullyConnectedLayer2Size = round(optimalParams(2));
% fullyConnectedLayer3Size = round(optimalParams(3));
% initialLearnRate = optimalParams(4);
% 
% % Define the layers with optimized parameters
% layers = [
%     featureInputLayer(size(combinedTrainingFeatures, 2))
%     fullyConnectedLayer(fullyConnectedLayer1Size)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(fullyConnectedLayer2Size)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(fullyConnectedLayer3Size)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(3)
%     softmaxLayer
%     classificationLayer
% ];
% 
% % Training options with optimized initial learn rate
% options = trainingOptions('sgdm', ...
%     'MaxEpochs', 50, ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', initialLearnRate, ...
%     'ValidationData', {combinedTestFeatures, testLabelsSNP(1:minTestRows)}, ...
%     'ValidationFrequency', 30, ...
%     'Verbose', false, ...
%     'Plots', 'training-progress');
% 
% % Train the network with optimized hyperparameters
% net = trainNetwork(combinedTrainingFeatures, trainingLabelsSNP(1:minTrainRows), layers, options);
% 
% % Evaluate the network
% predictedLabels = classify(net, combinedTestFeatures);
% accuracy = sum(predictedLabels == testLabelsSNP(1:minTestRows)) / numel(testLabelsSNP(1:minTestRows));
% fprintf('Accuracy of the classifier on combined features: %.2f%%\n', accuracy * 100);
% %%%%%%%%%%%%%
% % Load preprocessed SNP data from CSV
% dataFile = 'D:\psychiaric\new\All.csv'; % Update with your file path
% dataTable = readtable(dataFile);
% 
% % Extract Unique IDs, Labels, and Features
% uniqueIDs = dataTable{:, 1}; % Unique IDs (not used further)
% labels = dataTable{:, 2}; % Labels
% features = dataTable{:, 3:end}; % Numeric features
% 
% % Normalize features if needed (example: scale to [0, 1])
% features = normalize(features);
% 
% % Convert labels to categorical type
% labels = categorical(labels);
% 
% % Split data into training and test sets (80:20 ratio)
% cv = cvpartition(labels, 'HoldOut', 0.2);
% trainingFeaturesSNP = features(cv.training, :);
% trainingLabelsSNP = labels(cv.training);
% testFeaturesSNP = features(cv.test, :);
% testLabelsSNP = labels(cv.test);
% 
% % Combine SNP and .nii features (example)
% combinedTrainingFeatures = trainingFeaturesSNP;  % Replace with your combined features
% combinedTestFeatures = testFeaturesSNP;  % Replace with your combined features
% 
% % Ensure the number of rows match between SNP and .nii features (example)
% minTestRows = min(size(testFeaturesSNP, 1), size(testFeaturesSNP, 2));
% 
% % Run genetic algorithm to optimize parameters
% optimalParams = genetic_algorithm(combinedTrainingFeatures, combinedTestFeatures, trainingLabelsSNP, testLabelsSNP, minTestRows);
% 
% % Extract optimized parameters (example)
% fullyConnectedLayer1Size = round(optimalParams(1));
% fullyConnectedLayer2Size = round(optimalParams(2));
% fullyConnectedLayer3Size = round(optimalParams(3));
% initialLearnRate = optimalParams(4);
% 
% % Define layers with optimized parameters
% layers = [
%     featureInputLayer(size(combinedTrainingFeatures, 2))
%     fullyConnectedLayer(fullyConnectedLayer1Size)
%     reluLayer
%     fullyConnectedLayer(fullyConnectedLayer2Size)
%     reluLayer
%     fullyConnectedLayer(fullyConnectedLayer3Size)
%     reluLayer
%     fullyConnectedLayer(3) % Output layer matching numClasses
%     softmaxLayer
%     classificationLayer
% ];
% 
% % Training options with optimized initial learn rate
% options = trainingOptions('sgdm', ...
%     'MaxEpochs', 50, ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', initialLearnRate, ...
%     'ValidationData', {combinedTestFeatures, testLabelsSNP(1:minTestRows)}, ...
%     'ValidationFrequency', 30, ...
%     'Verbose', false, ...
%     'Plots', 'training-progress');
% 
% % Train the network with optimized hyperparameters
% net = trainNetwork(combinedTrainingFeatures, trainingLabelsSNP(1:minTestRows), layers, options);
% 
% % Evaluate the network
% predictedLabels = classify(net, combinedTestFeatures);
% accuracy = sum(predictedLabels == testLabelsSNP(1:minTestRows)) / numel(testLabelsSNP(1:minTestRows));
% fprintf('Accuracy of the classifier on combined features: %.2f%%\n', accuracy * 100);
