% Load CSV file
dataFile = 'D:\psychiaric\new\All.csv'; % Update with your file path


% Read the table from the CSV file
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
trainingFeatures = features(cv.training, :);
trainingLabels = labels(cv.training);
testFeatures = features(cv.test, :);
testLabels = labels(cv.test);

% Reshape features for ResNet-50 input
% ResNet-50 input size: [224, 224, 3]
inputSize = [224, 224, 3];

% Determine the intermediate shape
intermediateSize = [15, 15, 1]; % For 212 features, 15x15 is the closest square (225), use 1 channel initially

% Initialize empty arrays for the reshaped data
numTrainingSamples = size(trainingFeatures, 1);
numTestSamples = size(testFeatures, 1);
trainingFeaturesReshaped = zeros([inputSize, numTrainingSamples], 'single');
testFeaturesReshaped = zeros([inputSize, numTestSamples], 'single');

% Reshape the features into intermediate size and then resize to 224x224
for i = 1:numTrainingSamples
    img = reshape([trainingFeatures(i, :), zeros(1, 15*15 - size(trainingFeatures, 2))], intermediateSize); % Pad with zeros
    img = imresize(img, [inputSize(1), inputSize(2)]);
    img = repmat(img, [1, 1, inputSize(3)]); % Repeat the grayscale image across 3 channels
    trainingFeaturesReshaped(:, :, :, i) = img;
end

for i = 1:numTestSamples
    img = reshape([testFeatures(i, :), zeros(1, 15*15 - size(testFeatures, 2))], intermediateSize); % Pad with zeros
    img = imresize(img, [inputSize(1), inputSize(2)]);
    img = repmat(img, [1, 1, inputSize(3)]); % Repeat the grayscale image across 3 channels
    testFeaturesReshaped(:, :, :, i) = img;
end

% Load ResNet-50 model
net = resnet50;

% Extract features using ResNet-50
featureLayer = 'avg_pool';

% Convert to dlarray for compatibility
augmentedTrainingSet = dlarray(trainingFeaturesReshaped, 'SSCB');
augmentedTestSet = dlarray(testFeaturesReshaped, 'SSCB');

% Extract training features
trainingFeaturesCNN = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');
% Extract test features
testFeaturesCNN = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');

% Train a classifier (example using SVM)
classifier = fitcecoc(trainingFeaturesCNN', trainingLabels);

% Evaluate classifier
predictedLabels = predict(classifier, testFeaturesCNN');

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('Accuracy of the classifier: %.2f%%\n', accuracy * 100);

% Save classification results if needed
save('genetic_data_classification.mat', 'predictedLabels', 'testLabels', 'accuracy');

% Save the features and labels for future use in Classification Learner
featureTable = array2table([trainingFeaturesCNN', double(trainingLabels)]);
featureTable.Properties.VariableNames = [compose('Feature%d', 1:size(trainingFeaturesCNN, 1)), 'Label'];
save('training_features.mat', 'featureTable');

featureTableTest = array2table([testFeaturesCNN', double(testLabels)]);
featureTableTest.Properties.VariableNames = [compose('Feature%d', 1:size(testFeaturesCNN, 1)), 'Label'];
save('test_features.mat', 'featureTableTest');

% Define paths to SNP data CSV files
bipolarFile = 'D:\psychiaric\new\csv_processed\Bipolar.csv';
controlFile = 'D:\psychiaric\new\csv_processed\Control.csv';
MDDFile = 'D:\psychiaric\new\csv_processed\MDD.csv';

% Load SNP data from CSV files
bipolarData = readtable(bipolarFile);
controlData = readtable(controlFile);
MDDData = readtable(MDDFile);

% Preprocess SNP data
% Assuming the first column is patient ID (if not, adjust accordingly)
bipolarLabels = ones(height(bipolarData), 1); % Label for bipolar
controlLabels = 2 * ones(height(controlData), 1); % Label for control
MDDLabels = 3 * ones(height(MDDData), 1); % Label for MDD

% Combine all data into one matrix (features) and one vector (labels)
allData = [bipolarData{:, 2:end}; controlData{:, 2:end}; MDDData{:, 2:end}];
allLabels = [bipolarLabels; controlLabels; MDDLabels];

% Handle missing values (replace NaN with mean, median, or handle as appropriate)
allData = fillmissing(allData, 'constant', 0); % Example: Replace NaN with 0

% Shuffle data
idx = randperm(length(allLabels));
allData = allData(idx, :);
allLabels = allLabels(idx);

% Split data into training and testing sets
cv = cvpartition(allLabels, 'HoldOut', 0.2); % Hold-out cross-validation
trainingData = allData(cv.training,:);
trainingLabels = allLabels(cv.training);
testData = allData(cv.test,:);
testLabels = allLabels(cv.test);

% Train a classifier (example using SVM)
classifier = fitcecoc(trainingData, trainingLabels);

% Evaluate classifier
predictedLabels = predict(classifier, testData);

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);

fprintf('Accuracy of the classifier: %.2f%%\n', accuracy * 100);

% Save classification results if needed
save('snp_classification_results.mat', 'predictedLabels', 'testLabels', 'accuracy');


%%%%%%%

% Define paths to SNP data CSV files
bipolarFile = 'D:\psychiaric\new\csv_processed\Bipolar.csv';
controlFile = 'D:\psychiaric\new\csv_processed\Control.csv';
MDDFile = 'D:\psychiaric\new\csv_processed\MDD.csv';

% Load SNP data from CSV files
bipolarData = readtable(bipolarFile);
controlData = readtable(controlFile);
MDDData = readtable(MDDFile);

% Preprocess SNP data
% Assuming the first column is patient ID (if not, adjust accordingly)
bipolarLabels = ones(height(bipolarData), 1); % Label for bipolar
controlLabels = 2 * ones(height(controlData), 1); % Label for control
MDDLabels = 3 * ones(height(MDDData), 1); % Label for MDD

% Combine all data into one matrix (features) and one vector (labels)
allData = [bipolarData{:, 2:end}; controlData{:, 2:end}; MDDData{:, 2:end}];
allLabels = [bipolarLabels; controlLabels; MDDLabels];

% Handle missing values (replace NaN with mean, median, or handle as appropriate)
allData = fillmissing(allData, 'constant', 0); % Example: Replace NaN with 0

% Shuffle data
idx = randperm(length(allLabels));
allData = allData(idx, :);
allLabels = allLabels(idx);



% Split data into training and testing sets
cv = cvpartition(allLabels, 'HoldOut', 0.2); % Hold-out cross-validation
trainingData = allData(cv.training,:);
trainingLabels = allLabels(cv.training);
testData = allData(cv.test,:);
testLabels = allLabels(cv.test);

% Train a classifier (example using fitcecoc for SVM)
classifier = fitcecoc(trainingData, trainingLabels);

% Evaluate classifier
predictedLabels = predict(classifier, testData);

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);

fprintf('Accuracy of the classifier: %.2f%%\n', accuracy * 100);

% Save classification results if needed
save('snp_classification_results.mat', 'predictedLabels', 'testLabels', 'accuracy');



featureTable = array2table(extractFeaturesFromData); % Convert features to a table
featureTable.type = categorical([MDDLabels; bipolarLabels; controlLabels]); % Add labels as categorical type

% Save featureTable for later use in Classification Learner or other classifiers
save('features_for_classification.mat', 'featureTable');


