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

net = densenet201;

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

% Save combined features for future use
XTrain = combinedTrainingFeatures;
XTest = combinedTestFeatures;
YTrain = trainingLabelsSNP(1:minTrainRows);
YTest = testLabelsSNP(1:minTestRows);

save('combined_featuresSNP_Nii.mat', 'XTrain', 'XTest', 'YTrain', 'YTest');

