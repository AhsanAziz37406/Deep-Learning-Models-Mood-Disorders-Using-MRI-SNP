
% Load preprocessed SNP data from CSV
dataFile = 'D:\psychiaric\new\Datasets\SNP_Abnormal&normal.csv'; % Update with your file path
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

%%%%%%%

net = resnet18;
% Define the root folder and subfolders
rootFolder = 'D:\psychiaric\new\Datasets'; % Replace with your root folder path

% Define the categories (subfolders)
categories = {'Normal', 'Abnormal'};

% % Load data from each category folder
% imds = imageDatastore(fullfile(rootFolder, categories), ...
%     'LabelSource', 'foldernames', ...
%     'FileExtensions', '.nii', ...
%     'ReadFcn', @loadNiftiFile);
imds = imageDatastore(fullfile(rootFolder, 'Adult_Normal&Abnormal.nii', categories), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.nii'}, ...
    'ReadFcn', @loadNiftiFile);


tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

%%
%   Prepare Training and Test Image Sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
 
imageSize = net.Layers(1).InputSize
% imageSize = net.meta.normalization.imageSize(1:3)
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');


trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;
featureLayer = 'pool5';

tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeatures_EV_R18_Adult.mat
feat = testFeatures';
feat = double(feat);
label = testLabels;
label = double(label);
ho = 0.2; 
% Hold-out method
HO = cvpartition(label, 'HoldOut', ho, 'Stratify', false);

% Parameter setting
N = 10; 
max_Iter = 10; 
tau = 1; 
eta = 1; 
alpha = 1; 
beta = 1; 
rho = 0.2; 
phi = 0.5; 
Nf = 300; % Set number of selected features
% Ant Colony System
[sFeat, Nf, Sf, curve] = jACO(feat, label, N, max_Iter, tau, eta, alpha, beta, rho, phi, Nf, HO);

% Plot convergence curve
plot(1:max_Iter, curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ACS'); grid on;

% Preparing feature_Vector(x) for classification
x1 = sFeat;
y = testLabels;

x = array2table(x1);
x.type = y;

% Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Evaluate Classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));

% Display the mean accuracy
mean(diag(confMat))
toc

% Preparing feature_Vector(x) for classification
x = array2table(testFeatures');
x.type = testLabels;
classificationLearner

testfeaturesorgg = x;
save testfeaturesorg_EV_R18_Adult.mat

x = array2table(trainingFeatures');
x.type = trainingLabels;

trainingfeaturesorgg = x;
save trainingfeaturesorg_EV_R18_Adult.mat

classificationLearner

%%%%%%%%

imageSize = net.Layers(1).InputSize;
pixelRange = [-30 30];
scaleRange = [0.8 1.2];
aug = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandScale', scaleRange);
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, ...
    'DataAugmentation', aug, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'avg_pool'; % or another deep layer
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');

% Example of using Random Forest
classifier = fitcensemble(trainingFeatures', trainingLabels);

Nf = 3000; % or another higher value
ho = 0.1; % Hold-out proportion for test data
HO = cvpartition(label, 'HoldOut', ho, 'Stratify', false);
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imds.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedTestSet, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

netTransfer = trainNetwork(augmentedTrainingSet, layers, options);

trainingFeatures = activations(netTransfer, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(netTransfer, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');

%%%%%%%%%%%%%%%%%% to get evaluation metricsx from saved .mat

load('classifier_models\inceptionv3(2).mat'); 

classifiers = {'Linear SVM', 'Quadratic SVM', 'cubicSVM', 'mediumGaussianSVM', 'subspaceDiscriminant'};
classifierNames = {'Linear SVM', 'Quadratic SVM', 'Cubic SVM', 'Medium Gaussian SVM', 'Subspace Discriminant'};

figure;
hold on;
colors = lines(length(classifiers)); % Get a set of unique colors for the classifiers

% Ensure testFeatures are correctly transposed if needed
if size(testFeatures, 1) ~= 1920
    testFeatures = testFeatures';
end


for i = 1:length(classifiers)
    % Use the classifier to predict the scores
    classifier = eval(classifiers{i});
    [~, scores] = predict(classifier, testFeatures', 'ObservationsIn', 'columns'); % Ensure classifier is the correct object
    
    % Compute ROC curve
    [X, Y, ~, AUC] = perfcurve(trueLabels, scores(:, 2), 'Control'); 
    
    % Plot ROC curve
    plot(X, Y, 'DisplayName', sprintf('%s (AUC = %.2f)', classifierNames{i}, AUC), 'Color', colors(i, :));
end

% Customize plot
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Different Classifiers on inceptionv3');
legend('show');
grid on;
hold off;

