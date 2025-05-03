
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
imds = imageDatastore(fullfile(rootFolder, 'child_Normal_abnormal.dcm', categories), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.dcm'}, ...
    'ReadFcn', @dicomread);


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


trainingLabels_child = trainingSet.Labels;
testLabels_child = testSet.Labels;
featureLayer = 'pool5';

tic
trainingFeatures_child = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures_child = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeatures_EV_R18_child.mat
feat = testFeatures_child';
feat = double(feat);
label = testLabels_child;
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
y = testLabels_child;

x = array2table(x1);
x.type = y;

% Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures_child, trainingLabels_child, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
% Evaluate Classifier
predictedLabels = predict(classifier, testFeatures_child, 'ObservationsIn', 'columns');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels_child, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));

% Display the mean accuracy
mean(diag(confMat))
toc

% Preparing feature_Vector(x) for classification
x = array2table(testFeatures_child');
x.type = testLabels_child;
classificationLearner

testfeaturesorgg_child = x;
save testfeaturesorg_EV_R18_child.mat

x = array2table(trainingFeatures_child');
x.type = trainingLabels_child;

trainingfeaturesorgg_child = x;
save trainingfeaturesorg_EV_R18_child.mat

classificationLearner


