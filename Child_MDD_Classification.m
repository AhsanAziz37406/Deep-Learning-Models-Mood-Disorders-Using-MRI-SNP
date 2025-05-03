net = shufflenet;

rootFolder = 'D:\psychiaric\new'; 

imds = imageDatastore(fullfile(rootFolder,'Binary_E_Child'), 'LabelSource','foldernames', 'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.jpeg','.png','.bmp','.jfif'});

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
featureLayer = 'node_200';

tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeatures_Child_SN.mat
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
save testfeaturesorg_Child_SN.mat
x = array2table(trainingFeatures');
x.type = trainingLabels;
trainingfeaturesorgg = x;
save trainingfeaturesorg_Child_SN.mat
classificationLearner


%%% %%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% nifty and dicom necha seedah


net = inceptionv3;

rootFolder = 'D:\psychiaric\new'; 
% Define the categories (subfolders)
categories = {'Bipolar', 'Control', 'MDD_C'};
% % Load data from each category folder
% imds = imageDatastore(fullfile(rootFolder, categories), ...
%     'LabelSource', 'foldernames', ...
%     'FileExtensions', '.nii', ...
%     'ReadFcn', @loadNiftiFile);
imds = imageDatastore(fullfile(rootFolder, 'Processed_data_child', categories), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.nii', '.dcm'}, ...
    'ReadFcn', @loadImageFile);


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
featureLayer = 'avg_pool';

tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeatures_Child_V3.mat
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
Nf = 1900; % Set number of selected features
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
save testfeaturesorg_Child_V3.mat
x = array2table(trainingFeatures');
x.type = trainingLabels;
trainingfeaturesorgg = x;
save trainingfeaturesorg_Child_V3.mat
classificationLearner
