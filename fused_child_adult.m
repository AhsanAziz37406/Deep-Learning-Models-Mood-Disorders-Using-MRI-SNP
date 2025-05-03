% Load saved test features from the adult dataset
load('testFeatures_EV_R18_Adult.mat');      % Contains 'testFeatures' (features from adult test set)
load('testfeaturesorg_EV_R18_Adult.mat');   % Contains 'testLabels' (labels from adult test set)

% Load saved test features from the child dataset
load('testFeatures_EV_R18_child.mat');    % Contains 'testFeatures' (features from child test set)
load('testfeaturesorg_EV_R18_child.mat'); % Contains 'testLabels' (labels from child test set)

% Combine test features from adult and child datasets
combinedFeatures = [testFeatures, testFeatures_child];  % Combine adult and child test features

% Combine the test labels from adult and child datasets
combinedLabels = [testLabels; testLabels_child];  % Combine adult and child test labels

% Save combined features and labels for external validation
save('CombinedTF_child_adult_EV_R18.mat', 'combinedFeatures', 'combinedLabels');

% Optionally, load the combined data and proceed to classifier training/validation
load('CombinedTF_child_adult_EV_R18.mat');  % Contains 'combinedFeatures', 'combinedLabels'

% Train a classifier using the combined features (if needed)
classifier = fitcecoc(combinedFeatures, combinedLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Save the trained classifier for future use
save('trainedClassifier_combined_child_adult_R18.mat', 'classifier');

% You can also load this data into the MATLAB Classification Learner app for further exploration
classificationLearner
