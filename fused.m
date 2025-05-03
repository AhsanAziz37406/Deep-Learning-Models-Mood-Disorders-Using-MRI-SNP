
% Load features from the saved .mat files
data1 = load('M_Nii_G_D201.mat');
data2 = load('M_Nii_G_R50.mat');

% Assuming the features are named 'featuresD201' and 'featuresR50' respectively
featuresD201 = data1.XTest; % replace 'featuresD201' with the actual variable name
featuresR50 = data2.XTest; % replace 'featuresR50' with the actual variable name

% Concatenate the features horizontally
fusedFeatures = [featuresD201, featuresR50];

% Save the combined features into a new .mat file
save('FusedFeatures.mat', 'fusedFeatures');
%%%%%%%%%%%%%%

vector1= load('testFeatures50.mat');
vector2= load('testfeaturesorg201.mat');

r1=vector1.testFeatures' ;
r2=vector2.testFeatures' ;

labels= vector1.testLabels;
fused1= horzcat(r1,r2);

finalvector= array2table(fused1);
finalvector.type = labels;

% Save the finalvector
save('finalvector_D201_R50.mat', 'finalvector');

classificationLearner