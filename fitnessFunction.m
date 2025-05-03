function accuracy = fitnessFunction(selectedFeatures, trainingFeatures, trainingLabels, numClasses)
    % Convert logical index to numeric index
    selectedIndices = find(selectedFeatures);
    
    % Select features
    selectedTrainingFeatures = trainingFeatures(:, selectedIndices);
    
    % Train SVM classifier
    svmModel = fitcecoc(selectedTrainingFeatures', trainingLabels, ...
        'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
    
    % Cross-validation for accuracy estimation
    cv = crossval(svmModel, 'KFold', 5);
    accuracy = 1 - kfoldLoss(cv);  % 1 - classification error
end
