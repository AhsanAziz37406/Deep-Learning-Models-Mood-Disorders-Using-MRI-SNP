function layers = createVisionTransformer(inputSize, numClasses)
    % Define the Vision Transformer layers
    layers = [
        imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
        convolution2dLayer(16, 16, 'Stride', 16, 'Name', 'patch_embedding')
        flattenLayer('Name', 'flatten')
        fullyConnectedLayer(128, 'Name', 'fc_embedding')
        transformerEncoderLayer(128, 4, 'Name', 'encoder1')
        transformerEncoderLayer(128, 4, 'Name', 'encoder2')
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')];
end

