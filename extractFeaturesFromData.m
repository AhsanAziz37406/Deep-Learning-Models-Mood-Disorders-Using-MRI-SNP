function features = extractFeaturesFromData(net, data, featureLayer)
    % Ensure data has 3 channels (RGB format) expected by ResNet-50
    if size(data, 3) == 1
        % If data is single-channel (e.g., grayscale), convert it to RGB
        dataResized = cat(3, data, data, data); % Convert to RGB by replicating channels
    elseif size(data, 3) == 3
        % Data already has 3 channels (RGB), no need to modify
        dataResized = data;
    else
        error('Input data must have 1 or 3 channels for ResNet-50 input.');
    end
    
    % Resize data if needed to match ResNet-50 input size
    inputSize = net.Layers(1).InputSize(1:3); % Expected input size of ResNet-50
    dataResized = imresize(dataResized, inputSize(1:2)); % Resize data if needed
    
    % Extract features using ResNet-50
    features = activations(net, dataResized, featureLayer, 'OutputAs', 'columns');
end
