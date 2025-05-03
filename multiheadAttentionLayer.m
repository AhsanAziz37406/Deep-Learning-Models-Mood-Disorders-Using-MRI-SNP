function layer = multiheadAttentionLayer(numHeads, embedDim, varargin)
    p = inputParser;
    addParameter(p, 'Name', '');
    parse(p, varargin{:});
    
    Name = p.Results.Name;
    
    % Define the components of the multi-head attention layer
    queryLayer = fullyConnectedLayer(embedDim, 'Name', [Name, '_query']);
    keyLayer = fullyConnectedLayer(embedDim, 'Name', [Name, '_key']);
    valueLayer = fullyConnectedLayer(embedDim, 'Name', [Name, '_value']);
    outputLayer = fullyConnectedLayer(embedDim, 'Name', [Name, '_output']);
    
    % Combine layers to form the multi-head attention layer
    layer = [
        queryLayer
        keyLayer
        valueLayer
        additionLayer(2, 'Name', [Name, '_add'])
        reluLayer('Name', [Name, '_relu'])
        outputLayer];
end
