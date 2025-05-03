function layer = feedForwardNetwork(embedDim, varargin)
    p = inputParser;
    addParameter(p, 'Name', '');
    parse(p, varargin{:});
    
    Name = p.Results.Name;
    
    layer = [
        fullyConnectedLayer(embedDim, 'Name', [Name, '_fc1'])
        reluLayer('Name', [Name, '_relu'])
        fullyConnectedLayer(embedDim, 'Name', [Name, '_fc2'])];
end