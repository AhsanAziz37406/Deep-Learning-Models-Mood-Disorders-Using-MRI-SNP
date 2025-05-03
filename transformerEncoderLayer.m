function layer = transformerEncoderLayer(embedDim, numHeads, varargin)
    p = inputParser;
    addParameter(p, 'Name', '');
    parse(p, varargin{:});
    
    Name = p.Results.Name;
    
    attentionLayer = multiheadAttentionLayer(numHeads, embedDim, 'Name', [Name, '_attention']);
    normLayer1 = layerNormalizationLayer('Name', [Name, '_norm1']);
    ffnLayer = feedForwardNetwork(embedDim, 'Name', [Name, '_ffn']);
    normLayer2 = layerNormalizationLayer('Name', [Name, '_norm2']);
    addLayer1 = additionLayer(2, 'Name', [Name, '_add1']);
    addLayer2 = additionLayer(2, 'Name', [Name, '_add2']);
    
    layer = [
        attentionLayer
        addLayer1
        normLayer1
        ffnLayer
        addLayer2
        normLayer2];
end