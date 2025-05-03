% Define the input and output directories
inputDir = 'D:\psychiaric\new\Processed_data_child'; % Replace with your input directory path
outputDir = 'D:\psychiaric\new\E_Images_with_Child';    % Replace with your output directory path
categories = {'Bipolar', 'Control', 'MDD_C'};        % Your category subfolders

% Create the output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Loop through each category
for c = 1:numel(categories)
    categoryDir = fullfile(inputDir, categories{c});
    files = dir(fullfile(categoryDir, '**', '*.*')); % Get all files in the category directory
    
    % Loop through each file
    for i = 1:numel(files)
        [~, ~, ext] = fileparts(files(i).name);
        fullFilePath = fullfile(files(i).folder, files(i).name);
        
        if strcmpi(ext, '.nii')
            % Load NIfTI file
            niftiData = niftiread(fullFilePath);
            numSlices = size(niftiData, 3);
            
            % Save each slice as a PNG image
            for sliceNum = 1:numSlices
                imgSlice = niftiData(:, :, sliceNum);
                imgSlice = rescale(imgSlice); % Normalize the slice to [0, 1]
                outputFileName = fullfile(outputDir, sprintf('%s_slice_%03d.png', files(i).name(1:end-4), sliceNum));
                imwrite(imgSlice, outputFileName);
            end
            
        elseif strcmpi(ext, '.dcm')
            % Load DICOM file
            dicomData = dicomread(fullFilePath);
            numSlices = size(dicomData, 3);
            
            % Save each slice as a PNG image
            for sliceNum = 1:numSlices
                imgSlice = dicomData(:, :, sliceNum);
                imgSlice = rescale(imgSlice); % Normalize the slice to [0, 1]
                outputFileName = fullfile(outputDir, sprintf('%s_slice_%03d.png', files(i).name(1:end-4), sliceNum));
                imwrite(imgSlice, outputFileName);
            end
        end
    end
end

disp('All slices have been extracted and saved as PNG images.');