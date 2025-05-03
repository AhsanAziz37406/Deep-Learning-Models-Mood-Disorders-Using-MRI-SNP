% Define the folder containing the .dcm files
dcmFolder = 'D:\psychiaric\new\Binary_CHILD_NORMAL_ABNORMAL\Extracted_child_NORMAL_DCMfiles';  % Path to the folder with the renamed .dcm files
pngFolder = 'D:\psychiaric\new\Binary_CHILD_NORMAL_ABNORMAL\Normal';  % Path where you want to save the PNG files

% Check if the destination folder exists, if not, create it
if ~exist(pngFolder, 'dir')
    mkdir(pngFolder);
end

% Get a list of all .dcm files in the folder
dcmFiles = dir(fullfile(dcmFolder, '*.dcm'));

% Loop through each .dcm file
for i = 1:length(dcmFiles)
    dcmFileName = dcmFiles(i).name;  % Get the .dcm file name
    dcmFilePath = fullfile(dcmFolder, dcmFileName);  % Full path to the .dcm file
    
    % Read the DICOM file
    dcmData = dicomread(dcmFilePath);
    
    % Convert the image to grayscale if it is not already
    if size(dcmData, 3) > 1
        dcmData = rgb2gray(dcmData);
    end
    
    % Normalize the image data to the range [0, 1]
    dcmData = mat2gray(dcmData);
    
    % Create the corresponding PNG file name
    [~, name, ~] = fileparts(dcmFileName);  % Get the name part of the file
    pngFileName = [name '.png'];  % Change the extension to .png
    pngFilePath = fullfile(pngFolder, pngFileName);  % Full path to the PNG file
    
    % Write the image data to a PNG file
    imwrite(dcmData, pngFilePath);
end

disp('All DICOM slices have been extracted and saved as PNG files.');