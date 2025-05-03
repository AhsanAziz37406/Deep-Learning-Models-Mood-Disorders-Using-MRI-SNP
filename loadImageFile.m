% function data = loadImageFile(filename)
%     [~, ~, ext] = fileparts(filename);  % Get file extension
% 
%     switch lower(ext)
%         case '.nii'
%             data = loadNiftiFile(filename);  % Use the existing function for .nii files
%         case '.dcm'
%             data = loadDicomFile(filename);  % Use a function for .dcm files
%         otherwise
%             error('Unsupported file extension: %s', ext);
%     end
% end
% 
% function data = loadNiftiFile(filename)
%     % Load NIfTI file using niftiread
%     niftiData = niftiread(filename);
%     % Process or return the loaded data
%     data = niftiData;  % Add more processing if needed
% end
% 
% function data = loadDicomFile(filename)
%     % Load DICOM file using dicomread
%     dicomData = dicomread(filename);
%     % Process or return the loaded data
%     data = dicomData;  % Add more processing if needed
% end


function img = loadImageFile(filename)
    [~, ~, ext] = fileparts(filename);  % Get file extension
    
    switch lower(ext)
        case '.nii'
            img = loadNiftiFile(filename);  % Use the existing function for .nii files
        case '.dcm'
            img = loadDicomFile(filename);  % Use a function for .dcm files
        otherwise
            error('Unsupported file extension: %s', ext);
    end
end

function img = loadNiftiFile(filename)
    nii = niftiread(filename);
    img = rescale(nii);  % Normalize the image
    if size(img, 3) > 1
        img = img(:, :, 1);  % Take the first slice if it's a 3D volume
    end
    img = repmat(img, [1 1 3]);  % Convert to RGB by replicating the single channel
end

function img = loadDicomFile(filename)
    dicomData = dicomread(filename);
    img = rescale(dicomData);  % Normalize the image
    if size(img, 3) > 1
        img = img(:, :, 1);  % Take the first slice if it's a 3D volume
    end
    img = repmat(img, [1 1 3]);  % Convert to RGB by replicating the single channel
end
