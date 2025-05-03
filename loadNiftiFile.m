% % Define a function to read nifti files
% function img = loadNiftiFile(filename)
%     nii = niftiread(filename);
%     img = rescale(nii); % Normalize the image
% end

% function nii = loadNiftiFile(filePath)
%     % Use the niftiread function
%     nii = niftiread(filePath); % Using the built-in niftiread function
% end



function img = loadNiftiFile(filename)
    nii = niftiread(filename);
    img = rescale(nii); % Normalize the image
    if size(img, 3) > 1
        img = img(:, :, 1); % Take the first slice if it's a 3D volume
    end
end


%%%%%%%for 3d

% function img = loadNiftiFile(filename)
%     nii = niftiread(filename);
%     img = rescale(nii); % Normalize the image
% 
%     % Optionally resize or preprocess the entire 3D volume
%     % Example: Reshape to [X, Y, Z, C] where C is the number of channels
%     img = reshape(img, size(img, 1), size(img, 2), size(img, 3), 1); % For grayscale
% 
%     % Uncomment the above line if your network requires a 4D input
% 
%     % Ensure the output size matches your network input requirements
%     % For example, if your network requires 224x224x3 images (like ResNet-50)
%     img = imresize(img, [224 224]); 
% 
%     % Note: Adjust preprocessing based on your specific network architecture and input requirements
% end
