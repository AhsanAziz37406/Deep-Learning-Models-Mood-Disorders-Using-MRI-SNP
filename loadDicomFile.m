function data = loadDicomFile(filename)
    % Load DICOM file using dicomread
    dicomData = dicomread(filename);
    % If you need to read metadata, use dicominfo
    info = dicominfo(filename);
    
    % Process or return the loaded data
    data = dicomData;  % You can add more processing here if needed
end
