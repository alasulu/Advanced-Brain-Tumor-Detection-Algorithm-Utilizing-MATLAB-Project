clear
clc

% Define source directories for masks and tumors, and destination directory
masksDir = '/Users/macbookair/Desktop/tumors/pituitary/pituitary_mask';
tumorsDir = '/Users/macbookair/Desktop/tumors/pituitary/pituitary_tumor';
destinationDir = '/Users/macbookair/Desktop/tumors/pituitary/merged_image';

% Check if the destination directory exists, if not, create it
if ~exist(destinationDir, 'dir')
    mkdir(destinationDir);
end

% Get a list of all mask files in the masks directory
maskPattern = fullfile(masksDir, '*.png');
maskFiles = dir(maskPattern);

% Loop through each mask file
for k = 1:length(maskFiles)
    % Get the base file name (assuming mask and tumor files have the same name)
    maskFileName = maskFiles(k).name;
    tumorFileName = maskFileName;  % They have the same name
    
    % Full path to mask and tumor files
    maskFile = fullfile(masksDir, maskFileName);
    tumorFile = fullfile(tumorsDir, tumorFileName);
    
    % Check if the corresponding tumor file exists
    if exist(tumorFile, 'file')
        % Read the mask and tumor images
        maskImage = imread(maskFile);
        tumorImage = imread(tumorFile);
        
        % Ensure both images are of the same size
        if size(maskImage, 1) == size(tumorImage, 1) && size(maskImage, 2) == size(tumorImage, 2)
            % Apply the mask to the tumor image
            combinedImage = uint8(double(tumorImage) .* (double(maskImage) / 255));
            
            % Save the combined image
            destinationFile = fullfile(destinationDir, maskFileName);
            imwrite(combinedImage, destinationFile);
        else
            warning('Size mismatch between %s and %s. Skipping these files.', maskFileName, tumorFileName);
        end
    else
        warning('Corresponding tumor file for %s not found. Skipping.', maskFileName);
    end
end

disp('Images have been combined and saved successfully.');
