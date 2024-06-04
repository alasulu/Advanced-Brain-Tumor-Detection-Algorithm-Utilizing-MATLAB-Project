clear
clc

% Define source and destination directories
sourceDir = '/Users/macbookair/Desktop/data/Glioma';
destinationDir = '/Users/macbookair/Desktop/tumors/glioma/glioma_tumor';

% Check if the destination directory exists, if not, create it
if ~exist(destinationDir, 'dir')
    mkdir(destinationDir);
end

% Get a list of all files in the source directory ending with .png
filePattern = fullfile(sourceDir, '*.png');
files = dir(filePattern);

% Initialize a counter for the new filenames
fileCounter = 1;

% Regular expression pattern to match filenames with any number ending in .png
regexPattern = '^\d+\.png$';

% Loop through each file
for k = 1:length(files)
    % Get the base file name
    baseFileName = files(k).name;
    
    % Check if the filename matches the pattern
    if regexp(baseFileName, regexPattern)
        % Define the new filename
        newFileName = sprintf('tumor%d.png', fileCounter);
        
        % Full path to source and destination files
        sourceFile = fullfile(sourceDir, baseFileName);
        destinationFile = fullfile(destinationDir, newFileName);
        
        % Move and rename the file
        movefile(sourceFile, destinationFile);
        
        % Increment the counter
        fileCounter = fileCounter + 1;
    end
end

disp('Files have been moved and renamed successfully.');
