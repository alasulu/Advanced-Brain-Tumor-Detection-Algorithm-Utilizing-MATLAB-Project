clear
clc

% Define source and destination directories
sourceDir = '/Users/macbookair/Desktop/data/Glioma';
destinationDir = '/Users/macbookair/Desktop/tumors/glioma/glioma_mask';

% Check if the destination directory exists, if not, create it
if ~exist(destinationDir, 'dir')
    mkdir(destinationDir);
end

% Get a list of all files in the source directory ending with _mask.png
filePattern = fullfile(sourceDir, '*_mask.png');
files = dir(filePattern);

% Initialize a counter for the new filenames
fileCounter = 1;

% Loop through each file, move and rename it
for k = 1:length(files)
    % Get the source file path
    baseFileName = files(k).name;
    sourceFile = fullfile(sourceDir, baseFileName);
    
    % Create the new filename
    newFileName = sprintf('tumor%d.png', fileCounter);
    destinationFile = fullfile(destinationDir, newFileName);
    
    % Move and rename the file
    movefile(sourceFile, destinationFile);
    
    % Increment the counter
    fileCounter = fileCounter + 1;
end

disp('Files have been moved and renamed successfully.');
