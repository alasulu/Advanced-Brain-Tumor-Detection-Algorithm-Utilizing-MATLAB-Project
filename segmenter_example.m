% Load the .mat file
load('30.mat'); % Replace with your actual data file name

% Normalize the image and convert to uint8
normalizedImage = uint8(256 * mat2gray(cjdata.image));

% Apply Gaussian blur
sigma = 0.1; % Adjust for the required blur intensity
blurredImage = imgaussfilt(normalizedImage, sigma);

% Apply median filtering
filteredImage = medfilt2(normalizedImage, [3, 3]);

% Apply CLAHE for contrast enhancement
claheImage = adapthisteq(filteredImage, 'clipLimit', 0.01, 'Distribution', 'uniform');

% Skull removal process
[pixelCount, grayLevels] = imhist(claheImage);

% Crop the image to remove borders
croppedImage = claheImage(3:end-3, 4:end-4);

% Binary thresholding
binaryImage = croppedImage > 20;

% Remove small noise components
binaryImage = bwareaopen(binaryImage, 10);

% Fill edges and holes
binaryImage(end, :) = true;
binaryImage = imfill(binaryImage, 'holes');

% Erode the binary image to refine skull removal
se = strel('disk', 20);
binaryImage = imerode(binaryImage, se);

% Initialize the final image with the processed image
finalImage = claheImage;

% Flatten the image for k-means clustering
flattenedImage = double(finalImage(:));

% Perform k-means clustering to segment brain and tumor
numClusters = 2;
[clusterIdx, clusterCenters] = kmeans(flattenedImage, numClusters);

% Identify the cluster with the tumor (highest intensity)
tumorCluster = find(clusterCenters == max(clusterCenters));

% Create a mask for the tumor pixels
tumorMask = false(size(finalImage));
tumorMask(clusterIdx == tumorCluster) = true;

% Apply the tumor mask to segment the image
segmentedTumor = finalImage;
segmentedTumor(~tumorMask) = 0;

% Remove skull by applying the binary mask
finalImage(~binaryImage) = 0;

% Segment the tumor using thresholding
thresholdValue = 0.666; % Can be dynamically calculated if needed
binaryTumor = imbinarize(finalImage, thresholdValue);

% Clean small noise pixels and fill holes
binaryTumor = bwareaopen(binaryTumor, 1200);
binaryTumor = imfill(binaryTumor, 'holes');

% Create and apply the final tumor mask
tumorMask = binaryTumor;
finalTumorSegmented = finalImage;
finalTumorSegmented(~tumorMask) = 0;

% Display the results
figure;
subplot(1, 2, 1);
imshow(finalImage);
title('Skull Removed Image');

subplot(1, 2, 2);
imshow(finalTumorSegmented);
title('Segmented Tumor');
