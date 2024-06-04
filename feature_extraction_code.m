clear
clc

%%%--- Import ---%%%

kaynakKlasoru = '/Users/macbookair/Desktop/tumors/glioma/merged_image'; % Path 
dosyalar = dir(fullfile(kaynakKlasoru, '*.png')); % Import all .png files in kaynakKlasoru
numImages = numel(dosyalar); % Preallocate a properties matrix
% Preallocating parameter matrices
energies = zeros(numImages, 1); % Energy parameter
means = zeros(numImages, 1); % Means parameter
stds = zeros(numImages, 1); % Standart deviation parameter
rms_values = zeros(numImages, 1); % RMS parameter
variances = zeros(numImages, 1); % Variances parameter

%%%--- Preallocating for Feature extraction ---%%%

% Preallocating GLCM features
entropies = zeros(numImages, 1);
contrast = zeros(numImages, 1);
correlation = zeros(numImages, 1);
homogeneity = zeros(numImages, 1);
dissimilarityGLCM = zeros(numImages,1);
% Preallocating parameter matrices
autoc = zeros(numImages,1); % Preallocating  empty matrices for Autocorrelation
contr = zeros(numImages,1); % Preallocating  empty matrices for Contrast
corrm = zeros(numImages,1); % Preallocating  empty matrices for Correlation
corrp = zeros(numImages,1); % Preallocating  empty matrices for Correlation: [1,2] (out.corrp)
cprom = zeros(numImages,1);  % Preallocating  empty matrices for Cluster Prominence: [2] (out.cprom)
cshad = zeros(numImages,1); % Preallocating  empty matrices for Cluster Shade: [2] (out.cshad)
dissi = zeros(numImages,1); % Preallocating  empty matrices for Dissimilarity: [2] (out.dissi)
energ = zeros(numImages,1); % Preallocating  empty matrices for Energy: matlab / [1,2] (out.energ)
entro = zeros(numImages,1);% Preallocating  empty matrices for Entropy: [2] (out.entro)
homom = zeros(numImages,1); % Preallocating  empty matrices for Homogeneity: matlab (out.homom)
homop = zeros(numImages,1); % Preallocating  empty matrices for Homogeneity: [2] (out.homop)
maxpr = zeros(numImages,1); % Preallocating  empty matrices for Maximum probability: [2] (out.maxpr)
sosvh = zeros(numImages,1); % Preallocating  empty matrices for Sum of sqaures: Variance [1] (out.sosvh)
savgh = zeros(numImages,1);  % Preallocating  empty matrices for Sum average [1] (out.savgh)
svarh = zeros(numImages,1); % Preallocating  empty matrices for Sum variance [1] (out.svarh)
senth = zeros(numImages,1); % Preallocating  empty matrices for Sum entropy [1] (out.senth)
dvarh = zeros(numImages,1); % Preallocating  empty matrices for Difference variance [1] (out.dvarh)
denth = zeros(numImages,1);%  Preallocating  empty matrices for Difference entropy [1] (out.denth)
inf1h = zeros(numImages,1); % Preallocating  empty matrices for Information measure of correlation1 [1] (out.inf1h)
inf2h = zeros(numImages,1); % Preallocating  empty matrices for Informaiton measure of correlation2 [1] (out.inf2h)
indnc = zeros(numImages,1); % Preallocating  empty matrices for Inverse difference normalized (INN) [3] (out.indnc)
idmnc = zeros(numImages,1); % Preallocating  empty matrices for Inverse difference moment normalized [3](out.idmnc)
% Preallocating GLSZM features
smallAreaEmphasis = zeros(numImages, 1);
% Preallocating Neighboring Gray Tone Difference Matrix (NGTDM) Features
coarseness = zeros(numImages,1);
% Preallocating GLDM features
entropyGLDM = zeros(numImages,1);
energyGLDM = zeros(numImages,1);

%%%--- Feature extraction ---%%%

saveFolderPath = '/Users/macbookair/Desktop/tumors/glioma/csv_out';
columnHeaders = {'Energy', 'Mean', 'Std', 'Entropy', 'RMS Value', 'Variance', 'Contrast', 'Correlation', 'Homogeniety', 'SmallAreaEmphasis', 'entropyGLDM', 'energyGLDM', 'dissimilarityGLCM', ...
    'autoc', 'contr', 'corrm', 'corrp', 'cprom', 'cshad', 'dissi', 'energ', 'entro', 'homom', 'homop', 'maxpr', 'sosvh', 'savgh', 'svarh', 'senth', 'dvarh', 'denth', 'inf1h', 'inf2h', 'indnc', 'idmnc'};
% Excel File Name Contents and Path
excelFilePath = fullfile(saveFolderPath, "Features.xlsx"); % Save as excel workbook
xlswrite(excelFilePath, columnHeaders, 'Sheet1', 'A1'); % Write and header settings for excel workbook

%%%--- Some preprocessing ---%%%

for dosyaIndeks = 1:numel(dosyalar)
    % Import the path 
    dosyaAdi = dosyalar(dosyaIndeks).name;
    dosyaYolu = fullfile(kaynakKlasoru, dosyaAdi);
    % Read the files
    img = imread(dosyaYolu);
    outputfolder = '/Users/macbookair/Desktop/tumors/glioma/glioma_mask';
    extractedtumorformat = dir(fullfile(outputfolder, '*.png'));
    extractedtumornaming = extractedtumorformat(dosyaIndeks).name;
    extractedtumorfolder = fullfile(outputfolder, extractedtumornaming);
    medianImage = medfilt2(img, [3 3]); % Median filtering
    % Kontrast sınırlama histogram eşitleme
    claheImage = adapthisteq(medianImage, 'clipLimit', 0.01, 'Distribution', 'uniform'); % Contrast limitation and 
    % histogram equalization
    adaptivehist = adapthisteq(medianImage); % Adaptive histogram equalization
    [pixelCount, grayLevels] = imhist(adaptivehist); % Skull removing operations
    adaptivehist = adaptivehist(3:end-3, 4:end-4);
    binaryImage = adaptivehist > 20;
    binaryImage = bwareaopen(binaryImage, 10);
    binaryImage(end, :) = true;
    binaryImage = imfill(binaryImage, 'holes');
    se = strel('disk', 20, 0);
    binaryImage = imerode(binaryImage, se);
    finalImage = adaptivehist;
    % Final skull removing operations
    finalImage(~binaryImage) = 0;
    sout=imresize(finalImage,[512,512]); % Resizing all images 
    t0 = 10; % Parameter 
    th = t0+((max(finalImage(:))+min(finalImage(:)))./2);
    for i=1:1:size(finalImage,1)
        for j=1:1:size(finalImage,2)
            if finalImage(i,j)>th
                sout(i,j)=1;
            else
                sout(i,j)=0;
            end
        end
    end
    label=bwlabel(sout);
    stats1=regionprops(logical(sout),'Solidity','Area','BoundingBox');
    density=[stats1.Solidity];
    area=[stats1.Area];
    high_dense_area=density>0.6;
    max_area=max(area(high_dense_area));
    tumor_label=find(area==max_area);
    tumor=ismember(label,tumor_label);
    se = strel('disk', 10, 0); % Define structuring element (square of size 3x3)
    %se değeri ile tümöre ait parlaklık üzerinde değişiklikler yapılabiliyor.
    % Eroding the original image with with structuring element
    dilatedfinal = imread(extractedtumorfolder);
    dilatedimage = imdilate(finalImage,se);
    extractedTumor = medianImage;
    extractedTumor(dilatedfinal == 0) = 0;
    extractedTumor = im2bw(extractedTumor,0.1);
    medianImage(extractedTumor ==0) = 0;
    %medianImage = medianImage .* dilatedfinal;
    % Tümöre ait özellikleri hesapla
    %Energy
    glcms = graycomatrix(medianImage);
    %stats2 = graycoprops(glcms, {'Contrast', 'Correlation', 'Homogeneity', 'Energy'});
    stats2 = graycoprops(glcms, {'Contrast', 'Correlation', 'Homogeneity', 'Energy'});
    % Writing GLCM Features
    contrast(dosyaIndeks) = stats2.Contrast;
    correlation(dosyaIndeks) = stats2.Correlation;
    homogeneity(dosyaIndeks) = stats2.Homogeneity;
    energies(dosyaIndeks) = stats2.Energy;
    dissimilarityGLCM(dosyaIndeks) = mean(mean(glcms));
    % Writing GLSZM Features
    % Small Area Emphasis (SAE) Calculation
    N = size(glcms, 1); % Matrice size
    SAE = 0;
    for i = 1:N
        for j = 1:N
            SAE = SAE + ((i - j)^2 * glcms(i, j));
        end
    end
    % SAE değerini hesaplanan özellikler listesine ekleme
    smallAreaEmphasis(dosyaIndeks) = SAE;
     % Writing GLDM features
        [~, gmag] = imgradient(medianImage);
        glcm2 = graycomatrix(gmag);
        % Writing Entropy (GLDM) feature
        entropyGLDM(dosyaIndeks)= entropy(glcm2);
        % Writing Energy (GLDM) feature
        energyGLDM(dosyaIndeks)= sum(sum(glcm2.^2));
    % First order statistics
    means(dosyaIndeks) = mean2(medianImage); % Mean
    stds(dosyaIndeks) = std2(medianImage); %STD
    entropies(dosyaIndeks) = entropy(medianImage); %Entropy
    img_vector = double(medianImage(:));
    squared_values = img_vector .^ 2;
    mean_squared = mean(squared_values);
    rms_values(dosyaIndeks) = sqrt(mean_squared); %RMS Values
    variances(dosyaIndeks) = var(double(medianImage(:))); %Variance
    GLCM2 = graycomatrix(medianImage,'Offset',[2 0;0 2]);
    stats4 = GLCM_Features1(GLCM2,1);
    autoc(dosyaIndeks) = stats4.autoc;
    contr(dosyaIndeks) = stats4.contr;
    corrm(dosyaIndeks) = stats4.corrm;
    corrp(dosyaIndeks) = stats4.corrp;
    cprom(dosyaIndeks) = stats4.cprom;
    cshad(dosyaIndeks) = stats4.cshad;
    dissi(dosyaIndeks) = stats4.dissi;
    energ(dosyaIndeks) = stats4.energ;
    entro(dosyaIndeks) = stats4.entro;
    homom(dosyaIndeks) = stats4.homom;
    homop(dosyaIndeks) = stats4.homop;
    maxpr(dosyaIndeks) = stats4.maxpr;
    sosvh(dosyaIndeks) = stats4.sosvh;
    savgh(dosyaIndeks) = stats4.savgh;
    svarh(dosyaIndeks) = stats4.svarh;
    senth(dosyaIndeks) = stats4.senth;
    dvarh(dosyaIndeks) = stats4.dvarh;
    denth(dosyaIndeks) = stats4.denth;
    inf1h(dosyaIndeks) = stats4.inf1h;
    inf2h(dosyaIndeks) = stats4.inf2h;
    indnc(dosyaIndeks) = stats4.indnc;
    idmnc(dosyaIndeks) = stats4.idmnc;   
    % Excel dosyasına özellikleri yazma
    rowData = {energies(dosyaIndeks), means(dosyaIndeks), stds(dosyaIndeks), entropies(dosyaIndeks), rms_values(dosyaIndeks), variances(dosyaIndeks), contrast(dosyaIndeks), correlation(dosyaIndeks),homogeneity(dosyaIndeks),smallAreaEmphasis(dosyaIndeks),entropyGLDM(dosyaIndeks),energyGLDM(dosyaIndeks),dissimilarityGLCM(dosyaIndeks),autoc(dosyaIndeks),contr(dosyaIndeks),corrm(dosyaIndeks),corrp(dosyaIndeks),cprom(dosyaIndeks),cshad(dosyaIndeks),dissi(dosyaIndeks),energ(dosyaIndeks),entro(dosyaIndeks),homom(dosyaIndeks),homop(dosyaIndeks),maxpr(dosyaIndeks),sosvh(dosyaIndeks),savgh(dosyaIndeks),svarh(dosyaIndeks),senth(dosyaIndeks),dvarh(dosyaIndeks),denth(dosyaIndeks),inf1h(dosyaIndeks),inf2h(dosyaIndeks),indnc(dosyaIndeks),idmnc(dosyaIndeks)};
    xlswrite(excelFilePath, rowData, 'Sheet1', sprintf('A%d', dosyaIndeks + 1)); % +1 because headers are on the first row
end
disp('Tüm işlemler tamamlandı.');
function [out] = GLCM_Features1(glcmin,pairs)

if ((nargin > 2) || (nargin == 0))
   error('Too many or too few input arguments. Enter GLCM and pairs.');
elseif ( (nargin == 2) ) 
    if ((size(glcmin,1) <= 1) || (size(glcmin,2) <= 1))
       error('The GLCM should be a 2-D or 3-D matrix.');
    elseif ( size(glcmin,1) ~= size(glcmin,2) )
        error('Each GLCM should be square with NumLevels rows and NumLevels cols');
    end    
elseif (nargin == 1) % only GLCM is entered
    pairs = 0; % default is numbers and input 1 for percentage
    if ((size(glcmin,1) <= 1) || (size(glcmin,2) <= 1))
       error('The GLCM should be a 2-D or 3-D matrix.');
    elseif ( size(glcmin,1) ~= size(glcmin,2) )
       error('Each GLCM should be square with NumLevels rows and NumLevels cols');
    end    
end
format long e
if (pairs == 1)
    newn = 1;
    for nglcm = 1:2:size(glcmin,3)
        glcm(:,:,newn)  = glcmin(:,:,nglcm) + glcmin(:,:,nglcm+1);
        newn = newn + 1;
    end
elseif (pairs == 0)
    glcm = glcmin;
end
size_glcm_1 = size(glcm,1);
size_glcm_2 = size(glcm,2);
size_glcm_3 = size(glcm,3);
out.autoc = zeros(1,size_glcm_3); % Autocorrelation: [2] 
out.contr = zeros(1,size_glcm_3); % Contrast: matlab/[1,2]
out.corrm = zeros(1,size_glcm_3); % Correlation: matlab
out.corrp = zeros(1,size_glcm_3); % Correlation: [1,2]
out.cprom = zeros(1,size_glcm_3); % Cluster Prominence: [2]
out.cshad = zeros(1,size_glcm_3); % Cluster Shade: [2]
out.dissi = zeros(1,size_glcm_3); % Dissimilarity: [2]
out.energ = zeros(1,size_glcm_3); % Energy: matlab / [1,2]
out.entro = zeros(1,size_glcm_3); % Entropy: [2]
out.homom = zeros(1,size_glcm_3); % Homogeneity: matlab
out.homop = zeros(1,size_glcm_3); % Homogeneity: [2]
out.maxpr = zeros(1,size_glcm_3); % Maximum probability: [2]
out.sosvh = zeros(1,size_glcm_3); % Sum of sqaures: Variance [1]
out.savgh = zeros(1,size_glcm_3); % Sum average [1]
out.svarh = zeros(1,size_glcm_3); % Sum variance [1]
out.senth = zeros(1,size_glcm_3); % Sum entropy [1]
out.dvarh = zeros(1,size_glcm_3); % Difference variance [4]
%out.dvarh2 = zeros(1,size_glcm_3); % Difference variance [1]
out.denth = zeros(1,size_glcm_3); % Difference entropy [1]
out.inf1h = zeros(1,size_glcm_3); % Information measure of correlation1 [1]
out.inf2h = zeros(1,size_glcm_3); % Informaiton measure of correlation2 [1]
%out.mxcch = zeros(1,size_glcm_3);% maximal correlation coefficient [1]
%out.invdc = zeros(1,size_glcm_3);% Inverse difference (INV) is homom [3]
out.indnc = zeros(1,size_glcm_3); % Inverse difference normalized (INN) [3]
out.idmnc = zeros(1,size_glcm_3); % Inverse difference moment normalized [3]
% correlation with alternate definition of u and s
%out.corrm2 = zeros(1,size_glcm_3); % Correlation: matlab
%out.corrp2 = zeros(1,size_glcm_3); % Correlation: [1,2]
glcm_sum  = zeros(size_glcm_3,1);
glcm_mean = zeros(size_glcm_3,1);
glcm_var  = zeros(size_glcm_3,1);
u_x = zeros(size_glcm_3,1);
u_y = zeros(size_glcm_3,1);
s_x = zeros(size_glcm_3,1);
s_y = zeros(size_glcm_3,1);
% checked p_x p_y p_xplusy p_xminusy
p_x = zeros(size_glcm_1,size_glcm_3); % Ng x #glcms[1]  
p_y = zeros(size_glcm_2,size_glcm_3); % Ng x #glcms[1]
p_xplusy = zeros((size_glcm_1*2 - 1),size_glcm_3); %[1]
p_xminusy = zeros((size_glcm_1),size_glcm_3); %[1]
% checked hxy hxy1 hxy2 hx hy
hxy  = zeros(size_glcm_3,1);
hxy1 = zeros(size_glcm_3,1);
hx   = zeros(size_glcm_3,1);
hy   = zeros(size_glcm_3,1);
hxy2 = zeros(size_glcm_3,1);
%Q    = zeros(size(glcm));
for k = 1:size_glcm_3 % number glcms
    glcm_sum(k) = sum(sum(glcm(:,:,k)));
    glcm(:,:,k) = glcm(:,:,k)./glcm_sum(k); % Normalize each glcm
    glcm_mean(k) = mean2(glcm(:,:,k)); % compute mean after norm
    glcm_var(k)  = (std2(glcm(:,:,k)))^2;
    for i = 1:size_glcm_1
        for j = 1:size_glcm_2
            out.contr(k) = out.contr(k) + (abs(i - j))^2.*glcm(i,j,k);
            out.dissi(k) = out.dissi(k) + (abs(i - j)*glcm(i,j,k));
            out.energ(k) = out.energ(k) + (glcm(i,j,k).^2);
            out.entro(k) = out.entro(k) - (glcm(i,j,k)*log(glcm(i,j,k) + eps));
            out.homom(k) = out.homom(k) + (glcm(i,j,k)/( 1 + abs(i-j) ));
            out.homop(k) = out.homop(k) + (glcm(i,j,k)/( 1 + (i - j)^2));
            out.sosvh(k) = out.sosvh(k) + glcm(i,j,k)*((i - glcm_mean(k))^2);
            %out.invdc(k) = out.homom(k);
            out.indnc(k) = out.indnc(k) + (glcm(i,j,k)/( 1 + (abs(i-j)/size_glcm_1) ));
            out.idmnc(k) = out.idmnc(k) + (glcm(i,j,k)/( 1 + ((i - j)/size_glcm_1)^2));
            u_x(k)          = u_x(k) + (i)*glcm(i,j,k); % changed 10/26/08
            u_y(k)          = u_y(k) + (j)*glcm(i,j,k); % changed 10/26/08
            % code requires that Nx = Ny 
            % the values of the grey levels range from 1 to (Ng) 
        end
    end
    out.maxpr(k) = max(max(glcm(:,:,k)));
end
for k = 1:size_glcm_3
    for i = 1:size_glcm_1
        for j = 1:size_glcm_2
            p_x(i,k) = p_x(i,k) + glcm(i,j,k); 
            p_y(i,k) = p_y(i,k) + glcm(j,i,k); % taking i for j and j for i
            if (ismember((i + j),[2:2*size_glcm_1])) 
                p_xplusy((i+j)-1,k) = p_xplusy((i+j)-1,k) + glcm(i,j,k);
            end
            if (ismember(abs(i-j),[0:(size_glcm_1-1)])) 
                p_xminusy((abs(i-j))+1,k) = p_xminusy((abs(i-j))+1,k) +...
                    glcm(i,j,k);
            end
        end
    end
end
for k = 1:(size_glcm_3)
    for i = 1:(2*(size_glcm_1)-1)
        out.savgh(k) = out.savgh(k) + (i+1)*p_xplusy(i,k);
        % the summation for savgh is for i from 2 to 2*Ng hence (i+1)
        out.senth(k) = out.senth(k) - (p_xplusy(i,k)*log(p_xplusy(i,k) + eps));
    end
end
% compute sum variance with the help of sum entropy
for k = 1:(size_glcm_3)
    for i = 1:(2*(size_glcm_1)-1)
        out.svarh(k) = out.svarh(k) + (((i+1) - out.senth(k))^2)*p_xplusy(i,k);
        % the summation for savgh is for i from 2 to 2*Ng hence (i+1)
    end
end
% compute difference variance, difference entropy, 
for k = 1:size_glcm_3
    for i = 0:(size_glcm_1-1)
        out.denth(k) = out.denth(k) - (p_xminusy(i+1,k)*log(p_xminusy(i+1,k) + eps));
        out.dvarh(k) = out.dvarh(k) + (i^2)*p_xminusy(i+1,k);
    end
end
% compute information measure of correlation(1,2) [1]
for k = 1:size_glcm_3
    hxy(k) = out.entro(k);
    for i = 1:size_glcm_1
        for j = 1:size_glcm_2
            hxy1(k) = hxy1(k) - (glcm(i,j,k)*log(p_x(i,k)*p_y(j,k) + eps));
            hxy2(k) = hxy2(k) - (p_x(i,k)*p_y(j,k)*log(p_x(i,k)*p_y(j,k) + eps));
%             for Qind = 1:(size_glcm_1)
%                 Q(i,j,k) = Q(i,j,k) +...
%                     ( glcm(i,Qind,k)*glcm(j,Qind,k) / (p_x(i,k)*p_y(Qind,k)) ); 
%             end
        end
        hx(k) = hx(k) - (p_x(i,k)*log(p_x(i,k) + eps));
        hy(k) = hy(k) - (p_y(i,k)*log(p_y(i,k) + eps));
    end
    out.inf1h(k) = ( hxy(k) - hxy1(k) ) / ( max([hx(k),hy(k)]) );
    out.inf2h(k) = ( 1 - exp( -2*( hxy2(k) - hxy(k) ) ) )^0.5;
end
corm = zeros(size_glcm_3,1);
corp = zeros(size_glcm_3,1);
for k = 1:size_glcm_3
    for i = 1:size_glcm_1
        for j = 1:size_glcm_2
            s_x(k)  = s_x(k)  + (((i) - u_x(k))^2)*glcm(i,j,k);
            s_y(k)  = s_y(k)  + (((j) - u_y(k))^2)*glcm(i,j,k);
            corp(k) = corp(k) + ((i)*(j)*glcm(i,j,k));
            corm(k) = corm(k) + (((i) - u_x(k))*((j) - u_y(k))*glcm(i,j,k));
            out.cprom(k) = out.cprom(k) + (((i + j - u_x(k) - u_y(k))^4)*...
                glcm(i,j,k));
            out.cshad(k) = out.cshad(k) + (((i + j - u_x(k) - u_y(k))^3)*...
                glcm(i,j,k));
        end
    end
    s_x(k) = s_x(k) ^ 0.5;
    s_y(k) = s_y(k) ^ 0.5;
    out.autoc(k) = corp(k);
    out.corrp(k) = (corp(k) - u_x(k)*u_y(k))/(s_x(k)*s_y(k));
    out.corrm(k) = corm(k) / (s_x(k)*s_y(k));
end
end