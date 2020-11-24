%Runs a gabor filter and calculates the variance of the filtered image.
%Mean, var and stddev also calculated.
%/Commented out/ Calculates the percent of the non-background segment of
%k-means clustering result

myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.bmp')); %gets all tif files

gabor_variance = zeros(length(myFiles),1); % array to store variance values

for k = 1:length(myFiles)
    
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    gaborFileName = fullfile('NewData_411/image bands/gabor/train/low', baseFileName);
    fprintf(1, 'Now reading %s\n', baseFileName);
    
    % Reads the image, resizes and transform to gray image
    I = imread(fullFileName);
    I = imresize(I,[256 256]);
    Igray = rgb2gray(I);

    imageSize = size(Igray);
    numRows = imageSize(1);
    numCols = imageSize(2);

    wavelengthMin = 4/sqrt(2);
    wavelengthMax = hypot(numRows,numCols);
    n = floor(log2(wavelengthMax/wavelengthMin));
    wavelength = 2.^(0:(n-2)) * wavelengthMin;

    deltaTheta = 30;
    orientation = 0:deltaTheta:(180-deltaTheta);

    g = gabor(wavelength,orientation);
    
    
    
    gabormag = imgaborfilt(Igray,g);

    for i = 1:length(g)
        sigma = 0.5*g(i).Wavelength;
        K = 3;
        gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),K*sigma); 
    end

    X = 1:numCols;
    Y = 1:numRows;
    [X,Y] = meshgrid(X,Y);
    featureSet = cat(3,gabormag,X);
    featureSet = cat(3,featureSet,Y);

    numPoints = numRows*numCols;
    X = reshape(featureSet,numRows*numCols,[]);

    X = bsxfun(@minus, X, mean(X));
    X = bsxfun(@rdivide,X,std(X));

    coeff = pca(X);
    feature2DImage = reshape(X*coeff(:,1),numRows,numCols);
    feature2DImageGray = mat2gray(feature2DImage);
    %figure
    %imshow(feature2DImageGray)
    imwrite(feature2DImageGray, gaborFileName);
    v = var(feature2DImageGray(:));
    gabor_variance(k,1) = v;
    %disp (gabor_variance);

    %L = kmeans(X,4,'Replicates',5);

    %L = reshape(L,[numRows numCols]);
    %figure
    %imshow(label2rgb(L))
    
    %Aseg1 = zeros(size(I),'like',I);
    %Aseg2 = zeros(size(I),'like',I);
    %BW = L == 4;
    %BW = repmat(BW,[1 1 3]);
    %Aseg1(BW) = I(BW);
    %Aseg2(~BW) = I(~BW);
    %figure
    %imshowpair(Aseg1,Aseg2,'montage');

    %Aseg1g = rgb2gray(Aseg1);
    %Aseg2g = rgb2gray(Aseg2);

    %Aseg1g_totpx = numel( Aseg1g );           %total number of pixel of img1
    %Aseg1g_NB = length( Aseg1g(Aseg1g~=0) );   %number of pixel not black of img1
    %Aseg2g_totpx = numel( Aseg2g );           %total number of pixel of img2
    %Aseg2g_NB = length( Aseg2g(Aseg2g~=0) );   %number of pixel not black of img2
    %diffPx = abs( Aseg1g_NB - Aseg2g_NB );    %difference of pixels between the 2 imgs
    %Percentage = (abs(Aseg2g_NB))/(Aseg1g_NB + Aseg2g_NB); % percentage of non-background area
    %gabor_percentage (1,k) = Percentage;
    %display(Percentage);
end

%mean_variance = mean(gabor_variance,1);
%var_variance = var(gabor_variance,1);
%std_variance = std(gabor_variance,1);
%mean_percentage = mean(gabor_percentage,2);
%var_percentage = var(gabor_percentage,1);
%std_percentage = std(gabor_percentage,1);
