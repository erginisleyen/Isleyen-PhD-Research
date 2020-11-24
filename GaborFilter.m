%Runs a gabor filter and calculates the variance of the filtered image.

myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.bmp')); %gets all tif files

gabor_variance = zeros(length(myFiles),1); % array to store variance values

for k = 1:length(myFiles)
    
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    gaborFileName = fullfile('Example/example', baseFileName);
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
    disp (gabor_variance);
end
