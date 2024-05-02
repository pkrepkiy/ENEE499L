
% Author: Peter Krepkiy (pkrepkiy@umd.edu, krepkiyp@yahoo.com)

% Last edit: May 1, 2024

% Revision: 0

% Description:

% This program reads .tif images for the purposes of extracting information
% that is useful for research efforts with regard to the Long Solenoid
% Experiment (LSE) at the University of Maryland Institute for Research in
% Electronics and Applied Physics

% There are two modes for background subtraction and cropping:
% MODE 1: MEAN VALUE SUBTRACTION
%
% This mode detects a beam object, defined by a border, takes the average
% value of all of the pixels outside of the border, and subtracts those
% values from the entire image.
%
% MODE 2: CONSTANT VALUE SUBTRACTION
%
% This mode subtracts the background by a constant value as specified by
% the user, e.g. 10000

% The CROPPING Mode can be selected ON and OFF
%
% The cropping mode will create a beam object from the maximum intensity
% point of the image. The sensitivity of the cropping will depend on 
% the IMAGE DETECTION SENSITIVITY which is a ratio that defines how far 
% the algorithm will search from the maximum in all directions to define
% the outline of the object. That is, it defines the threshold to stop
% searching radially outwards in each direction.
% 
% The SEARCH DEPTH is another sensitivity parameter that defines the 
% distance from the first point that is below the the threshold, to 
% STOP searching. Use a higher search depth for beam images that are
% very spotty/chunky or that have non-uniform distributions.
%
% Lastly, the REJECT THRESHOLDs in X and Y will reject beam objects that
% are too small in X or Y. Use higher values for images with a lot of 
% bright artifacts/spots, 


%-------------------------------------------------------------------------

%
% Establish directories and control variables
%

close all; clear; clc

warning('off','all')


disp('---------------------------------------------------')
fprintf('GENERATE FIGURES?\n')

OUTPUT_FIG = input('Y/N [Y]: ','s');

if isempty(OUTPUT_FIG)
    OUTPUT_FIG = 'Y';
elseif ~isa(OUTPUT_FIG,'char') | (~strcmp(OUTPUT_FIG,'Y') & ~strcmp(OUTPUT_FIG,'N'))

    error('Invalid input. Enter Y or N for figure output')

end




%
% USER INPUT BACKGROUND SUBTRACTION
%

disp('---------------------------------------------------')
fprintf('BACKGROUND SUBTRACTION?\n')

BACKGROUND_SUB = input('Y/N [Y]: ','s');


while all(((~strcmp(BACKGROUND_SUB,'Y') & ~strcmp(BACKGROUND_SUB,'N')) & ~isempty(BACKGROUND_SUB)))
   
        fprintf('\nILLEGAL VALUE')
    
        BACKGROUND_SUB = input('\nY/N [Y]: ','s');

end


if isempty(BACKGROUND_SUB) | ...
    isa(BACKGROUND_SUB,'char') & (BACKGROUND_SUB == 'Y')



    BACKGROUND_SUB = 'Y';
    %
    % SPECIFY TYPE OF SUBTRACTION
    %

    disp('---------------------------------------------------') 
    fprintf('MEAN VALUE OR CONSTANT VALUE? (MEAN or CONSTANT)\n')

    subtraction_type = input('MEAN/CONST [MEAN]: ','s');

    %
    % CHECK INPUT
    %
    while all( ~isempty(subtraction_type) & (~isa(subtraction_type,'char') | ~strcmp(subtraction_type,'MEAN') & ~strcmp(subtraction_type,'CONST')))
    
        fprintf('\nInvalid input. Enter MEAN or CONSTANT for background subtraction type')
    
        subtraction_type = input('\nMEAN/CONST [MEAN]: ','s');
            
    end

    if isempty(subtraction_type)

        subtraction_type = 'MEAN';
    end

    if strcmp(subtraction_type,'CONST')

        fprintf('\nSPECIFY CONSTANT SUBTRACTION VALUE:\n')

        subtraction_value = uint32(str2double(input('INTEGER GREATER THAN ZERO: ','s')));

        while isempty(subtraction_value) | size(subtraction_value) ~= 1 | subtraction_value == 0

            fprintf('ILLEGAL VALUE\n')
            subtraction_value = uint32(str2double(input('INTEGER GREATER THAN ZERO: ','s')));

    
        end
    end

end



%
% USER INPUT IMAGE CROPPING
%

disp('---------------------------------------------------')
fprintf('IMAGE CROP?\n')


IMG_CROP = input('Y/N [Y]: ','s');


while all(((~strcmp(IMG_CROP,'Y') & ~strcmp(IMG_CROP,'N')) & ~isempty(IMG_CROP)))
   
        fprintf('\nILLEGAL VALUE')
    
        IMG_CROP = input('\nY/N [Y]: ','s');

end


if isempty(IMG_CROP) | ...
    isa(IMG_CROP,'char') & (IMG_CROP == 'Y')

    IMG_CROP = 'Y';


    disp('---------------------------------------------------');
    fprintf('SET BEAM IMAGE DETECTION SENSITIVITY. USE A HIGHER VALUE FOR HIGHER SENSITIVITY.\n')
    thresholdMultiplier = str2double(input('DETECTION SENSITIVITY BETWEEN 0 AND 1 [1/sqrt(2)]: ','s'));
    
    if isnan(thresholdMultiplier)
        fprintf('\nUSING DEFAULT VALUE: 1/sqrt(2)\n');
       thresholdMultiplier = double(1/sqrt(2));
    else
    
        % Check if the input is a valid integer
        while all(thresholdMultiplier >= 1 | thresholdMultiplier <= 0)
    
            disp(thresholdMultiplier)
            fprintf('ILLEGAL VALUE\n')
            thresholdMultiplier = input('DETECTION SENSITIVITY BETWEEN 0 AND 1 [1/sqrt(2)]: ','s');
    
        end
    end
    
    % thresholdMultiplier = str2num(thresholdMultiplier);
    disp(thresholdMultiplier)
    
    
    disp('---------------------------------------------------');
    %fprintf('\nChange the depth of search for object (search sensitivity).\nThis value searches N pixels past\nthe first pixel with a value lower than the threshold. Increase for beam\nimages that are less dense or have more holes.\n\n')
    
    
    
    fprintf('SET OBJECT SEARCH DEPTH.\nUSE A HIGHER VALUE FOR SPOTTY IMAGES/THOSE WITH NONUNIFORM DISTRIBUTION.\n')
    searchDepth = input('SEARCH DEPTH/SENSITIVITY [20]: ');
    
    
    if isempty(searchDepth)
        fprintf('\nUSING DEFAULT VALUE: 20\n');
       searchDepth = 20;
    else
    
        % Check if the input is a valid integer
        if isnan(searchDepth) || (searchDepth <= 0)
    
            error('INVALID INPUT. ENTER AN INTEGER GREATER THAN 0');
        end
    end
    
    disp('---------------------------------------------------');
    %fprintf('\nDefine the threshold to reject objects that have less length\nalong X and Y than the reject value. Any object that has X or Y distance\nless than N will be rejected. Increase this value if there is a lot of\nbright FOD or artifacts. Decrease for slits or smaller beams.\n\n')
    
    
    fprintf('SET REJECT THRESHOLD IN X.\nREJECT ALL BEAM OBJECTS THAT ARE A SMALLER SIZE IN X.\nUSE A HIGHER VALUE FOR SLIT IMAGES')
    rejectThresholdX = input('\n\nREJECT THRESHOLD IN X [15]: ');
    
    if isempty(rejectThresholdX)
        fprintf('\nUSING DEFAULT X REJECT THRESHOLD: 15\n\n');
        rejectThresholdX = 15;
    else
    
        % Check if the input is a valid integer
        if isnan(searchDepth) || (searchDepth <= 0)
    
            error('INVALID INPUT. ENTER AN INTEGER GREATER THAN 0');
        end
    end
    
    rejectThresholdY = input('REJECT THRESHOLD IN Y [15]: ');
    
    if isempty(rejectThresholdY)
        fprintf('\nUSING DEFAULT Y REJECT VALUE: 15\n');
        rejectThresholdY = 15;
    else
    
        % Check if the input is a valid integer
        if isnan(searchDepth) || (searchDepth <= 0)
    
            error('INVALID INPUT. ENTER AN INTEGER GREATER THAN 0');
        end
    end


else

    IMG_CROP = 'N';
end

%
% Search all .tif files in the working directory
%
files = dir(fullfile(pwd, '*.tif'));


if length(files) > 1

    %
    % Close all opened files, if any
    %
    fclose('all');


    disp('---------------------------------------------------')
    fprintf('ENTER EXPERIMENT DATE:\n')
    
    %
    % SANITIZE INPUT 
    %
    Exp_date = input('MM_DD_YYYY: ','s');
    Exp_date(isspace(Exp_date)) = [];

    while isempty(Exp_date)



    fprintf('ILLEGAL VALUE\n')

    Exp_date = input('MM_DD_YYYY: ','s');
    Exp_date(isspace(Exp_date)) = [];
    


    end


    %
    % Open a .csv object for output to a CSV if more than one .tif
    %
    
    csv_file = [Exp_date '.csv'];
    fid = fopen(csv_file, 'w');

    fprintf(fid,'Image Name,First Moment X (pixel),First Moment Y (pixel),XF_rms (mm),YF_rms (mm),2XF_rms (mm),2YF_rms (mm),xxVar (mm^2),yyVar (mm^2),2(xxVar)^2 (mm^4),2(yyVar)^2 (mm^4)\n');


end



calY = 0.0566; % VERTICAL CALIBRATION in mm/pixel

calX = 0.0566; % HORIZONTAL CALIBRATION in mm/pixel

disp('---------------------------------------------------');

disp('CALIBRATION VALUES:')
disp(['X: ', num2str(calX), ' mm/pixel '])
disp(['Y: ', num2str(calY),' mm/pixel'])
fprintf('\n')
disp('USE THE impixelinfo COMMAND TO INSPECT THE BEAM IMAGE')




fprintf('\n')
disp('---------------------------------------------------');
disp(['Searching for files in working directory ' pwd ' ...'])


%
% SET NUMBER OF EDGE MARKERS
%
edgeMarkerNum = 360;

count = 0;

for i = 1:length(files)

    currFileName = files(i).name;
    [~, ~, fileExtension] = fileparts(currFileName);
    
    if strcmpi(fileExtension, '.tif')

        count = count +1;
        
        image = imread(currFileName);
        grayImage = im2gray(image);
        
        fprintf('\n')
        disp(['PROCESSING IMAGE ' num2str(currFileName) ' ...'])

        %
        % Find max and set the initial cropped image
        %
        [maxIntensity, linearIndex] = max(grayImage(:));
        [yMaxCoord, xMaxCoord] = ind2sub(size(grayImage), linearIndex);

        %
        % Reject maximum intensities that are within 5 pixels of the border
        %
        if (yMaxCoord < 5 || yMaxCoord > size(grayImage,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(grayImage,2) - 5)

            
            %
            % Define a temporary image to find a lower maximum (bright FOD
            % rejection)
            %
        
            %
            % Create gaussian filter
            %
            sigma = 1;
            h = fspecial('gaussian', [5 5], sigma);
        
            %
            % Create a temporary smoothed image to reject bright FOD
            %
            smoothedImage = imfilter(grayImage, h, 'replicate');
        
            %
            % Keep decreasing the "maximum" value until larger objects are found 
            %
            while (yMaxCoord < 5 || yMaxCoord > size(grayImage,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(grayImage,2) - 5)
            
           
                smoothedImage(yMaxCoord,xMaxCoord) = NaN;
        
                [maxIntensity, linearIndex] = max(smoothedImage(:));
        
                [yMaxCoord, xMaxCoord] = ind2sub(size(smoothedImage), linearIndex);
        
            end
        end


        
        if strcmp(IMG_CROP,'Y')

        
            %% SET GAIN THRESHOLD FOR BEAM SHAPE
            
            threshold = maxIntensity-(maxIntensity*thresholdMultiplier);
    
            %% GENERATE FIRST BEAM OBJECT
    
            edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, grayImage, threshold, edgeMarkerNum,searchDepth);
    
    
            %
            % Get min and max coordinates of the edge outline
            %
            minObjectX = min(edgeCoords(:,1));
            maxObjectX = max(edgeCoords(:,1));
            minObjectY = min(edgeCoords(:,2));
            maxObjectY = max(edgeCoords(:,2));
            
    
            %% Object / max Intensity rejection
           
    
            %
            % Reject beam objects that are smaller than a particular length in
            % X and Y, 25 by default
            %
            if ((maxObjectX - minObjectX) < rejectThresholdX) || ((maxObjectY - minObjectY) < rejectThresholdY)
            
            
                %
                % Define a temporary image to find a lower maximum (bright FOD
                % rejection)
                %
            
                %
                % Create gaussian filter
                %
                sigma = 1;
                h = fspecial('gaussian', [5 5], sigma);
            
                %
                % Create a temporary smoothed image to reject bright FOD
                %
                smoothedImage = imfilter(grayImage, h, 'replicate');
            
                %
                % Keep decreasing the "maximum" value until larger objects are found 
                %
                while ((maxObjectX - minObjectX) < rejectThresholdX) || ((maxObjectY - minObjectY) < rejectThresholdY)
            
                    %
                    % Set the current maximum to NaN to find a new max.
                    %
                    smoothedImage(yMaxCoord,xMaxCoord) = NaN;
            
                    [maxIntensity, linearIndex] = max(smoothedImage(:));
            
                    threshold = maxIntensity-(maxIntensity*thresholdMultiplier);
            
            
                    [yMaxCoord, xMaxCoord] = ind2sub(size(smoothedImage), linearIndex);
            
                    edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, grayImage, threshold, edgeMarkerNum,searchDepth);
            
            
                    %
                    % Get new min and max coordinates of the edge outline
                    %
                    minObjectX = min(edgeCoords(:,1));
                    maxObjectX = max(edgeCoords(:,1));
                    minObjectY = min(edgeCoords(:,2));
                    maxObjectY = max(edgeCoords(:,2));
            
                end
            
            end
    
            [X,Y]=meshgrid(1:size(grayImage,2), 1:size(grayImage,1));
    
    
            inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));      
    
            %% Calculate the Geometric centroid in X and Y
            GeometricCentroidX = mean(X(inside));
            GeometricCentroidY = mean(Y(inside));
    
    
            %% Create new edge coordinates based on the geometric centroid (to prevent shadows)
    
    
            edgeCoords = createEdgeMarkers(GeometricCentroidX, GeometricCentroidY, grayImage, threshold, edgeMarkerNum,searchDepth);
            
    
            [X,Y]=meshgrid(1:size(grayImage,2), 1:size(grayImage,1));
    
    
            % Find which points are inside the object's outline
            inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));
    
            %
            % Calculate NEW geometric centroids
            %
    
            GeometricCentroidX = mean(X(inside));
            GeometricCentroidY = mean(Y(inside));
    
            % This creates a logical array- points inside the object are a "1"
            % and points outside the object are a "0"
            propInside = reshape(inside,size(grayImage,1),size(grayImage,2));
         
        else
        % IF NO IMG CROPPING
   

            [X,Y]=meshgrid(1:size(grayImage,2), 1:size(grayImage,1));
    
            % Find which points are inside the object's outline
            
        
            % FIX THIS LATER
            % GeometricCentroidX = mean(grayImage(X));
            % GeometricCentroidY = mean(Y);

            propInside = true(size(grayImage,1),size(grayImage,2));

        end





        %% DO BACKGROUND SUBTRACTION
        %
        if strcmp(BACKGROUND_SUB,'Y')
            
            %
            % CONST type subtracted was already ascertained in the setup
            %
            if strcmp(subtraction_type,'MEAN')

                %
                % SUBTRACT THE AVERAGE OF ALL OF THE VALUES OUTSIDE OF
                % THE BEAM OBJECT
                %
                subtraction_value = mean(mean(grayImage.*uint16(~propInside)));

            end

            SubtractedImage = grayImage - uint16(subtraction_value);
            disp(['SUBTRACTION: ' num2str(subtraction_value)]);

        else
            SubtractedImage = grayImage;

        end


        %% CROP THE IMAGE
        if strcmp(IMG_CROP,'Y')

        %
        % Mask the subtracted image with the logical array for calculations
        %
        cropImage = SubtractedImage .* uint16(propInside);

        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));



        %
        % IF USER SELECTED NO CROPPING
        %
        else
            cropImage = SubtractedImage;

            minObjectX = 1;
            maxObjectX = size(grayImage,2);
            minObjectY = 1;
            maxObjectY = size(grayImage,1);
        
        end


        % Total pixel intensity sums
        I_0 = sum(sum(cropImage(:,:)));

        %% Calculate first moment in X and Y

        % Create logical mask of non-zero values in the cropped image
        nonZeroMask = cropImage ~= 0;

        % Create row vector from row dimension of image
        xList = int64(1:size(cropImage,2));

        % Create column vector (transpose) from column dimension of image
        yList = int64((1:size(cropImage,1)))';

        % Create vector of Y indices that are inside the beam
        yVec = int64(round(minObjectY):round(maxObjectY));

        % Create vector of X indices that are inside the beam
        xVec = int64(round(minObjectX):round(maxObjectX));


        % Calculate X moment (centroid) by multiplying each row by the
        % beam value at that position, summing over each row and column,
        % dividing by sum of intensity
        x_avg0 = sum(sum(xList .* int64(cropImage(yVec,:))),2)/I_0;

        % Repeat for Y moment (centroid)
        y_avg0 = sum(sum(yList .* int64(cropImage(:,xVec)),1))/I_0;

        %% Calculate second moment in X and Y

        XF_rms = sqrt(sum(sum(((xList-x_avg0).^2) .* int64(cropImage(yVec,:)),2))/I_0)*calX;

        % Repeat for Y second moment
        YF_rms = sqrt(sum(sum(((yList-y_avg0).^2) .* int64(cropImage(:,xVec)),1))/I_0)*calY;

        %% Calculate 2XF_rms and 2YF_rms

        X2F_rms = 2*XF_rms;

        Y2F_rms = 2*YF_rms;

        %% Calculate xxVar and yyVar

        xxVar = XF_rms^2;

        yyVar = YF_rms^2;

        %% Calculate 2sqrt(xxVar) and 2sqrt(yyVar)

        Squared_xxVar_2 = 2*sqrt(xxVar);

        Squared_yyVar_2 = 2*sqrt(yyVar);

      
        %% Plot results

        %
        % PLOTTING
        %

        if OUTPUT_FIG == 'Y'

            figure(i)
           
            imshow(SubtractedImage), title(num2str(currFileName))
        
            hold on
    
            plot(xMaxCoord,yMaxCoord, 'k*', 'MarkerSize', 25);

            % plot(GeometricCentroidX,GeometricCentroidY,'b+','MarkerSize',25);
    
            plot(x_avg0, y_avg0,'gx','MarkerSize',25)
    

            if strcmp(IMG_CROP,'Y')

                plot(edgeCoords(:, 1), edgeCoords(:, 2), 'ro-','MarkerSize',3);

            else

                borderPoints = bwboundaries(grayImage);
                        
                % Plot the border points
                for k = 1:length(borderPoints)
                    boundary = borderPoints{k};
                    plot(boundary(:,2), boundary(:,1), 'ro-','MarkerSize',3);
                end
    
            end

            plot(minObjectX,minObjectY,'cd')
            plot(maxObjectX,maxObjectY,'cd')
            plot(minObjectX,maxObjectY,'cd')
            plot(maxObjectX,minObjectY,'cd')
    
    
            legend('Peak intensity','Geometric Centroid','First Moment')
    
            hold off
        
        end

        fprintf('\n')

        disp(['FIRST MOMENT IN X: ' num2str(round(x_avg0))])
        disp(['FIRST MOMENT IN Y: ' num2str(round(y_avg0))])

        disp(['2*XF_rms: ' num2str(X2F_rms) ' mm'])
        disp(['2*YF_rms: ' num2str(Y2F_rms) ' mm'])

        disp(['xxVar: ' num2str(xxVar) ' mm^2'])
        disp(['yyVar: ' num2str(yyVar) ' mm^2'])


        %
        % WRITE TO .csv IF MORE THAN ONE IMAGE
        %

        if length(files) > 1
    
            fprintf('\nWRITING TO .CSV ....\n')

            fprintf(fid, '%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', num2str(currFileName), y_avg0, x_avg0,XF_rms, YF_rms,X2F_rms,Y2F_rms,xxVar,yyVar,Squared_xxVar_2, Squared_yyVar_2);
            
        end

        end
    end

    %end


if length(files) > 1

    %
    % Close the CSV file
    %
    fclose(fid);

end


function outCoords = createEdgeMarkers(xCoord, yCoord, subtractedImage, threshold, numMarkers,searchDepth)


        % Iterate over angles to evenly divide the circle
        
        angle = linspace(0, 360, numMarkers + 1);
        angle = angle(1:end-1);  % Exclude the last element to avoid duplicates

        outCoords = zeros(numMarkers,2);

        for i= 1:numMarkers

            % Compute x and y components based on the angle
            dx = cosd(angle(i));
            dy = sind(angle(i));

            % Initialize variables
           newCoord = [xCoord, yCoord] +  [dx, dy];
    

            % Loop to move along the direction until the threshold condition is met
            while ((newCoord(2) + searchDepth*dy <= size(subtractedImage,1) && newCoord(2) + searchDepth*dy > 1)) && ((newCoord(1) + searchDepth*dx <= size(subtractedImage,2)) && (newCoord(1) + searchDepth*dx > 1))

                % Check current position
                if subtractedImage(round(newCoord(2)), round(newCoord(1))) < threshold

                    % Check up to 20 indices in front
                    checkVar = newCoord + [dx, dy];
                    count = 1;

                    while count < searchDepth && subtractedImage(round(checkVar(2)), round(checkVar(1))) < threshold
                        checkVar = checkVar + [dx, dy];
                        count = count + 1;
                    end

                    if count == searchDepth
                        break;
                    end
                end

              
                count = 0;
                newCoord = newCoord +  [dx, dy];

                
            end

            % Reset coordinates for the next iteration
            outCoords(i,:) = newCoord;
        end
end


