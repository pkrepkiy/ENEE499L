
% Author: Peter Krepkiy (pkrepkiy@umd.edu, krepkiyp@yahoo.com), 

% Last edit: March 11, 2024

% Revision: 0

% Description:

% This program reads .tif images for the purposes of extracting information
% that is useful for research efforts with regard to the Long Solenoid
% Experiment (LSE) at the University of Maryland Institute for Research in
% Electronics and Applied Physics


%
% Establish directories and control variables
%

close all; clear; clc

warning('off','all')


fprintf('Generate figures? (Default value: Y)\n')

output_img = input('Y or N: ','s');

if isempty(output_img)
    output_img = 'Y';
elseif ~isa(output_img,'char') | (output_img ~= 'Y' & output_img ~= 'N')

    error('Invalid input. Enter Y or N for figure output')

end


fprintf('Background subtraction and cropping? (Default value: Y)\n')

back_sub_and_img_crop = input('Y or N: ','s');

if isempty(back_sub_and_img_crop)
    back_sub_and_img_crop = 'Y';
elseif ~isa(back_sub_and_img_crop,'char') | (back_sub_and_img_crop ~= 'Y' & back_sub_and_img_crop ~= 'N')

    error('Invalid input. Enter Y or N for background subtraction and crop')

end


disp('---------------------------------------------------');

fprintf('\nSet the threshold multiplier. The algorithm searches for pixels that \nare lower than this threshold to form the beam object. The formula is like so: \nthreshold = maxIntensity - maxIntensity*thresholdMultiplier. \nThis value must be between 0 and 1.\n\n');

thresholdMultiplier = input('Enter intensity cutoff threshold or enter to skip (default is 1/sqrt(2)): ');

if isempty(thresholdMultiplier)
    fprintf('\nUsing default value: 1/sqrt(2)\n');
   thresholdMultiplier = 1/sqrt(2);
else

    % Check if the input is a valid integer
    if isnan(thresholdMultiplier) || (thresholdMultiplier >= 1) || (thresholdMultiplier <= 0)

        error('Invalid input. Please enter a valid number between 0 and 1.');
    end
end


%
% Set number of edge markers for beam outline (default 360)
%
edgeMarkerNum = 360;


disp('---------------------------------------------------');
fprintf('\nChange the depth of search for object (search sensitivity).\nThis value searches N pixels past\nthe first pixel with a value lower than the threshold. Increase for beam\nimages that are less dense or have more holes.\n\n')

searchDepth = input('Enter search depth or enter to skip (default is 20): ');


if isempty(searchDepth)
    fprintf('\nUsing default value: 20\n');
   searchDepth = 20;
else

    % Check if the input is a valid integer
    if isnan(searchDepth) || (searchDepth <= 0)

        error('Invalid input. Please enter a valid number greater than 0');
    end
end

disp('---------------------------------------------------');
fprintf('\nDefine the threshold to reject objects that have less length\nalong X and Y than the reject value. Any object that has X or Y distance\nless than N will be rejected. Increase this value if there is a lot of\nbright FOD or artifacts. Decrease for slits or smaller beams.\n\n')

rejectThresholdX = input('Enter reject size threshold in X or enter to skip (default is 15): ');

if isempty(rejectThresholdX)
    fprintf('\nUsing default X reject value: 15\n\n');
    rejectThresholdX = 15;
else

    % Check if the input is a valid integer
    if isnan(searchDepth) || (searchDepth <= 0)

        error('Invalid input. Please enter a valid number greater than 0');
    end
end

rejectThresholdY = input('Enter reject size threshold in Y or enter to skip (default is 15): ');

if isempty(rejectThresholdY)
    fprintf('\nUsing default Y reject value: 15\n');
    rejectThresholdY = 15;
else

    % Check if the input is a valid integer
    if isnan(searchDepth) || (searchDepth <= 0)

        error('Invalid input. Please enter a valid number greater than 0');
    end
end

calY = 0.0566; % VERTICAL CALIBRATION in mm/pixel

calX = 0.0566; % HORIZONTAL CALIBRATION in mm/pixel







fprintf('\n')
disp('---------------------------------------------------');
disp(['Searching for files in working directory ' pwd ' ...'])


%
% Search all .tif files in the working directory
%
files = dir(fullfile(pwd, '*.tif'));


if length(files) > 1

    %
    % Close all opened files, if any
    %
    fclose('all');


    %
    % Open a .csv object for output to a CSV if more than one .tif
    %
    
    csv_file = [datestr(now,'dd_mmm_yyyy_HH_MM') '.csv'];
    fid = fopen(csv_file, 'w');

    fprintf(fid,'Image Name,First Moment X (pixel),First Moment Y (pixel),XF_rms (mm),YF_rms (mm),2XF_rms (mm),2YF_rms (mm),xxVar (mm^2),yyVar (mm^2),2(xxVar)^2 (mm^4),2(yyVar)^2 (mm^4)\n');


end


count = 0;

for i = 1:length(files)

    currFileName = files(i).name;
    [~, ~, fileExtension] = fileparts(currFileName);
    
    if strcmpi(fileExtension, '.tif')

        count = count +1;
        
        image = imread(currFileName);
        grayImage = im2gray(image);
        
        fprintf('\n')
        disp(['Processing image ' num2str(currFileName) ' ...'])


        %
        % Find max and set the initial cropped image
        %
        [maxIntensity, linearIndex] = max(grayImage(:));
        [yMaxCoord, xMaxCoord] = ind2sub(size(grayImage), linearIndex);

        %
        % Reject maximum intensities that are within 5 pixels of the border
        %
        if (yMaxCoord < 5 || yMaxCoord > size(gray,1) - 5) || (xMaxCoord < 5 || xMaxCoord > size(grayImage,2) - 5)

            
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
        

        if back_sub_and_img_crop == 'Y'
    
            binaryMask = grayImage < maxIntensity- (maxIntensity*thresholdMultiplier);
    
            blackImage = grayImage;
    
            blackImage(binaryMask) = 0;  

        else
            blackImage = grayImage;

        end


        %
        % Set gain threshold for filter and number of edge markers
        %
        threshold = maxIntensity-(maxIntensity*thresholdMultiplier);


        edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, blackImage, threshold, edgeMarkerNum,searchDepth);



        %
        % Get min and max coordinates of the edge outline
        %
        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));
        
        %
        % Object / max Intensity rejection
        %
       

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
        
                edgeCoords = createEdgeMarkers(xMaxCoord, yMaxCoord, blackImage, threshold, edgeMarkerNum,searchDepth);
        
        
                %
                % Get new min and max coordinates of the edge outline
                %
                minObjectX = min(edgeCoords(:,1));
                maxObjectX = max(edgeCoords(:,1));
                minObjectY = min(edgeCoords(:,2));
                maxObjectY = max(edgeCoords(:,2));
        
            end
        
        end

        [X,Y]=meshgrid(1:size(blackImage,2), 1:size(blackImage,1));


        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));        


        % Calculate the Geometric centroid in X and Y
        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));


        %
        % Create new edge coordinates based on the geometric centroid (to
        % prevent shadows)
        %

        edgeCoords = createEdgeMarkers(GeometricCentroidX, GeometricCentroidY, blackImage, threshold, edgeMarkerNum,searchDepth);
        

        [X,Y]=meshgrid(1:size(blackImage,2), 1:size(blackImage,1));


        % Find which points are inside the object's outline
        inside = inpolygon(X(:), Y(:), edgeCoords(:,1), edgeCoords(:,2));
        

        %
        % Calculate NEW geometric centroids
        %

        GeometricCentroidX = mean(X(inside));
        GeometricCentroidY = mean(Y(inside));

        % This creates a logical array- points inside the object are a "1"
        % and points outside the object are a "0"
        propInside = reshape(inside,size(blackImage,1),size(blackImage,2));


        if back_sub_and_img_crop == 'Y'

        % Mask the original image with the logical array
        cropImage = grayImage .* uint16(propInside);

        else
            cropImage = grayImage;
        
        end
        
        % Set zero values to NaN to be omitted from intensity centroid
        % calculation
        % cropImage(cropImage == 0) = NaN;

        minObjectX = min(edgeCoords(:,1));
        maxObjectX = max(edgeCoords(:,1));
        minObjectY = min(edgeCoords(:,2));
        maxObjectY = max(edgeCoords(:,2));



        % Total pixel intensity sums
        I_0 = sum(sum(cropImage(:,:)));


        %% Calculate first moment in X and Y

        % Create row vector from row dimension of image
        xList = uint64(1:size(cropImage,2));

        % Create column vector (transpose) from column dimension of image
        yList = uint64((1:size(cropImage,1)))';

        % Create vector of Y indices that are inside the beam
        yVec = uint64(round(minObjectY):round(maxObjectY));

        % Create vector of X indices that are inside the beam
        xVec = uint64(round(minObjectX):round(maxObjectX));


        % Calculate X moment (centroid) by multiplying each row by the
        % beam value at that position, summing over each row and column,
        % dividing by sum of intensity
        x_avg0 = sum(sum(xList .* uint64(cropImage(yVec,:))),2)/I_0;

        % Repeat for Y moment (centroid)
        y_avg0 = sum(sum(yList .* uint64(cropImage(:,xVec)),1))/I_0;

        %% Calculate second moment in X and Y

        XF_rms = sqrt(sum(sum(((xList-x_avg0).^2) .* uint64(cropImage(yVec,:)),2))/I_0)*calX;

        % Repeat for Y second moment
        YF_rms = sqrt(sum(sum(((yList-y_avg0).^2) .* uint64(cropImage(:,xVec)),1))/I_0)*calY;

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
        % Plotting
        %

        if output_img == 'Y'

            figure(i)
           
            imshow(grayImage), title(num2str(currFileName))
        
            hold on
    
            plot(xMaxCoord,yMaxCoord, 'k*', 'MarkerSize', 25);
    
            plot(GeometricCentroidX,GeometricCentroidY,'b+','MarkerSize',25);
    
            plot(x_avg0, y_avg0,'gx','MarkerSize',25)
    
            plot(edgeCoords(:, 1), edgeCoords(:, 2), 'ro-','MarkerSize',3);
    
            plot(minObjectX,minObjectY,'cd')
            plot(maxObjectX,maxObjectY,'cd')
            plot(minObjectX,maxObjectY,'cd')
            plot(maxObjectX,minObjectY,'cd')
    
    
            legend('Peak intensity','Geometric Centroid','First Moment')
    
            hold off
        
        end

        fprintf('\n')

        disp(['First Moment (coordinate in X): ' num2str(round(x_avg0))])
        disp(['First Moment (coordinate in Y): ' num2str(round(y_avg0))])

        % disp(['XF_rms: ' num2str(XF_rms) ' mm'])
        % disp(['YF_rms: ' num2str(YF_rms) ' mm'])


        disp(['2*XF_rms: ' num2str(X2F_rms) ' mm'])
        disp(['2*YF_rms: ' num2str(Y2F_rms) ' mm'])


        disp(['xxVar: ' num2str(xxVar) ' mm^2'])
        disp(['yyVar: ' num2str(yyVar) ' mm^2'])

        % disp(['2(xxVar)^2: ' num2str(Squared_xxVar_2) ' mm^4'])
        % disp(['2(yyVar)^2: ' num2str(Squared_yyVar_2) ' mm^4'])



        %
        % Write to .csv
        %
        
        if length(files) > 1

            fprintf(fid, '%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', num2str(currFileName), y_avg0, x_avg0,XF_rms, YF_rms,X2F_rms,Y2F_rms,xxVar,yyVar,Squared_xxVar_2, Squared_yyVar_2);
            
        end

        % thresholdMultiplier = thresholdMultiplier + 0.001;

        end
    end

    % end


if length(files) > 1

    %
    % Close the CSV file
    %
    fclose(fid);

end


function outCoords = createEdgeMarkers(xCoord, yCoord, blackImage, threshold, numMarkers,searchDepth)


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
            while ((newCoord(2) + searchDepth*dy <= size(blackImage,1) && newCoord(2) + searchDepth*dy > 1)) && ((newCoord(1) + searchDepth*dx <= size(blackImage,2)) && (newCoord(1) + searchDepth*dx > 1))

                % Check current position
                if blackImage(round(newCoord(2)), round(newCoord(1))) < threshold

                    % Check up to 20 indices in front
                    checkVar = newCoord + [dx, dy];
                    count = 1;

                    while count < searchDepth && blackImage(round(checkVar(2)), round(checkVar(1))) < threshold
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


