function fake_currency_detection()
    % Main function for fake Indian currency detection with visual output
    
    % Clear workspace and close all figures
    close all;
    clear all;
    clc;
    
    % Display welcome message
    disp('Indian Rupee (₹) Fake Currency Detection System');
    disp('--------------------------------------------');
    
    % Ask user to input an image
    [filename, pathname] = uigetfile({'.jpg;.png;.bmp;.jpeg', 'Image Files'}, 'Select a currency note image');
    if isequal(filename, 0)
        disp('User canceled the operation');
        return;
    end
    img_path = fullfile(pathname, filename);
    
    % Read and display the original image
    original_img = imread(img_path);
    figure('Name', 'Original Image', 'NumberTitle', 'off');
    imshow(original_img);
    title('Original Currency Note');
    
    % Convert to grayscale for some analyses
    gray_img = rgb2gray(original_img);
    
    % Convert to HSV color space for color-based features
    hsv_img = rgb2hsv(original_img);
    
    % Initialize results structure
    results = struct();
    
    % Create figure for feature analysis
    feature_fig = figure('Name', 'Feature Analysis', 'NumberTitle', 'off', 'Position', [100 100 1200 800]);
    
    % 1. Check for Security Thread
    subplot(3, 3, 1);
    results.security_thread = check_security_thread(original_img);
    title('Security Thread Detection');
    
    % 2. Check for Watermark
    subplot(3, 3, 2);
    results.watermark = check_watermark(gray_img);
    title('Watermark Detection');
    
    % 3. Check for Latent Image (for higher denominations)
    subplot(3, 3, 3);
    results.latent_image = check_latent_image(gray_img);
    title('Latent Image Detection');
    
    % 4. Check for Micro-lettering
    subplot(3, 3, 4);
    results.micro_lettering = check_micro_lettering(gray_img);
    title('Micro-lettering Detection');
    
    % 5. Check for Color-shifting Ink (for 500 and 2000 Rs notes)
    subplot(3, 3, 5);
    results.color_shifting_ink = check_color_shifting_ink(hsv_img);
    title('Color-shifting Ink Detection');
    
    % 6. Check for Intaglio Printing (raised print)
    subplot(3, 3, 6);
    results.intaglio_printing = check_intaglio_printing(gray_img);
    title('Intaglio Printing Detection');
    
    % 7. Check for Identification Mark (for visually impaired)
    subplot(3, 3, 7);
    results.identification_mark = check_identification_mark(gray_img);
    title('Identification Mark Detection');
    
    % 8. Check for Serial Number and its properties
    subplot(3, 3, 8);
    results.serial_number = check_serial_number(original_img);
    title('Serial Number Detection');
    
    % Display final results
    final_result = display_results(results);
    
    % Display final classification on original image
    figure('Name', 'Final Result', 'NumberTitle', 'off');
    imshow(original_img);
    if final_result
        text(size(original_img, 2)/2, size(original_img, 1)/2, 'GENUINE', ...
            'Color', 'green', 'FontSize', 50, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    else
        text(size(original_img, 2)/2, size(original_img, 1)/2, 'FAKE', ...
            'Color', 'red', 'FontSize', 50, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
    title('Final Classification');
end

%% Feature Detection Functions (Updated to show visualizations)

function result = check_security_thread(img)
    % Check for security thread - a silver line in Indian currency
    % Convert to HSV color space
    hsv_img = rgb2hsv(img);
    
    % Define thresholds for silver/gold color in HSV
    silver_mask = (hsv_img(:,:,1) >= 0.1) & (hsv_img(:,:,1) <= 0.2) & ...
                 (hsv_img(:,:,2) >= 0.3) & (hsv_img(:,:,2) <= 0.7) & ...
                 (hsv_img(:,:,3) >= 0.7);
    
    % Find connected components
    cc = bwconncomp(silver_mask);
    stats = regionprops(cc, 'Orientation', 'MajorAxisLength', 'MinorAxisLength', 'BoundingBox');
    
    % Visualize detection
    imshow(img);
    hold on;
    
    % Check if we found a vertical line with appropriate aspect ratio
    found = false;
    for i = 1:length(stats)
        if abs(stats(i).Orientation) > 80 && ...
           (stats(i).MajorAxisLength / stats(i).MinorAxisLength) > 10
            rectangle('Position', stats(i).BoundingBox, ...
                     'EdgeColor', 'r', 'LineWidth', 2);
            found = true;
            break;
        end
    end
    hold off;
    
    result.detected = found;
    if found
        result.message = 'Security thread detected.';
        result.score = 0.9;
    else
        result.message = 'Security thread NOT detected.';
        result.score = 0.1;
    end
end

function result = check_watermark(gray_img)
    % Check for watermark (Gandhi portrait in Indian currency)
    enhanced_img = adapthisteq(gray_img);
    [cA, ~, ~, ~] = dwt2(enhanced_img, 'haar');
    watermark_region = cA > 0.7 * max(cA(:));
    
    % Visualize detection
    imshow(gray_img);
    hold on;
    
    if sum(watermark_region(:)) > 0.05 * numel(watermark_region)
        faceDetector = vision.CascadeObjectDetector();
        bbox = step(faceDetector, imresize(im2uint8(mat2gray(cA)), [100 100]));
        
        if ~isempty(bbox)
            % Scale bbox back to original coordinates
            bbox = bbox * size(cA,1)/100;
            rectangle('Position', [bbox(1), bbox(2), bbox(3), bbox(4)], ...
                     'EdgeColor', 'g', 'LineWidth', 2);
            result.detected = true;
            result.message = 'Watermark detected.';
            result.score = 0.85;
        else
            result.detected = false;
            result.message = 'Watermark region but no face.';
            result.score = 0.5;
        end
    else
        result.detected = false;
        result.message = 'No watermark detected.';
        result.score = 0.1;
    end
    hold off;
end

function result = check_latent_image(gray_img)
    % Check for latent image (denomination numeral visible when held flat)
    edge_img = edge(gray_img, 'Sobel');
    numeral_area = bwareaopen(edge_img, 50);
    
    % Visualize detection
    imshow(gray_img);
    hold on;
    
    if sum(numeral_area(:)) > 1000
        [B,~] = bwboundaries(numeral_area, 'noholes');
        for k = 1:length(B)
            boundary = B{k};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
        end
        result.detected = true;
        result.message = 'Latent image detected.';
        result.score = 0.6;
    else
        result.detected = false;
        result.message = 'Latent image NOT detected.';
        result.score = 0.4;
    end
    hold off;
end

function result = check_micro_lettering(gray_img)
    % Check for micro-lettering (RBI and denomination value in small letters)
    edge_img = edge(gray_img, 'canny', [0.01 0.2], 1);
    window_size = 20;
    density = conv2(double(edge_img), ones(window_size)/window_size^2, 'same');
    micro_text_regions = density > 0.3;
    
    % Visualize detection
    imshow(gray_img);
    hold on;
    
    if sum(micro_text_regions(:)) > 0.01 * numel(micro_text_regions)
        [B,~] = bwboundaries(micro_text_regions, 'noholes');
        for k = 1:length(B)
            boundary = B{k};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
        end
        result.detected = true;
        result.message = 'Micro-lettering detected.';
        result.score = 0.85;
    else
        result.detected = false;
        result.message = 'Micro-lettering NOT detected.';
        result.score = 0.15;
    end
    hold off;
end

function result = check_color_shifting_ink(hsv_img)
    % Check for color-shifting ink (for 500 and 2000 Rs notes)
    hue_variance = stdfilt(hsv_img(:,:,1), ones(5));
    color_shift_regions = hue_variance > 0.15;
    
    % Visualize detection
    imshow(hsv2rgb(hsv_img));
    hold on;
    
    if sum(color_shift_regions(:)) > 0.01 * numel(color_shift_regions)
        [B,~] = bwboundaries(color_shift_regions, 'noholes');
        for k = 1:length(B)
            boundary = B{k};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
        end
        result.detected = true;
        result.message = 'Color-shifting ink detected.';
        result.score = 0.85;
    else
        result.detected = false;
        result.message = 'Color-shifting ink NOT detected.';
        result.score = 0.2;
    end
    hold off;
end

function result = check_intaglio_printing(gray_img)
    % Check for intaglio printing (raised print feel)
    edge_img = edge(gray_img, 'canny', [0.01 0.05], 1);
    edge_density = sum(edge_img(:)) / numel(edge_img);
    
    % Visualize detection
    imshow(gray_img);
    hold on;
    
    if edge_density > 0.15
        [B,~] = bwboundaries(edge_img, 'noholes');
        for k = 1:length(B)
            boundary = B{k};
            plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1);
        end
        result.detected = true;
        result.message = 'Intaglio printing detected.';
        result.score = 0.8;
    else
        result.detected = false;
        result.message = 'Intaglio printing NOT detected.';
        result.score = 0.3;
    end
    hold off;
end

function result = check_identification_mark(gray_img)
    % Check for identification mark (for visually impaired)
    blobAnalyzer = vision.BlobAnalysis('AreaOutputPort', true, ...
                                      'CentroidOutputPort', false, ...
                                      'BoundingBoxOutputPort', true, ...
                                      'MinimumBlobArea', 50, ...
                                      'MaximumBlobArea', 500);
    [~, bboxes] = step(blobAnalyzer, imbinarize(gray_img, 'adaptive'));
    
    % Visualize detection
    imshow(gray_img);
    hold on;
    
    valid_marks = 0;
    for i = 1:size(bboxes, 1)
        aspect_ratio = bboxes(i,3) / bboxes(i,4);
        if aspect_ratio > 0.8 && aspect_ratio < 1.2
            rectangle('Position', bboxes(i,:), 'EdgeColor', 'g', 'LineWidth', 2);
            valid_marks = valid_marks + 1;
        end
    end
    
    if valid_marks >= 1
        result.detected = true;
        result.message = 'Identification mark detected.';
        result.score = 0.7;
    else
        result.detected = false;
        result.message = 'Identification mark NOT detected.';
        result.score = 0.2;
    end
    hold off;
end

function result = check_serial_number(img)
    % Check for serial number presence and format
    gray_img = rgb2gray(img);
    bin_img = ~imbinarize(gray_img, 'adaptive');
    ocr_results = ocr(bin_img);
    serial_pattern = '[A-Z]{2}\d{6}[A-Z]?';
    serial_numbers = regexp(ocr_results.Text, serial_pattern, 'match');
    
    % Visualize detection
    imshow(img);
    hold on;
    
    if ~isempty(serial_numbers)
        text_regions = ocr_results.WordBoundingBoxes;
        for i = 1:size(text_regions, 1)
            rectangle('Position', text_regions(i,:), 'EdgeColor', 'g', 'LineWidth', 2);
        end
        result.detected = true;
        result.message = sprintf('Serial number: %s', serial_numbers{1});
        result.score = 0.9;
    else
        text_regions = detectMSERFeatures(gray_img);
        if ~isempty(text_regions)
            plot(text_regions);
            result.detected = true;
            result.message = 'Text regions found.';
            result.score = 0.6;
        else
            result.detected = false;
            result.message = 'No serial number detected.';
            result.score = 0.1;
        end
    end
    hold off;
end

%% Results Display Function (returns final classification)

function is_genuine = display_results(results)
    % Calculate overall authenticity score
    weights = [0.25, 0.15, 0.1, 0.10, 0.1, 0.1, 0.1, 0.1];
    scores = [results.security_thread.score, results.watermark.score, ...
              results.latent_image.score, results.micro_lettering.score, ...
              results.color_shifting_ink.score, results.intaglio_printing.score, ...
              results.identification_mark.score, results.serial_number.score];
    
    overall_score = sum(weights .* scores);
    
    % Display results
    fprintf('\n=== Currency Note Analysis Results ===\n');
    fprintf('1. Security Thread: %s (Score: %.2f)\n', results.security_thread.message, results.security_thread.score);
    fprintf('2. Watermark: %s (Score: %.2f)\n', results.watermark.message, results.watermark.score);
    fprintf('3. Latent Image: %s (Score: %.2f)\n', results.latent_image.message, results.latent_image.score);
    fprintf('4. Micro-lettering: %s (Score: %.2f)\n', results.micro_lettering.message, results.micro_lettering.score);
    fprintf('5. Color-shifting Ink: %s (Score: %.2f)\n', results.color_shifting_ink.message, results.color_shifting_ink.score);
    fprintf('6. Intaglio Printing: %s (Score: %.2f)\n', results.intaglio_printing.message, results.intaglio_printing.score);
    fprintf('7. Identification Mark: %s (Score: %.2f)\n', results.identification_mark.message, results.identification_mark.score);
    fprintf('8. Serial Number: %s (Score: %.2f)\n', results.serial_number.message, results.serial_number.score);
    
    fprintf('\n=== Overall Authenticity Score: %.2f/1.00 ===\n', overall_score);
    
    if overall_score > 0.66
        fprintf('\nCONCLUSION: The note appears to be GENUINE.\n');
        is_genuine = true;
    else
        fprintf('\nCONCLUSION: The note appears to be FAKE.\n');
        is_genuine = false;
    end
end
