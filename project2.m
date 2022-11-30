function project2
    %data = load("C:\Users\Anthony\OneDrive\Documents\MATLAB\New Folder\CS170_Small_Data__90.txt");
    % Display a welcome message and get user input
    disp(['Welcome to the Feature Selection Algorithm.' 10])
    prompt = "Type in the name of the file to test: ";
    file_name = input(prompt, 's');
    disp(['Type in the number of the algorithm you want to run.' 10 ...
          9 '1) Forward Selection' 10 ...
          9 '2) Backward Elimination '])
    algorithm_choice = input('');
    
    % Display the amount of features and instances
    data = load(file_name);
    num_features = num2str(size(data, 2) - 1);
    num_instances = size(data, 1);
    disp(['This dataset has ', num2str(num_features), ' features (not' ...
          ' including the class attribute), with ', num2str(num_instances), ...
          ' instances.' 10])

    % Perform the chosen feature search
    if algorithm_choice == 1
        % Display accuracy from k-fold cross validation using all features
        listz = (1:size(data, 2) - 2);
        initial_accuracy = leave_one_out_cross_validation(data, listz, size(data, 2));
        disp(['Running nearest neighbor with all ', num2str(num_features), ...
              ' features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of ', num2str(initial_accuracy * 100), '%'])
        
        % Display accuracy from k-fold cross validation using no features
        listz = [];
        initial_accuracy = leave_one_out_cross_validation(data, listz, -1);
        disp(['Running nearest neighbor with 0 ', ...
              'features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of ', num2str(initial_accuracy * 100), '%' 10])
        
        % Begin forward search
        disp(['Beginning search.' 10])
        feature_search(data);
    else
        % Display accuracy from k-fold cross validation using no features
        listz = (1);
        initial_accuracy = backward_leave_one_out_cross_validation(data, listz, 2);
        disp(['Running nearest neighbor with 0 ', ...
              'features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of ', num2str(initial_accuracy * 100), '%'])

        % Display accuracy from k-fold cross validation using all features
        listz = (1:size(data, 2) - 1);
        initial_accuracy = backward_leave_one_out_cross_validation(data, listz, 0);
        disp(['Running nearest neighbor with all ', num2str(num_features), ...
              ' features, using "leave-one-out" evaluation, I get an ' ...
              'accuracy of ', num2str(initial_accuracy * 100), '%' 10])
        
        % Begin backward elimination
        disp(['Beginning backward search.' 10])
        backward_feature_search(data);
    end
end

% Function to calculate K-fold cross validation
function accuracy = leave_one_out_cross_validation(data, current_set, feature_to_add)
    % Create List with features being used and adjust values by 1
    features = current_set + 1;
    if feature_to_add ~= -1
        features = [features feature_to_add];
    end
    % Take the columns not being considered and set them to zero
    list_features = (2:size(data, 2));
    not_features = setxor(features, list_features);
    data(:, not_features) = 0;
   
    number_correctly_classified = 0;

    % Loop through the dataset columns
    for i = 1 : size(data, 1)
        object_to_classify = data(i, 2:end);
        label_object_to_classify = data(i, 1);

        nearest_neighbor_distance = inf;
        nearest_neighbor_location = inf;

        % For each feature, consider distance to every other feature
        for k = 1 : size(data, 1)
            % Compute the euclidean distance for each neighbor except onseself
            if k ~= i
                distance = sqrt(sum((object_to_classify - data(k, 2:end)).^2));

                % Update the nearest neighbor if one is found
                if distance < nearest_neighbor_distance
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k; 
                    nearest_neighbor_label = data(nearest_neighbor_location, 1);
                end
            end
        end

         % Increment the objects classified correctly
        if label_object_to_classify == nearest_neighbor_label
            number_correctly_classified = number_correctly_classified + 1;
        end
    end

    % Calculate the accuracy of the k-fold cross validation
    accuracy = number_correctly_classified / size(data, 1);
end

% Function to search for the best features to add
function feature_search(data)
    % Declare and initialize variables
    current_set_of_features = [];
    best_set_of_features = [];
    best_accuracy = 0;
    warning_displayed = 0;

    % Loop through the levels for each feature
    for i = 1 : size(data, 2) - 1 
        feature_to_add_at_this_level = [];
        best_so_far_accuracy = 0;

        % Consider adding every other feature
        for k = 1 : size(data, 2) - 1
            if isempty(intersect(current_set_of_features, k))
                % calculate the accuracy for the feature being considered
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1);
                printList = regexprep(mat2str(current_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
                if isempty(current_set_of_features)
                    disp([9 'using feature(s) {', num2str(k), printList, '}', ...
                          ' accuracy is ', num2str(accuracy * 100), '%'])
                else
                    disp([9 'using feature(s) {', num2str(k), ',', printList, '}', ...
                          ' accuracy is ', num2str(accuracy * 100), '%'])
                end
                
                % Update the best accuracy and the feature being added
                if accuracy > best_so_far_accuracy
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                end
            end
        end

        % Store the new feature for the search
        if (best_so_far_accuracy < best_accuracy) && (warning_displayed == 0)
            warning_displayed = 1;
            disp([10 '(Warning, Accuracy has decreased! Continuing search' ...
                   ' in case of local maxima)'])
        end
        current_set_of_features(i) = feature_to_add_at_this_level;
        printList = regexprep(mat2str(current_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
        disp([10 'Feature set {', printList, '}' ...
              ' was best, accuracy is ' num2str(best_so_far_accuracy * 100), '%' 10])
        
        % Update best accuracy and feature set
        if best_so_far_accuracy > best_accuracy
                best_accuracy = best_so_far_accuracy;
                best_set_of_features = current_set_of_features;
        end

    end

    % Output the result of the search
    printList = regexprep(mat2str(best_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
    disp(['Finished search!! The best feature subset is {', ...
          printList, '} Which had an accuracy of ', num2str(best_accuracy * 100), '%'])
end

% Function to perform backward elimination
function backward_feature_search(data)
    % Declare and initialize variables
    current_set_of_features = (1:size(data, 2) - 1);
    best_set_of_features = [];
    best_accuracy = 0;
    warning_displayed = 0;

    % Loop through the levels for the features
    for i = 1 : size(data, 2) - 1
        feature_to_remove_at_this_level = [];
        best_so_far_accuracy = 0;

        % Consider removing each feature
        for k = 1 : size (data, 2) - 1
            % Consider Removing the feature only if it is in the current set
            if ~isempty(intersect(current_set_of_features, k))
                % Calculate the accuracy resulting from the removed feature
                accuracy = backward_leave_one_out_cross_validation(data, current_set_of_features, k + 1);
                temp_print_features = current_set_of_features;
                temp_print_features(temp_print_features == k) = [];
                printList = regexprep(mat2str(temp_print_features), {'\[', '\]', '\s+'}, {'', '', ','});
                if isempty(temp_print_features)
                    disp([9 'using feature(s) { }', ...
                          ' accuracy is ', num2str(accuracy * 100), '%'])
                else
                    disp([9 'using feature(s) {', printList, '}', ...
                          ' accuracy is ', num2str(accuracy * 100), '%'])
                end

                % Update the best accuracy and the feature being removed
                if accuracy > best_so_far_accuracy
                    best_so_far_accuracy = accuracy;
                    feature_to_remove_at_this_level = k;
                end
            end
        end
        % Remove the feature for the search
        if (best_so_far_accuracy < best_accuracy) && (warning_displayed == 0)
            warning_displayed = 1;
            disp([10 '(Warning, Accuracy has decreased! Continuing search' ...
                   ' in case of local maxima)'])
        end
        current_set_of_features(current_set_of_features == feature_to_remove_at_this_level) = [];
        printList = regexprep(mat2str(current_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
        if isempty(current_set_of_features)
            disp([10 'Feature set { } was best, accuracy is ' ...
                num2str(best_so_far_accuracy * 100), '%' 10])
        else
        disp([10 'Feature set {', printList, '} was best, accuracy is ' ...
            num2str(best_so_far_accuracy * 100), '%' 10])
        end
        % Update best accuracy and feature set
        if best_so_far_accuracy > best_accuracy
                best_accuracy = best_so_far_accuracy;
                best_set_of_features = current_set_of_features;
        end
    end

    % Output the result of the search
    printList = regexprep(mat2str(best_set_of_features), {'\[', '\]', '\s+'}, {'', '', ','});
    disp(['Finished backward search!! The best feature subset is {', ...
          printList, '} Which had an accuracy of ', num2str(best_accuracy * 100), '%'])
end     

function accuracy = backward_leave_one_out_cross_validation(data, current_set, feature_to_remove)
    % Create a list with the features being used and adjust values by 1
    features = current_set + 1;
    features(features == feature_to_remove) = [];

    % Take the columns not being considered and set them to zero
    list_features = (2:size(data, 2));
    not_features = setxor(features, list_features);
    data(:, not_features) = 0;
    
    number_correctly_classified = 0;

    % Loop through the dataset columns
    for i = 1 : size(data, 1)
        object_to_classify = data(i, 2:end);
        label_object_to_classify = data(i, 1);

        nearest_neighbor_distance = inf;
        nearest_neighbor_location = inf;

        % For each feature, consider distance to every other feature
        for k = 1 : size(data, 1)
            % Compute the euclidean distance for each neighbor except onseself
            if k ~= i
                distance = sqrt(sum((object_to_classify - data(k, 2:end)).^2));

                % Update the nearest neighbor if one is found
                if distance < nearest_neighbor_distance
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k; 
                    nearest_neighbor_label = data(nearest_neighbor_location, 1);
                end
            end
        end

        % Increment the objects classified correctly
        if label_object_to_classify == nearest_neighbor_label
            number_correctly_classified = number_correctly_classified + 1;
        end
    end

    % Calculate the accuracy of the k-fold cross validation
    accuracy = number_correctly_classified / size(data, 1);
end