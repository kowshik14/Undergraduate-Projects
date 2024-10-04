function [hEOG_train, hEOG_test, vEOG_train, vEOG_test, y_train, y_test] = data_preprocessing()
    hEOG_train = load_data('./data/EOGHorizontalSignal_TRAIN.arff');
    hEOG_test = load_data('./data/EOGHorizontalSignal_TEST.arff');
    vEOG_train = load_data('./data/EOGVerticalSignal_TRAIN.arff');
    vEOG_test = load_data('./data/EOGVerticalSignal_TEST.arff');

    y_train = hEOG_train.target;
    y_test = hEOG_test.target;

    % Optionally combine features
    % X_train_combined = [hEOG_train(:, 1:end-1), vEOG_train(:, 1:end-1)];
    % X_test_combined = [hEOG_test(:, 1:end-1), vEOG_test(:, 1:end-1)];
end

function data = load_data(file_path)
    data = read_arff(file_path);  % You need to implement or find a function to read ARFF files
    data.target = double(data.target);  % Convert target to double
end
