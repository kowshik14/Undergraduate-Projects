function train()
    % Step 1: Preprocess data
    [hEOG_train, hEOG_test, vEOG_train, vEOG_test, y_train, y_test] = data_preprocessing();

    % Step 2: Extract features
    combined_train_features = feature_extraction(hEOG_train, vEOG_train);
    combined_test_features = feature_extraction(hEOG_test, vEOG_test);

    export_features(combined_train_features, y_train, 'extracted_features_train');
    export_features(combined_test_features, y_test, 'extracted_features_test');
    fprintf("Extracted features exported to the 'results' folder\n");

    % Step 3: Build model
    model = build_model();

    % Step 4: Train model
    model = train_model(model, combined_train_features, y_train);

    % Step 5: Evaluate model
    evaluate_model(model, combined_test_features, y_test);

    % Step 6: Save model
    save_model(model, './results/random_forest_model.mat');
end

function export_features(combined_features, y, filename)
    T = array2table(combined_features);
    T.target = y;
    writetable(T, ['./results/', filename, '.csv']);
end
