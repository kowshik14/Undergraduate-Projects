function model = build_model()
    model = TreeBagger(100, [], 'target', 'Method', 'classification');  % Random Forest model
end

function model = train_model(model, X_train, y_train)
    model = train(model, X_train, y_train);
end

function evaluate_model(model, X_test, y_test)
    y_pred = predict(model, X_test);
    accuracy = sum(strcmp(y_test, y_pred)) / length(y_test);
    fprintf('Accuracy: %.2f\n', accuracy);
    % Generate classification report if necessary
end

function save_model(model, model_path)
    save(model_path, 'model');
end

function model = load_model(model_path)
    loaded = load(model_path);
    model = loaded.model;
end
