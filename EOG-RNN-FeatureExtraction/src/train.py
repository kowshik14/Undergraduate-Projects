from data_preprocessing import preprocess_data, export_features
from feature_extraction import extract_features
from model import build_model, train_model, evaluate_model, save_model
import pandas as pd
def main():
    # Step 1: Preprocess data
    hEOG_train, hEOG_test, vEOG_train, vEOG_test, y_train, y_test = preprocess_data()

    # Step 2: Extract features
    combined_train_features = extract_features(hEOG_train, vEOG_train)
    combined_test_features = extract_features(hEOG_test, vEOG_test)

    export_features(combined_train_features, y_train, 'extracted_features_train')
    export_features(combined_test_features, y_test, 'extracted_features_test')
    print("Extracted Features Exported to the 'results' folder")

    # Step 3: Build model
    model = build_model()

    # Step 4: Train model
    model = train_model(model, combined_train_features, y_train)

    # Step 5: Evaluate model
    evaluate_model(model, combined_test_features, y_test)

    # Step 6: Save model
    save_model(model, './results/random_forest_model.pkl')

if __name__ == "__main__":
    main()
