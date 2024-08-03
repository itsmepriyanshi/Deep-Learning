import logging
from preprocessing.data_preprocessing import load_and_preprocess_data
from models.classification_model import train_model, find_best_learning_rate, evaluate_model
from visualization.model_visualization import plot_loss_curves

# Set up logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO)

def main():
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data/employee_attrition.csv')
        
        # Train model
        model, history = train_model(X_train, y_train)
        
        # Find best learning rate
        find_best_learning_rate(model, X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Plot loss curves
        plot_loss_curves(history)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
