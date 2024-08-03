from sklearn.metrics import accuracy_score
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, x_test, y_test):
    try:
        y_preds = model.predict(x_test)
        y_preds = tf.round(y_preds)
        accuracy = accuracy_score(y_test, y_preds)
        logging.info(f"Model evaluated successfully with accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise
