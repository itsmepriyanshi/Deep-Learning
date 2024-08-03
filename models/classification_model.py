import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_model(X_train, y_train):
    model = create_model()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, verbose=0)
    return model, history

def find_best_learning_rate(model, X_train, y_train):
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch/3)
    )
    history = model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[lr_scheduler])
    lrs = 1e-5 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss")
    plt.show()

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
