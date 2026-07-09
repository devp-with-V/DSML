import numpy as np
from shared.registry import register_method

@register_method("cnn_classifier")
def run_cnn(X, y=None):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError:
        raise ImportError("TensorFlow is required for CNN. Run `pip install tensorflow`")
        
    if y is None:
        raise ValueError("Target y is required")
        
    # Assume X is already a reshaped image array: (N, Height, Width, Channels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    preds = np.argmax(model.predict(X_test), axis=-1)
    metrics = {"accuracy": float(accuracy_score(y_test, preds))}
    
    return {"model": model, "metrics": metrics, "importances": None, "y_test": locals().get("y_test"), "preds": locals().get("preds")}
