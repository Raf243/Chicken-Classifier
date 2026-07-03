from utils import forward_prop, np

def predict(X, parameters):
    AL, _ = forward_prop(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions

def accuracy(predictions, Y):
    return np.mean(predictions == Y) * 100