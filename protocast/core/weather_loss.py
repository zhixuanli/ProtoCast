"""Normalized FP32 forecast error for ProtoCast."""
def forecast_squared_error(prediction, target):
    return (prediction.float() - target.float()).square()

def forecast_mse(prediction, target):
    return forecast_squared_error(prediction, target).mean()
