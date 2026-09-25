import numpy as np

def bootstrap_testset(preds, labels, iterations=200, seed=42):
    """
    Bootstrap the test set to calculate RMSE error bars.
    Expects numpy arrays for preds and labels.
    """
    rng = np.random.RandomState(seed=seed)
    idx = np.arange(labels.shape[0]) 
    
    test_rmses = []
    for i in range(iterations):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        
        # Calculate RMSE
        mse_boot = np.mean((preds[pred_idx] - labels[pred_idx]) ** 2)
        rmse_boot = np.sqrt(mse_boot)
        test_rmses.append(rmse_boot)
        
    bootstrap_mean = np.mean(test_rmses)

    ci_lower = np.percentile(test_rmses, 2.5)
    ci_upper = np.percentile(test_rmses, 97.5)
    return bootstrap_mean, ci_lower, ci_upper