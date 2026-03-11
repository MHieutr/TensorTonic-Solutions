import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    
    ----------
    param : np.ndarray
        Current parameters.
    grad : np.ndarray
        Gradient of the loss w.r.t param.
    m : np.ndarray
        First moment estimate.
    v : np.ndarray
        Second moment estimate.
    t : int
        Current timestep (starting from 1).
    lr : float
        Learning rate.
    beta1 : float
        Decay rate for first moment.
    beta2 : float
        Decay rate for second moment.
    eps : float
        Small constant for numerical stability.

    Return (param_new, m_new, v_new).
    """

    # Write code here
    # Ensure numpy arrays
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # 1. Update biased first moment
    m_new = beta1 * m + (1 - beta1) * grad

    # 2. Update biased second moment
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # 3. Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # 4. Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new