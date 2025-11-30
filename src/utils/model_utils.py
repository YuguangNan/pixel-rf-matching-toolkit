import numpy as np
import tensorflow as tf
from keras.models import load_model


# ============================================================
# ===================== Common Basic Utilities ================
# ============================================================

def db20(x):
    """
    Convert a complex value to 20*log10(|x|).
    """
    return 20 * np.log10(np.abs(x) + 1e-12)



# ============================================================
# =================== Model Loading Wrapper ===================
# ============================================================

def load_forward_model(model_path):
    """
    Load a Keras forward model (compile disabled).
    """
    model = load_model(model_path, compile=False)
    print("✔ Forward model loaded:", model_path)
    return model



# ============================================================
# =================== Flat Vector → Complex ===================
# ============================================================

def flat_to_complex(arr):
    """
    Convert an interleaved real/imag array into complex form.
    Example: [Re0, Im0, Re1, Im1, ...] → complex array.
    """
    real = arr[0::2]
    imag = arr[1::2]
    return real + 1j * imag



# ============================================================
# ================== Forward Model Prediction =================
# ============================================================

def model_predict(model, grid15):
    """
    Predict S11/S21/S22 using the forward model.
    Output format assumed to be 150 elements:
        S11: 50 real/imag pairs
        S21: 50 real/imag pairs
        S22: 50 real/imag pairs
    """
    x = grid15[np.newaxis, ..., np.newaxis].astype(np.float32)
    out = model.predict(x, verbose=0)[0]  # shape (150,)

    # Extract complex S-parameters
    S11 = flat_to_complex(out[0:50])
    S21 = flat_to_complex(out[50:100])
    S22 = flat_to_complex(out[100:150])

    return S11, S21, S22



# ============================================================
# =========== Generalized S-parameter extraction ==============
# ============================================================

def model_predict_sparams(model, layout):
    """
    More general version used in GA script.
    Output dimension is assumed to be:
        S11:   2*N_FREQ values
        S21:   2*N_FREQ values
        S22:   2*N_FREQ values
    Total = 6*N_FREQ
    The function internally reconstructs complex values.
    """
    x = layout.astype(np.float32)[np.newaxis, ..., np.newaxis]
    out = model.predict(x, verbose=0)[0]

    # Number of output points must be divisible by 6
    assert out.shape[0] % 6 == 0, \
        f"Unexpected model output size {out.shape[0]}, expected multiple of 6."

    N_freq = out.shape[0] // 6   # each S-parameter uses 2*N_freq (Re/Im)

    # S11 re/im pairs
    real_11 = out[0 : 2 * N_freq : 2]
    imag_11 = out[1 : 2 * N_freq : 2]
    S11 = real_11 + 1j * imag_11

    # S21
    real_21 = out[2 * N_freq : 4 * N_freq : 2]
    imag_21 = out[2 * N_freq + 1 : 4 * N_freq : 2]
    S21 = real_21 + 1j * imag_21

    # S22
    real_22 = out[4 * N_freq : 6 * N_freq : 2]
    imag_22 = out[4 * N_freq + 1 : 6 * N_freq : 2]
    S22 = real_22 + 1j * imag_22

    return S11, S21, S22
