import numpy as np


# ============================================================
# ====================== Basic Conversions ====================
# ============================================================

def s_to_z(S, z0=50.0):
    """
    Convert S-parameter to impedance:
        Z = Z0 * (1 + S) / (1 - S)
    Valid only for 1-port S parameters.
    """
    return z0 * (1 + S) / (1 - S + 1e-12)



# ============================================================
# ================== S-parameters <-> ABCD ====================
# ============================================================

def s_to_abcd_2port(S11, S21, S22, Z0=50.0):
    """
    Convert a 2-port S matrix (S11, S21, S22) into ABCD parameters.
    S12 is assumed equal to S21 (reciprocal network).

    If S21 is extremely small (division issues), return None.
    """
    S12 = S21

    denom = 2 * S21
    if abs(denom) < 1e-14:
        # ABCD becomes singular if S21 ~ 0
        return None

    A = ((1 + S11) * (1 - S22) + S12 * S21) / denom
    B = Z0 * ((1 + S11) * (1 + S22) - S12 * S21) / denom
    C = ((1 - S11) * (1 - S22) - S12 * S21) / (Z0 * denom)
    D = ((1 - S11) * (1 + S22) + S12 * S21) / denom

    return A, B, C, D


def abcd_to_s(A, B, C, D, Z0=50.0):
    """
    Convert ABCD parameters back to S-parameters.
    """
    denom = A + B / Z0 + C * Z0 + D
    if abs(denom) < 1e-14:
        return None, None, None

    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S21 = 2 * (A * D - B * C) / denom
    S12 = S21
    S22 = (-A + B / Z0 - C * Z0 + D) / denom
    return S11, S21, S22



# ============================================================
# ========= Matching metrics from S-parameters (GA) ===========
# ============================================================

def match_metrics_from_s(S11_f, S21_f, S22_f, Zs, Zl, Z0=50.0):
    """
    Compute source-side and load-side total reflection coefficients
    under arbitrary source/load impedances (Zs, Zl).

    Method:
        1. Convert S11/S21/S22 → ABCD.
        2. Compute input/output impedance seen from each side.
        3. Compute total reflection Gamma_S and Gamma_L.
    """
    abcd = s_to_abcd_2port(S11_f, S21_f, S22_f, Z0)
    if abcd is None:
        return None, None

    A, B, C, D = abcd

    # Port 2 terminated with Zl → find Zin at Port 1
    Zin = (A * Zl + B) / (C * Zl + D)
    Gamma_S = (Zin - Zs) / (Zin + Zs)

    # Port 1 terminated with Zs → find Zout at Port 2
    Zout = (D * Zs + B) / (C * Zs + A)
    Gamma_L = (Zout - Zl) / (Zout + Zl)

    return Gamma_S, Gamma_L
