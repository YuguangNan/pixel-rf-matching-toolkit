import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from keras.models import load_model

# ============================================================
# Import utilities (REPLACING only function references)
# ============================================================
from src.utils.sparam_utils import (
    db20,
    s_to_abcd_2port,
    match_metrics_from_s,
    parse_cti,
    flat_to_complex
)

from src.utils.model_utils import (
    load_forward_model,
    model_predict_sparams
)

from src.utils.ads_utils import (
    run_ads_simulation
)

from src.utils.layout_utils import (
    save_proj_a,          # unified save function
    connectivity_metrics  # your BFS / floating island logic
)

from src.utils.ga_utils import (
    tournament_select,
    crossover,
    mutate
)


# ============================================================
# ORIGINAL CONSTANTS (unchanged)
# ============================================================
MODEL_PATH      = r"dnn_model_final.h5"
EM_SETUP_DIR    = r"D:\emSimulation\emSetup_MoM"
PROJ_A_PATH     = os.path.join(EM_SETUP_DIR, "proj_a")
CTI_PATH        = os.path.join(EM_SETUP_DIR, "proj.cti")

GRID_SIZE = 15

FREQS = np.array([
    1e8, 5.304347826e8, 9.608695652e8, 1.391304348e9, 1.82173913e9,
    2.252173913e9, 2.4e9, 2.682608696e9, 3.113043478e9, 3.543478261e9,
    3.973913043e9, 4.404347826e9, 4.834782609e9, 5.265217391e9,
    5.695652174e9, 6.126086956e9, 6.556521739e9, 6.986956522e9,
    7.417391304e9, 7.847826087e9, 8.278260869e9, 8.708695652e9,
    9.139130435e9, 9.569565217e9, 1e10
])
TARGET_FREQ_IDX = 14

# GA hyperparameters (unchanged)
POP_SIZE      = 50
N_GENERATIONS = 40
CROSS_RATE    = 0.7
MUT_RATE      = 0.02
ELITE_K       = 2

# fitness constants (unchanged)
DISCONNECTED_PENALTY = 1e6
MIN_FILL = 0.20
MAX_FILL = 0.60
PATH_MIN = 10
FLOATING_WEIGHT    = 8.0
MAIN_FRAC_WEIGHT   = 2.0
PATH_WEIGHT        = 2.0
TARGET_REFL_DB     = -10.0
TARGET_S21_MIN_DB  = -1.0

Z_SOURCE = 50 + 0j
Z_LOAD   = 60 + 10j


# ============================================================
# GA layout encoding (unchanged)
# ============================================================
N_VAR = GRID_SIZE * GRID_SIZE

def genome_to_layout(genome):
    return genome.reshape(GRID_SIZE, GRID_SIZE)

def layout_to_genome(layout):
    return layout.reshape(-1)


# ============================================================
# GA fitness function (full original logic preserved)
# ============================================================
def fitness(genome, model):
    layout = genome_to_layout(genome)
    fill = layout.sum() / (GRID_SIZE * GRID_SIZE)

    # hard limits
    if fill < 0.05 or fill > 0.9:
        return DISCONNECTED_PENALTY

    # connectivity
    start = (GRID_SIZE // 2, 0)
    end   = (GRID_SIZE // 2, GRID_SIZE - 1)

    connected, sp_len, main_cond, floating = connectivity_metrics(layout, start, end)

    if not connected:
        return DISCONNECTED_PENALTY + floating * 10 + abs(fill - 0.4) * 50

    fitness_val = 0.0

    # floating penalty
    fitness_val += FLOATING_WEIGHT * floating

    # main conductor ratio
    total = layout.sum()
    main_frac = main_cond / (total + 1e-6)
    fitness_val += MAIN_FRAC_WEIGHT * (1.0 - main_frac)

    # path too short
    if sp_len > 0 and sp_len < PATH_MIN:
        fitness_val += PATH_WEIGHT * (PATH_MIN - sp_len)

    # fill soft constraint
    if fill < MIN_FILL:
        fitness_val += (MIN_FILL - fill) * 50
    elif fill > MAX_FILL:
        fitness_val += (fill - MAX_FILL) * 50

    fitness_val += abs(fill - 0.4) * 5.0

    # predict S-parameters
    S11, S21, S22 = model_predict_sparams(model, layout)
    S11_f = S11[TARGET_FREQ_IDX]
    S21_f = S21[TARGET_FREQ_IDX]
    S22_f = S22[TARGET_FREQ_IDX]

    # total reflection Γ
    Gamma_S, Gamma_L = match_metrics_from_s(S11_f, S21_f, S22_f,
                                            Z_SOURCE, Z_LOAD, Z0=50.0)
    if Gamma_S is None:
        return DISCONNECTED_PENALTY + 1000

    refl_S_db = db20(Gamma_S)
    refl_L_db = db20(Gamma_L)

    fitness_val += max(0.0, refl_S_db - TARGET_REFL_DB)
    fitness_val += max(0.0, refl_L_db - TARGET_REFL_DB)

    s21_db = db20(S21_f)
    fitness_val += max(0.0, TARGET_S21_MIN_DB - s21_db)

    return fitness_val


# ============================================================
# GA main loop (original logic preserved)
# ============================================================
def init_population():
    return np.random.randint(0, 2, size=(POP_SIZE, N_VAR), dtype=np.int8)


def run_ga(model):
    pop = init_population()
    best_genome = None
    best_fitness = float("inf")

    best_history = []
    layout_snapshots = []

    snapshot_gens = sorted(set([
        0,
        max(0, N_GENERATIONS // 10),
        max(0, N_GENERATIONS // 4),
        max(0, N_GENERATIONS // 2),
        max(0, (3 * N_GENERATIONS) // 4),
        N_GENERATIONS - 1
    ]))

    for gen in range(N_GENERATIONS):
        fitnesses = np.array([fitness(ind, model) for ind in pop])
        gen_best_idx = np.argmin(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_genome = pop[gen_best_idx].copy()

        best_history.append(gen_best_fit)

        if gen in snapshot_gens:
            layout_snapshots.append((gen, genome_to_layout(pop[gen_best_idx])))

        print(f"[Gen {gen:03d}] best={gen_best_fit:.4f}  global_best={best_fitness:.4f}")

        elite_idx = fitnesses.argsort()[:ELITE_K]
        elites = pop[elite_idx].copy()

        new_pop = []
        while len(new_pop) < POP_SIZE - ELITE_K:
            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE - ELITE_K:
                new_pop.append(mutate(c2))

        pop = np.vstack([elites, np.array(new_pop, dtype=np.int8)])

    return best_genome, best_fitness, best_history, layout_snapshots


# ============================================================
# MAIN PROCESS
# ============================================================
def main():
    print(f"▶ Matching target: Zs = {Z_SOURCE}  ZL = {Z_LOAD}")
    model = load_forward_model(MODEL_PATH)

    print("\n===== GA START =====")
    best_genome, best_fit, best_history, layout_snaps = run_ga(model)
    print("\n🎯 GA completed, best fitness =", best_fit)

    best_layout = genome_to_layout(best_genome)

    # DNN predicted S-parameters
    S11_f_all, S21_f_all, S22_f_all = model_predict_sparams(model, best_layout)
    idx = TARGET_FREQ_IDX

    S11_f = S11_f_all[idx]
    S21_f = S21_f_all[idx]
    S22_f = S22_f_all[idx]

    Gamma_S_f, Gamma_L_f = match_metrics_from_s(
        S11_f, S21_f, S22_f,
        Z_SOURCE, Z_LOAD, Z0=50.0
    )

    # Save proj_a
    start = (GRID_SIZE // 2, 0)
    end   = (GRID_SIZE // 2, GRID_SIZE - 1)
    save_proj_a(best_layout, start, end, PROJ_A_PATH)

    # Run ADS
    run_ads_simulation()

    freqs_cti, S11_ads, S21_ads, S22_ads = parse_cti(CTI_PATH)

    idx_cti = np.argmin(np.abs(freqs_cti - FREQS[idx]))

    S11_a = S11_ads[idx_cti]
    S21_a = S21_ads[idx_cti]
    S22_a = S22_ads[idx_cti]

    Gamma_S_a, Gamma_L_a = match_metrics_from_s(
        S11_a, S21_a, S22_a,
        Z_SOURCE, Z_LOAD, Z0=50.0
    )

    # ============================================================
    # PLOTS (unchanged)
    # ============================================================
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(best_history)), best_history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Best Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # evolution snapshots
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.flatten()
    for ax, (gen, layout) in zip(axes, layout_snaps):
        ax.imshow(layout, cmap="gray_r", origin="lower")
        ax.set_title(f"Gen {gen}")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(layout_snaps), rows*cols):
        axes[k].axis("off")
    plt.suptitle("GA Layout Evolution", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # S-parameters comparison
    plt.figure()
    plt.plot(freqs_cti/1e9, db20(S11_ads), label="ADS |S11|")
    plt.plot(FREQS/1e9,    db20(S11_f_all), "--", label="DNN |S11|")
    plt.xlabel("GHz"); plt.ylabel("dB")
    plt.grid(); plt.legend()
    plt.title("S11: DNN vs ADS")
    plt.tight_layout()
    plt.savefig("S11.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(freqs_cti/1e9, db20(S21_ads), label="ADS |S21|")
    plt.plot(FREQS/1e9,    db20(S21_f_all), "--", label="DNN |S21|")
    plt.xlabel("GHz"); plt.ylabel("dB")
    plt.grid(); plt.legend()
    plt.title("S21: DNN vs ADS")
    plt.tight_layout()
    plt.savefig("S21.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(freqs_cti/1e9, db20(S22_ads), label="ADS |S22|")
    plt.plot(FREQS/1e9,    db20(S22_f_all), "--", label="DNN |S22|")
    plt.xlabel("GHz"); plt.ylabel("dB")
    plt.grid(); plt.legend()
    plt.title("S22: DNN vs ADS")
    plt.tight_layout()
    plt.savefig("S22.png", dpi=200)
    plt.show()

    print("\n🎉 Inverse design finished successfully!")


if __name__ == "__main__":
    main()
