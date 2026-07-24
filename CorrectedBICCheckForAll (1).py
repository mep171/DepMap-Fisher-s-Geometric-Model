import os
import glob
import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)  # <--- ADD THIS LINE FIRST
import jax.numpy as jnp
from jax import random
import jaxopt

# --- Configuration ---
MAX_REGRESS_ITER = 50000
NUM_SEEDS = 200            # Number of random restarts per dimension D
BASE_SEED = 102
L2_LAMBDA = 1e-4           # Consistent L2 regularization weight
D_GUESS_VALUES = [1, 2, 3, 4, 5]
GUESS_X = 1.0

NORM_MATRIX_PATH = "depmap_empirical_sd_norm.csv"
TARGET_FOLDER = "OnlyCancerGenesFixed"
NORM_FLOOR = 1e-3

# --- Helper Functions ---

def gauge_fix_Fixed(Z, P, d):
    """Fixes translational, rotational, and reflectional gauge freedoms."""
    Pmean = P.mean(axis=0)
    Pshifted = P - Pmean
    Zshifted = Z + Pmean  

    M = Pshifted[:d, :d].T
    Q, R = np.linalg.qr(M)
    
    Protated = Pshifted @ Q
    Zrotated = Zshifted @ Q
    
    signs = np.sign(np.diag(Protated))
    S = np.diag(signs)
    return Zrotated @ S, Protated @ S


def calculate_r_squared(observed, predicted):
    """Standard unweighted R-squared score."""
    ss_res = jnp.sum(jnp.square(observed - predicted))
    ss_tot = jnp.sum(jnp.square(observed - jnp.mean(observed)))
    return float(1.0 - (ss_res / ss_tot))


def calculate_gauge_adjusted_bic(rss, n_samples, total_params, d):
    """Calculates BIC adjusting for gauge degrees of freedom."""
    gauge_freedoms = (d * (d + 1)) // 2
    k_eff = total_params - gauge_freedoms
    bic = n_samples * np.log(rss / n_samples) + k_eff * np.log(n_samples)
    return float(bic), k_eff


def align_empirical_norm(norm_csv_path, input_df):
    """Aligns noise scaling matrix with target expression matrix."""
    print("Aligning empirical LFC noise scales (norm)...")
    norm_wide = pd.read_csv(norm_csv_path, index_col=0)

    target_genes = [str(g).strip() for g in input_df.columns]
    target_cells = [str(c).strip() for c in input_df.index]

    valid_vals = norm_wide.to_numpy().flatten()
    valid_vals = valid_vals[~np.isnan(valid_vals)]
    global_median_sd = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.1

    aligned = norm_wide.reindex(index=target_genes, columns=target_cells).fillna(global_median_sd)
    aligned_cell_by_gene = aligned.T
    aligned_jax = jnp.clip(jnp.array(aligned_cell_by_gene.to_numpy(), dtype=np.float64), a_min=NORM_FLOOR)

    return aligned_cell_by_gene, aligned_jax

# --- Classes ---

class Landscape:
    def __init__(self, C, D, M):
        self.C, self.D, self.M = C, D, M

    def generate_initial_guess(self, key):
        key, kZ, kP = random.split(key, 3)
        Z = random.normal(kZ, (self.C, self.D))
        P = random.normal(kP, (self.M, self.D))
        X = GUESS_X
        return key, Z, P, X

    def calculate_fitness(self, Z, P, X):
        combined = Z[:, None, :] + P[None, :, :]
        dist_sq = jnp.sum(jnp.square(combined), axis=2)
        return jnp.log(jnp.abs(X) + 1e-10) - (dist_sq / 2.0)


class OptimizationProblem:
    def __init__(self, landscape_obj, observed, norm):
        self.ls, self.obs, self.norm = landscape_obj, observed, norm
        self.C, self.D, self.M = landscape_obj.C, landscape_obj.D, landscape_obj.M

    def get_parameter_vector(self, Z, P, X):
        return jnp.concatenate([jnp.array([X]), jnp.ravel(Z), jnp.ravel(P)])

    def reconstruct_ZP(self, p_vec):
        X = p_vec[0]
        z_end = 1 + self.C * self.D
        Z = p_vec[1:z_end].reshape((self.C, self.D))
        P = p_vec[z_end:].reshape((self.M, self.D))
        return Z, P, X

    def loss_function(self, parameter_vector, l2_lambda=L2_LAMBDA):
        Z, P, X = self.reconstruct_ZP(parameter_vector)
        pred = self.ls.calculate_fitness(Z, P, X)
        data_loss = jnp.mean(jnp.square((self.obs - pred) / self.norm))
        reg_loss = l2_lambda * (jnp.sum(jnp.square(Z)) + jnp.sum(jnp.square(P)))
        return data_loss + reg_loss

# --- Optimization Engine ---

def run_inference_for_d(d_val, observed, norm, num_seeds, base_key):
    C, M = observed.shape
    n_samples = C * M
    
    ls_obj = Landscape(C=C, D=d_val, M=M)
    prob_obj = OptimizationProblem(ls_obj, observed, norm)
    solver = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=prob_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    
    best_loss = float('inf')
    best_params = None
    best_seed_idx = -1
    current_key = base_key

    print(f"\n>>> Running multi-seed optimization for D={d_val} ({num_seeds} restarts)...")
    for i in range(num_seeds):
        current_key, Z_init, P_init, X_init = ls_obj.generate_initial_guess(current_key)
        init_pv = prob_obj.get_parameter_vector(Z_init, P_init, X_init)
        
        try:
            res = solver.run(init_pv, l2_lambda=L2_LAMBDA)
            loss_val = float(res.state.fun_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = res.params
                best_seed_idx = i
        except Exception:
            continue

    if best_params is None:
        raise RuntimeError(f"All optimization attempts failed for D={d_val}")

    rZ, rP, rX = prob_obj.reconstruct_ZP(best_params)
    Z_fixed, P_fixed = gauge_fix_Fixed(np.array(rZ), np.array(rP), d=d_val)
    
    pred_fit = ls_obj.calculate_fitness(jnp.array(Z_fixed), jnp.array(P_fixed), rX)
    rss = float(jnp.sum(jnp.square((observed - pred_fit) / norm)))
    r2 = calculate_r_squared(observed, pred_fit)
    
    bic, k_eff = calculate_gauge_adjusted_bic(
        rss=rss, 
        n_samples=n_samples, 
        total_params=len(best_params), 
        d=d_val
    )

    return {
        "summary": {
            "Dimension": d_val,
            "Best_Seed_Index": best_seed_idx,
            "Loss": best_loss,
            "R2_Score": r2,
            "RSS": rss,
            "BIC_Score": bic,
            "Effective_Params": k_eff,
            "Num_Cell_Lines": C,
            "Num_Genes": M
        },
        "Z_fixed": Z_fixed,
        "P_fixed": P_fixed,
        "X": float(rX),
        "predicted_fitness": np.array(pred_fit)
    }


def process_single_file(file_path):
    print(f"\n==================================================")
    print(f"Processing File: {file_path}")
    print(f"==================================================")
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        
        # --- Robust Formatting & Data Cleaning ---
        # 1. Force convert non-numeric/string values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 2. Drop rows (cell lines) or cols (genes) that are completely empty
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # 3. Fill isolated NaNs with median/0.0
        df = df.fillna(0.0)
        
        if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            print(f"Skipping {file_path}: File is empty after data cleaning.")
            return

        cell_line_names = list(df.index)
        gene_names = list(df.columns)
        
        # 4. Explicit float64 NumPy array for JAX compatibility
        np_vals = df.to_numpy(dtype=np.float64)
        observed = jnp.array(np_vals)
        C, M = observed.shape
        print(f"Data matrix loaded with shape ({C}, {M}) -> ({C} Cell Lines, {M} Genes)")

    except Exception as e:
        print(f"Skipping {file_path}: Failed to load/clean dataset. Error: {e}")
        return

    _, norm = align_empirical_norm(NORM_MATRIX_PATH, df)

    base_key = random.PRNGKey(BASE_SEED)
    results_list = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for d_val in D_GUESS_VALUES:
        try:
            d_res = run_inference_for_d(d_val, observed, norm, NUM_SEEDS, base_key)
            results_list.append(d_res["summary"])
            
            dim_cols = [f"Dim_{k+1}" for k in range(d_val)]
            
            z_df = pd.DataFrame(d_res["Z_fixed"], index=cell_line_names, columns=dim_cols)
            z_df.to_csv(f"{base_name}_D{d_val}_CellLine_Positions_Z.csv")
            
            p_df = pd.DataFrame(d_res["P_fixed"], index=gene_names, columns=dim_cols)
            p_df.to_csv(f"{base_name}_D{d_val}_Gene_Positions_P.csv")
            
            pred_df = pd.DataFrame(d_res["predicted_fitness"], index=cell_line_names, columns=gene_names)
            pred_df.to_csv(f"{base_name}_D{d_val}_Predicted_Fitness.csv")
            
            with open(f"{base_name}_D{d_val}_Scaling_X.txt", "w") as f:
                f.write(f"X: {d_res['X']}\n")

            print(f"  Result D={d_val} | R2: {d_res['summary']['R2_Score']:.4f} | BIC: {d_res['summary']['BIC_Score']:.2f} | Matrices Saved!")
            
        except Exception as e:
            print(f"  Optimization failed for D={d_val}: {e}")

    output_csv = f"{base_name}_BIC_Dimensionality_Results.csv"
    pd.DataFrame(results_list).to_csv(output_csv, index=False)
    print(f"\nSaved overall summary to: {output_csv}")


def main():
    if not os.path.exists(TARGET_FOLDER):
        print(f"Directory '{TARGET_FOLDER}' not found.")
        return

    file_list = glob.glob(os.path.join(TARGET_FOLDER, "*.csv"))
    if not file_list:
        print(f"No CSV files found in '{TARGET_FOLDER}'.")
        return

    for file_path in file_list:
        process_single_file(file_path)


if __name__ == "__main__":
    main()