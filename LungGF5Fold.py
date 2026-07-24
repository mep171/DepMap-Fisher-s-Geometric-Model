import os
import numpy as np
import pandas as pd
import jax

# --- Enable 64-bit precision ---
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
import jaxopt

# --- Configuration ---
MAX_REGRESS_ITER = 50000
L2_LAMBDA = 1e-5
D_FIXED = 2
K_FOLDS = 5
BEST_SEED = 215
GUESS_X = 1.0

# Empirical Noise Normalization Config
NORM_MATRIX_PATH = "depmap_empirical_sd_norm.csv"
NORM_FLOOR = 1e-3
DATA_FILE_PATH = "OnlyCancerGenesFixed/Lung_CancerGenesFixed.csv"


# --- Helper Functions ---

def gauge_fix_Fixed(Z, P, d):
    """
    Fixes translational, rotational, and reflectional gauge degrees of freedom.
    """
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


def align_empirical_norm(norm_csv_path, input_df):
    """Aligns noise scaling matrix with target expression matrix."""
    print("Aligning empirical LFC noise scales (norm)...")
    if not os.path.exists(norm_csv_path):
        print(f"Warning: '{norm_csv_path}' not found. Falling back to empirical std dev of observed data.")
        return None, None

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


def get_empirical_initial_params(obs_matrix, target_d, key, mask):
    # Use only training data to calculate means (ignore test points)
    train_data = jnp.where(mask == 1.0, obs_matrix, jnp.nan)
    
    # Calculate means while ignoring NaNs (masked values)
    z_means = -jnp.nanmean(train_data, axis=1, keepdims=True)
    p_means = -jnp.nanmean(train_data, axis=0, keepdims=True)
    
    z_means = jnp.nan_to_num(z_means)
    p_means = jnp.nan_to_num(p_means)
    
    # Tile the means across target dimensions
    Z_init = jnp.tile(z_means, (1, target_d))
    P_init = jnp.tile(p_means.T, (1, target_d))
    
    # Add small jitter to break symmetry
    k1, k2 = random.split(key)
    Z_init += random.normal(k1, Z_init.shape) * 0.05
    P_init += random.normal(k2, P_init.shape) * 0.05
    
    return Z_init, P_init


# --- Problem Setup Classes ---

class Landscape:
    def __init__(self, C, D, M):
        self.C, self.D, self.M = C, D, M

    def generate_initial_guess(self, key):
        key, kz, kp = random.split(key, 3)
        Z = random.normal(kz, (self.C, self.D))
        P = random.normal(kp, (self.M, self.D))
        return key, Z, P

    def calculate_fitness(self, Z, P, X):
        combined = Z[:, None, :] + P[None, :, :]
        dist_sq = jnp.sum(jnp.square(combined), axis=2)
        return jnp.log(jnp.abs(X) + 1e-10) - (dist_sq / 2.0)


class CVRegressionProblem:
    def __init__(self, landscape_obj, observed_fitness, norm_matrix, D_guess, mask):
        self.ls = landscape_obj
        self.obs = observed_fitness
        self.norm = norm_matrix
        self.D = D_guess 
        self.mask = mask 

    def get_parameter_vector(self, Z, P, X):
        return jnp.concatenate([jnp.array([X]), jnp.ravel(Z), jnp.ravel(P)])

    def reconstruct_ZP(self, p_vec):
        z_size = self.ls.C * self.D
        p_size = self.ls.M * self.D
        X = p_vec[0]
        Z = p_vec[1 : 1 + z_size].reshape((self.ls.C, self.D))
        P = p_vec[1 + z_size : 1 + z_size + p_size].reshape((self.ls.M, self.D))
        return Z, P, X

    def loss_function(self, p_vec):
        Z, P, X = self.reconstruct_ZP(p_vec)
        pred = self.ls.calculate_fitness(Z, P, X)
        
        # Weighted squared loss by sample-specific empirical norm matrix
        sq_err = jnp.square((self.obs - pred) / self.norm)
        
        data_loss = jnp.sum(sq_err * self.mask) / jnp.sum(self.mask)
        reg_loss = L2_LAMBDA * (jnp.sum(jnp.square(Z)) + jnp.sum(jnp.square(P)))
        
        return data_loss + reg_loss


def calculate_masked_pcc(obs, pred, mask):
    """Calculates Pearson Correlation Coefficient for masked elements only."""
    idx = jnp.where(mask > 0)
    y_obs = obs[idx]
    y_pred = pred[idx]
    
    mean_obs = jnp.mean(y_obs)
    mean_pred = jnp.mean(y_pred)
    
    numerator = jnp.sum((y_obs - mean_obs) * (y_pred - mean_pred))
    denominator = jnp.sqrt(jnp.sum(jnp.square(y_obs - mean_obs)) * jnp.sum(jnp.square(y_pred - mean_pred)))
    
    return float(numerator / (denominator + 1e-10))


def calculate_masked_r_squared(obs, pred, mask):
    """Calculates R-squared (coefficient of determination) for masked elements only."""
    idx = jnp.where(mask > 0)
    y_obs = obs[idx]
    y_pred = pred[idx]
    
    ss_res = jnp.sum(jnp.square(y_obs - y_pred))
    ss_tot = jnp.sum(jnp.square(y_obs - jnp.mean(y_obs)))
    
    return float(1.0 - (ss_res / (ss_tot + 1e-10)))


# --- Main Engine ---

def main():
    # 1. Load Data with robust cleaning
    try:
        df = pd.read_csv(DATA_FILE_PATH, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1).fillna(0.0)
        
        cell_line_names = list(df.index)
        gene_names = list(df.columns)
        
        observed = jnp.array(df.to_numpy(dtype=np.float64))
    except FileNotFoundError:
        print(f"Error: '{DATA_FILE_PATH}' not found.")
        return

    C, M = observed.shape
    total_elements = C * M

    # 2. Align Empirical Norm Matrix
    _, norm_jax = align_empirical_norm(NORM_MATRIX_PATH, df)
    
    # Fallback to scalar std dev if norm matrix alignment isn't available
    if norm_jax is None:
        std_norm = float(jnp.std(observed) + 1e-6)
        norm_jax = jnp.full_like(observed, std_norm)

    # 3. Manual Shuffle and Fold Generation
    key = random.PRNGKey(BEST_SEED)
    indices = jnp.arange(total_elements)
    shuffled_indices = random.permutation(key, indices)
    fold_indices = jnp.array_split(shuffled_indices, K_FOLDS)
    
    results_list = []
    print(f"Starting {K_FOLDS}-Fold CV with Empirical Weights (D={D_FIXED})...")

    dim_cols = [f"Dim_{k+1}" for k in range(D_FIXED)]

    for i in range(K_FOLDS):
        test_idx = fold_indices[i]
        
        # Create masks
        train_mask_flat = jnp.ones(total_elements).at[test_idx].set(0.0)
        test_mask_flat = jnp.zeros(total_elements).at[test_idx].set(1.0)
        train_mask = train_mask_flat.reshape((C, M))
        test_mask = test_mask_flat.reshape((C, M))

        # Empirical Initialization
        fold_key = random.fold_in(key, i)
        init_Z, init_P = get_empirical_initial_params(observed, D_FIXED, fold_key, train_mask)
        
        ls_obj = Landscape(C=C, D=D_FIXED, M=M)
        prob = CVRegressionProblem(ls_obj, observed, norm_jax, D_FIXED, train_mask)
        init_pv = prob.get_parameter_vector(init_Z, init_P, GUESS_X)
        
        solver = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=prob.loss_function, maxiter=MAX_REGRESS_ITER)
        
        try:
            res = solver.run(init_pv)
            rZ, rP, rX = prob.reconstruct_ZP(res.params)
            pred_fit = ls_obj.calculate_fitness(rZ, rP, rX)
            
            # Gauge Fixing
            Z_np, P_np = np.array(rZ), np.array(rP)
            Z_fixed, P_fixed = gauge_fix_Fixed(Z_np, P_np, d=D_FIXED)
            
            # Save labeled output dataframes
            pd.DataFrame(Z_fixed, index=cell_line_names, columns=dim_cols).to_csv(f"Fold{i+1}_Z_Fixed.csv")
            pd.DataFrame(P_fixed, index=gene_names, columns=dim_cols).to_csv(f"Fold{i+1}_P_Fixed.csv")
            pd.DataFrame({"X": [float(rX)]}).to_csv(f"Fold{i+1}_X_Val.csv", index=False)
            
            # Metrics Calculation
            err_sq = jnp.square(observed - pred_fit)
            train_mse = float(jnp.sum(err_sq * train_mask) / jnp.sum(train_mask))
            test_mse = float(jnp.sum(err_sq * test_mask) / jnp.sum(test_mask))
            
            # Calculate Test PCC and Test R2
            test_pcc = calculate_masked_pcc(observed, pred_fit, test_mask)
            test_r2 = calculate_masked_r_squared(observed, pred_fit, test_mask)

            results_list.append({
                'Fold': i + 1,
                'Train_MSE': train_mse,
                'Test_MSE': test_mse,
                'Test_PCC': test_pcc,
                'Test_R2': test_r2
            })
            print(f"Fold {i+1:02d} | Test MSE: {test_mse:.5f} | Test PCC: {test_pcc:.4f} | Test R2: {test_r2:.4f}")

        except Exception as e:
            print(f"Fold {i+1} failed: {e}")

    # 4. Save and Report Results
    if results_list:
        cv_df = pd.DataFrame(results_list)
        cv_df.to_csv("Lung_CV5_D2_Results.csv", index=False)
        
        print("\n--- CV Summary ---")
        print(f"Mean Test MSE: {cv_df['Test_MSE'].mean():.6f}")
        print(f"Mean Test PCC: {cv_df['Test_PCC'].mean():.4f} (± {cv_df['Test_PCC'].std():.4f})")
        print(f"Mean Test R2:  {cv_df['Test_R2'].mean():.4f} (± {cv_df['Test_R2'].std():.4f})")
        print("Results saved to 'Lung_CV5_D2_Results.csv'.")


if __name__ == "__main__":
    main()