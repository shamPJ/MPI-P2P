from mpi4py import MPI
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import copy

def generate_symmetric_er(size, p, seed=42):
    np.random.seed(seed)
    adj = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i + 1, size):
            if np.random.rand() < p:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def make_blobs(n_samples=100, dim=2, K=3, seed=0):
    np.random.seed(seed)
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    X = []
    for c in centers:
        X.append(np.random.randn(n_samples // K, dim) + c)
    return np.vstack(X)


def serialize_gmm(gmm):
    return {
        "means": gmm.means_,
        "covs": gmm.covariances_,
        "weights": gmm.weights_,
    }


def deserialize_gmm(params):
    gmm = GaussianMixture(
        n_components=len(params["weights"]),
        covariance_type="diag"
    )
    gmm.means_ = params["means"]
    gmm.covariances_ = params["covs"]
    gmm.weights_ = params["weights"]
    gmm.precisions_cholesky_ = 1.0 / np.sqrt(params["covs"])
    return gmm


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    K = 3
    NUM_FED_ITER = 20
    m_neib = 20
    p = 0.5

    # Graph
    if rank == 0:
        adj = generate_symmetric_er(size, p)
    else:
        adj = None
    adj = comm.bcast(adj, root=0)

    neighbors = [i for i in range(size) if adj[rank, i] == 1]
    
    # Print graph
    if rank == 0:
        print("\n=== Graph ===")
        for i in range(size):
            nbrs = [j for j in range(size) if adj[i, j] == 1]
            print(f"R{i} -> {nbrs}")
        print("=============\n")

    # Local dataset
    X_local = make_blobs(K=K, seed=rank)
    # Gather local datasets at root 
    X_all = comm.gather(X_local, root=0)

    # Initialize local GMM
    gmm_fed = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        random_state=rank
    ).fit(X_local)

    for it in range(NUM_FED_ITER):

        # --- Send GMM params ---
        params = serialize_gmm(gmm_fed)
        send_reqs = [comm.isend(params, dest=nbr, tag=it) for nbr in neighbors]

        # --- Receive neighbor params ---
        recv_reqs = [comm.irecv(source=nbr, tag=it) for nbr in neighbors]
        neighbor_params = [req.wait() for req in recv_reqs]

        # Ensure sends complete
        MPI.Request.Waitall(send_reqs)

        # --- Build augmented dataset ---
        augmented = [X_local]

        for p_neib in neighbor_params:
            gmm_neib = deserialize_gmm(p_neib)
            X_samp, _ = gmm_neib.sample(m_neib)
            augmented.append(X_samp)

        X_aug = np.vstack(augmented)

        # --- Refit local model ---
        gmm_fed = GaussianMixture(
            n_components=K,
            covariance_type="diag",
            random_state=rank
        ).fit(X_aug)
        
        # Print cluster means
        means = np.round(gmm_fed.means_, 2)

        #for r in range(size):
        #    comm.Barrier()
        #    if rank == r:
        #        print(f"[R{rank} | it{it}] recv={len(neighbor_params)}, augN={len(X_aug)}")
        #        print(f"[R{rank} | it{it}] means:\n{means}\n")
   
    final_means = np.round(gmm_fed.means_, 2)
    for r in range(size):
        comm.Barrier()
        if rank == r:
            print(f"[R{rank}] FINAL means:\n{final_means}\n")

    # -----------------------------
    # Centralized baseline (only rank 0)
    # -----------------------------
    if rank == 0:
        X_global = np.vstack(X_all)

        gmm_central = GaussianMixture(
            n_components=K,
            covariance_type="diag",
            random_state=0
        ).fit(X_global)
        
        central_mean = gmm_central.means_
        print("\n=== CENTRALIZED GMM ===")
        print(np.round(central_mean, 2))
        print("=======================\n")
    else:
        central_mean = None

    # Broadcast centralized mean to all ranks
    central_mean = comm.bcast(central_mean, root=0)

    # -----------------------------
    # Compute distance using Hungarian matching
    # -----------------------------
    # central_mean_components: (K, dim)
    # fed_mean_components: (K, dim)

    # Compute cost matrix: squared Euclidean distances between components
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = np.sum((gmm_fed.means_[i] - central_mean[j]) ** 2)

    # Solve assignment problem (min total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute total squared distance using optimal matching
    sq_dist = cost_matrix[row_ind, col_ind].sum()

    # Print results (ordered by assignment)
    for r in range(size):
        comm.Barrier()
        if rank == r:
            print(f"[R{rank}] Hungarian matched squared distance: {sq_dist:.4f}")
        
if __name__ == "__main__":
    main()
