from mpi4py import MPI
import numpy as np

def generate_symmetric_er(size, p, seed=42):
    """
    Generate a symmetric adjacency matrix for an ER graph.
    adj[i,j] = 1 means i and j are connected.
    """
    np.random.seed(seed)
    adj = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(i + 1, size):
            if np.random.rand() < p:
                adj[i, j] = 1
                adj[j, i] = 1  # ensure symmetry
    return adj

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    p = 0.4  # connection probability

    # Generate symmetric adjacency matrix at root
    if rank == 0:
        adj = generate_symmetric_er(size, p)
    else:
        adj = None

    # Broadcast adjacency matrix to all ranks
    adj = comm.bcast(adj, root=0)

    # Determine neighbors for this rank
    neighbors = [i for i in range(size) if adj[rank, i] == 1]
    print(f"Rank {rank} neighbors: {neighbors}")

    # Example P2P exchange (send local value to neighbors)
    local_value = rank * 10
    
    # Send values to neighbors (non-blocking)
    send_requests = [comm.isend(local_value, dest=nbr) for nbr in neighbors]

    # Post non-blocking receives from neighbors
    recv_requests = [comm.irecv(source=nbr) for nbr in neighbors]

    # Wait for all receives
    received = [(nbr, req.wait()) for nbr, req in zip(neighbors, recv_requests)]

    # Wait for all sends
    MPI.Request.Waitall(send_requests)
    

    print(f"Rank {rank} received: {received}")

if __name__ == "__main__":
    main()
