"""
mpiexec -n 2 python share_data_mpi.py
"""
from mpi4py import MPI
import numpy as np
from dummy_data import dummy_halo_properties
from data_distribution_helpers import compute_npts_to_send_and_receive

QC_LBOX = 1000.


if __name__ == "__main__":
    """
    """
    #  Fire up a communicator with one rank per compute node
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    rng = np.random.RandomState(rank)

    #  Generate some dummy point data spanning the simulation box
    #  In this setup, each rank gets its own collection of distinct points that span the box
    #  The end goal will be for each rank to end up with only points in its buffered subvolume
    new_data, new_metadata = dummy_halo_properties(
            rng.randint(100, 200), nranks, QC_LBOX)

    #  Calculate two dictionaries that store the information
    #  about how the points should be distributed amongst the nranks compute nodes
    npts_to_send_to_ranks, npts_to_receive_from_ranks = compute_npts_to_send_and_receive(
        comm, rank, nranks, new_data['rank'])

    #  Loop over each rank that will receive points from this rank
    for receiving_rank, npts_to_send in npts_to_send_to_ranks.items():
        mask = new_data['rank'] == receiving_rank

        #  We will send all the halo properties to each receiving rank, one at a time
        for colname, dtype in new_metadata['dtype_dict'].items():
            arr_to_send = new_data[colname][mask].astype(dtype)
            comm.Send(arr_to_send, dest=receiving_rank)

        #  Now loop over all ranks that send data to this rank
        all_received_data = dict()
        for sending_rank, npts_to_receive in npts_to_receive_from_ranks.items():

            received_data_from_sending_rank = dict()
            #  We will receive all the halo properties from each sending rank
            for colname, dtype in new_metadata['dtype_dict'].items():
                arr = np.empty(npts_to_receive, dtype=dtype)
                comm.Recv(arr, source=sending_rank)
                received_data_from_sending_rank[colname] = arr

            #  Bundle the received data into a dictionary
            all_received_data[sending_rank] = received_data_from_sending_rank

    #  Compute the total number of points received by this rank
    ntot_received = sum((len(d['x']) for d in all_received_data.values()))
    print(rank, ntot_received)
