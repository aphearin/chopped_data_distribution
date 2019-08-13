"""
"""
import numpy as np


def compute_npts_to_send_and_receive(comm, r, n, array_of_ranks):
    """ For rank r, compute two dictionaries that determine
    how the points should be distributed amongst the n ranks.

    Parameters
    ----------
    comm : MPI communicator

    r : int
        This rank

    n : int
        Total number of ranks

    array_of_ranks : ndarray
        Integer array of shape (npts, ) storing the rank to which each point belongs

    Returns
    -------
    npts_to_send_to_ranks : dict
        Keys are the ranks to which rank r will send points.
        Values are the number of points sent to that rank by rank r.

    npts_to_receive_from_ranks : dict
        Keys are the ranks from which rank r will receive points.
        Values are the number of points that will be sent to rank r by that rank.
    """
    npts_from_rank_to_rank0_map = get_npts_from_rank_to_rank0_map(
        comm, r, n, array_of_ranks)
    npts_to_send_to_ranks = get_npts_to_send_to_ranks(
        comm, r, n, npts_from_rank_to_rank0_map)

    npts_to_rank_from_rank_dict = get_npts_to_rank_from_rank_dict(
        comm, r, npts_from_rank_to_rank0_map)
    npts_to_receive_from_ranks = get_npts_to_receive_from_ranks(
        comm, r, n, npts_to_rank_from_rank_dict)
    return npts_to_send_to_ranks, npts_to_receive_from_ranks


def get_npts_to_receive_from_ranks(comm, rank, nranks, npts_to_rank_from_rank):
    """Calculate a list of two-element tuples,
    where the first element of each tuple is the sending rank,
    and the second element is the number of points that will be sent.

    Communication strategy uses rank 0 as an intermediary.

    Parameters
    ----------
    comm : MPI communicator

    rank : int
        This rank

    nranks : int
        Total number of ranks

    npts_to_rank_from_rank : dict
        Each key stores the rank that will receive a collection of data.
        The value bound to each key is a list of two-element tuples;
        the first element of each tuple is the rank sending the data,
        the second element is the number of points that will be sent.

        The npts_to_rank_from_rank dictionary is the output
        of the get_npts_to_rank_from_rank_dict function.

    Returns
    -------
    npts_to_receive_from_ranks : dict
        Keys are the ranks to which rank r will send points.
        Values are the number of points sent to that rank by rank r.

    """
    if rank == 0:
        npts_to_receive_from_ranks = npts_to_rank_from_rank[0]
        for r in range(1, nranks):
            comm.send(npts_to_rank_from_rank[r], dest=r)
    else:
        npts_to_receive_from_ranks = comm.recv(source=0)

    npts_to_receive_from_ranks = {s[0]: s[1] for s in npts_to_receive_from_ranks}
    return npts_to_receive_from_ranks


def get_npts_to_rank_from_rank_dict(comm, rank, npts_from_rank_to_rank0_map):
    """For rank 0, calculate a dictionary where each key stores
    the rank that will receive a collection of data.
    The value bound to each key is a list of two-element tuples.
    The first element of each tuple is the "sending rank", the second element is
    the number of points that will be sent.

    For all other ranks, return an empty dictionary.

    Parameters
    ----------
    comm : MPI communicator

    rank : int
        This rank

    npts_from_rank_to_rank0_map : dict
        Each key stores the "sending rank", i.e.,
        the rank that will send a collection of data.
        The value bound to each sending rank is a dictionary,
        where key of the dictionary is the "receiving rank",
        and the value is the number of points sent by the sending rank.

        The npts_from_rank_to_rank0_map dictionary is the output
        of the get_npts_from_rank_to_rank0_map function.

    Returns
    -------
    npts_to_rank_from_rank_dict : dict
        Each key is a rank that will receive some collection of data.
        The value bound to each key is a list of two-element tuples.
        The first element of each tuple is the "sending rank", the second element is
        the number of points that will be sent.
    """
    npts_to_rank_from_rank_dict = dict()

    if rank == 0:
        for from_rank, info in npts_from_rank_to_rank0_map.items():
            for to_rank, npts_to_rank in info.items():
                try:
                    npts_to_rank_from_rank_dict[to_rank].append((from_rank, npts_to_rank))
                except KeyError:
                    npts_to_rank_from_rank_dict[to_rank] = [(from_rank, npts_to_rank)]

    return npts_to_rank_from_rank_dict


def get_npts_from_rank_to_rank0_map(comm, rank, nranks, array_of_ranks):
    """Calculate a dictionary where each key stores the "sending rank", i.e.,
    the rank that will send a collection of data.
    The value bound to each sending rank is a dictionary,
    where key of the dictionary is the "receiving rank",
    and the value is the number of points sent to rank 0 by the sending rank.

    Parameters
    ----------
    comm : MPI communicator

    rank : int
        This rank

    nranks : int
        Total number of ranks

    array_of_ranks : ndarray
        Integer array of shape (npts, ) storing the rank to which each point belongs

    Returns
    -------
    Unfinished


    npts_from_rank_to_rank0_map : dict
        Dictionary storing the number of points sent to rank 0 by each rank.
        Each key is a sending rank; the value bound to each key is a dictionary,
        where key is the "receiving rank",
    and the value is the number of points sent to rank 0 by the sending rank.

    """
    npts_from_rank_to_rank0_map = dict()

    npts_to_send_each_rank = count_npts_to_send_each_rank(rank, array_of_ranks)

    if rank == 0:
        npts_from_rank_to_rank0_map[0] = npts_to_send_each_rank
        for from_rank in range(1, nranks):
            npts_from_rank_to_rank0_map[from_rank] = comm.recv(source=from_rank)
    else:
        comm.send(npts_to_send_each_rank, dest=0)

    return npts_from_rank_to_rank0_map


def count_npts_to_send_each_rank(sending_rank, array_of_ranks):
    """Calculate a dictionary where each key is the destination rank,
    and each value of the number of points that will be sent to that rank.
    """
    ranks_to_send_data, counts = np.unique(array_of_ranks, return_counts=True)
    npts_to_send_each_rank = {r: c for r, c in zip(ranks_to_send_data, counts)}
    try:
        _npts_to_keep = npts_to_send_each_rank.pop(sending_rank)
    except KeyError:
        pass
    return npts_to_send_each_rank


def get_npts_to_send_to_ranks(comm, rank, nranks, npts_from_rank_to_rank0_map):
    """
    """
    if rank == 0:
        npts_to_send_to_ranks = [(t, n) for t, n in npts_from_rank_to_rank0_map[0].items()]
        for r in range(1, nranks):
            comm.send(
                [(t, n) for t, n in npts_from_rank_to_rank0_map[r].items()], dest=r)
    else:
        npts_to_send_to_ranks = comm.recv(source=0)
    return {s[0]: s[1] for s in npts_to_send_to_ranks}
