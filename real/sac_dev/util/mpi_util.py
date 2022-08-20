import copy

import numpy as np

# from mpi4py import MPI

ROOT_PROC_RANK = 0


class MockMPI(object):
    class MockCommWorld(object):
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def Bcast(self, x, root):
            return x

        def Allreduce(self, x_buf, buffer, op):
            # buffer = x_buf.copy()
            np.copyto(buffer, x_buf)

        def Allgather(self, x_buf, buffer):
            # buffer = x_buf.copy()
            np.copyto(buffer, x_buf)

    def __init__(self):
        self.COMM_WORLD = self.MockCommWorld()
        self.SUM = None
        self.PROD = None
        self.MIN = None
        self.MAX = None


MPI = MockMPI()


def get_num_procs():
    return MPI.COMM_WORLD.Get_size()


def get_proc_rank():
    return MPI.COMM_WORLD.Get_rank()


def is_root_proc():
    rank = get_proc_rank()
    return rank == ROOT_PROC_RANK


def bcast(x):
    MPI.COMM_WORLD.Bcast(x, root=ROOT_PROC_RANK)
    return


def reduce_sum(x):
    return reduce_all(x, MPI.SUM)


def reduce_sum_inplace(x, destination):
    MPI.COMM_WORLD.Allreduce(x, destination, op=MPI.SUM)


def reduce_prod(x):
    return reduce_all(x, MPI.PROD)


def reduce_mean(x):
    buffer = reduce_sum(x)
    buffer /= get_num_procs()
    return buffer


def reduce_min(x):
    return reduce_all(x, MPI.MIN)


def reduce_max(x):
    return reduce_all(x, MPI.MAX)


def reduce_all(x, op):
    is_array = isinstance(x, np.ndarray)
    x_buf = x if is_array else np.array([x])
    buffer = np.zeros_like(x_buf)
    MPI.COMM_WORLD.Allreduce(x_buf, buffer, op=op)
    buffer = buffer if is_array else buffer[0]
    return buffer


def gather_all(x):
    x_buf = np.array([x])
    buffer = np.zeros_like(x_buf)
    buffer = np.repeat(buffer, get_num_procs(), axis=0)
    MPI.COMM_WORLD.Allgather(x_buf, buffer)
    buffer = list(buffer)
    return buffer


def reduce_dict_mean(local_dict):
    keys = sorted(local_dict.keys())
    local_vals = np.array([local_dict[k] for k in keys])
    global_vals = reduce_mean(local_vals)

    new_dict = copy.deepcopy(local_dict)
    for i, k in enumerate(keys):
        val = global_vals[i]
        new_dict[k] = val

    return new_dict
