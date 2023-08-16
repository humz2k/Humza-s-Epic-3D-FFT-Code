import numpy as np

def check(ng,dims):
    local_dims = ng / dims
    if np.sum(local_dims == local_dims.astype(int)) != 3:
        return False
    local_dims = local_dims.astype(int)
    nlocal = local_dims[0] * local_dims[1] * local_dims[2]
    if (nlocal % dims[2]) != 0:
        return False
    if (nlocal % dims[1]) != 0:
        return False
    if (nlocal % (dims[0] * dims[2])) != 0:
        return False
    if (local_dims[1] % dims[2]) != 0:
        return False
    if (ng % dims[1]) != 0:
        return False
    if (ng % (dims[0] * dims[2])) != 0:
        return False

print(check(960,np.array([3,2,2])))