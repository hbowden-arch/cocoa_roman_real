import sys, platform, os
sys.path.insert(0, os.environ['ROOTDIR'] + 
                   '/external_modules/code/CAMB/build/lib.linux-x86_64-'
                   +os.environ['PYTHON_VERSION'])
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
import torch
from cocoa_emu import Config, get_params_list, CocoaModel, get_lhs_params_list
import emcee
from scipy.stats import qmc, norm

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model

from cobaya.conventions import packages_path_input
from cobaya.run import run


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#size = 1
#rank = 0

print(size,rank)

n = 0

configfile = sys.argv[1]
config = Config(configfile)
print(config.params)

if config.init_sample_type == "lhs":
    label_train = f'{config.init_sample_type}_{config.n_lhs}'
    label_valid = label_train
    #print("We don't support LHS any more!")
    #exit(1)
else:
    iss = f'{config.init_sample_type}'
    label_train = iss+f'_t{config.gtemp_t}_{config.gnsamp_t}'
    label_valid = iss+f'_t{config.gtemp_v}_{config.gnsamp_v}'

if(rank==0):
    print("Initializing configuration space data vector dimension!")
    print("N_xi_pm: %d"%(config.probe_size[0]))
    print("N_ggl: %d"%(config.probe_size[1]))
    print("N_w: %d"%(config.probe_size[2]))
    print("N_gk: %d"%(config.probe_size[3]))
    print("N_sk: %d"%(config.probe_size[4]))
    print("N_kk: %d"%(config.probe_size[5]))


def get_params_from_lhs_sample(unit_sample, priors):
    assert len(unit_sample)==len(priors), "Length of the labels not equal to the length of samples"
    
    params = {}
    for i, label in enumerate(priors):
        if('dist' in priors[label]):
            lhs_loc = priors[label]['loc']
            lhs_scale = priors[label]['scale']
            param_i = norm(loc=lhs_loc,scale=lhs_scale).ppf(unit_sample[i])
        else:
            lhs_min = priors[label]['min']
            lhs_max = priors[label]['max']
            #param_i = qmc.scale(unit_sample[:,i], lhs_min, lhs_max)
            param_i = lhs_min + (lhs_max - lhs_min) * unit_sample[i]
        params[label] = param_i
    return params

def get_lhs_params_list(samples, priors):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_lhs_sample(samples[i], priors)
        params_list.append(params)
    return params_list




def get_lhs_samples(N_dim, N_lhs):
    ''' Generate Latin Hypercube sample at parameter space
    Input:
    ======
        - N_dim: 
            Dimension of parameter space
        - N_lhs:
            Number of LH grid per dimension in the parameter space
        - lhs_minmax:
            The boundary of parameter space along each dimension
    Output:
    =======
        - lhs_params:
            LHS of parameter space
    '''
    sample = qmc.LatinHypercube(d=N_dim).random(N_lhs)
    priors = {}
    for i in config.params:
        if('prior' in config.params[i]):
            priors[i] = config.params[i]['prior']

    #print(len(priors),np.shape(unit_sample))
    #assert len(unit_sample[0])==len(priors), "Length of the labels not equal to the length of samples"
    """
    params = {}
    for i, label in enumerate(priors):
        if('dist' in priors[label]):
            lhs_loc = priors[label]['loc']
            lhs_scale = priors[label]['scale']
            param_i = norm(loc=lhs_loc,scale=lhs_scale).ppf(unit_sample[:,i])
        else:
            lhs_min = priors[label]['min']
            lhs_max = priors[label]['max']
            #param_i = qmc.scale(unit_sample[:,i], lhs_min, lhs_max)
            param_i = lhs_min + (lhs_max - lhs_min) * unit_sample[:,i]
        params[label] = param_i
    
    return params"""
    lhs_params = get_lhs_params_list(sample, priors)
    return lhs_params

params_train = get_lhs_samples(config.n_dim, config.n_lhs)
params_train = comm.bcast(params_train, root=0)
#print(train_params['As_1e9'])


np.save(pjoin(config.traindir,f'total_samples_{label_train}.npy'),params_train)
#np.save(pjoin(config.traindir,f'total_samples_{label_valid}.npy'),params_valid)


# ================== Calculate data vectors ==========================

cocoa_model = CocoaModel(configfile, config.likelihood)

print(cocoa_model)

def get_local_data_vector_list(params_list, rank, label):
    ''' Evaluate data vectors dispatched to the local process
    Input:
    ======
        - params_list: 
            full parameters to be evaluated. Parameters dispatched is a subset of the full parameters
        - rank: 
            the rank of the local process
    Outputs:
    ========
        - train_params: model parameters of the training sample
        - train_data_vectors: data vectors of the training sample
    '''
    # print results real time 
    dump_file = pjoin(config.traindir, f'dump/{label}_{rank}-{size}.txt')
    fp = open(dump_file, "w")
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        if ((i-rank*N_local)%20==0):
            print(f'[{rank}/{size}] get_local_data_vector_list: iteration {i-rank*N_local}/{N_local}...')
        if type(params_list[i]) != dict:
            _p = {k:v for k,v in zip(config.running_params, params_list[i])}
        else:
            _p = params_list[i]
        params_arr  = np.array([_p[k] for k in config.running_params])
        # Here it calls cocoa to calculate data vectors at requested parameters
        data_vector = cocoa_model.calculate_data_vector(params_list[i])
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        context = ' '.join([f'{num:e}' for num in np.hstack([params_arr, data_vector])])
        fp.write(context+"\n")
        fp.flush()
    fp.close()
    return train_params_list, train_data_vector_list, None

def get_data_vectors(params_list, comm, rank, label):
    ''' Evaluate data vectors
    This function will further calls `get_local_data_vector_list` to dispatch jobs to and collect training data set from  other processes.
    Input:
    ======
        - params_list:
            Model parameters to be evaluated the model at
        - comm:
            MPI comm
        - rank:
            MPI rank
    Output:
    =======
        - train_params:
            model parameters of the training sample
        - train_data_vectors:
            data vectors of the training sample
    '''
    local_params_list, local_data_vector_list, local_sigma8_list = get_local_data_vector_list(params_list,rank,label)
    comm.Barrier() # Synchronize before collecting results
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list, local_sigma8_list], dest=0)
        train_params       = None
        train_data_vectors = None
        #train_sigma8       = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        #sigma8_list      = local_sigma8_list
        for source in range(1,size):
            new_params_list, new_data_vector_list, new_sigma8_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
            #sigma8_list      = sigma8_list + new_sigma8_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)
        #train_sigma8       = np.vstack(sigma8_list)
    return train_params, train_data_vectors

if(rank==0):
    print("Iteration: %d"%(n))
# ============== Retrieve training sample ======================
if(n==0):
    if(rank==0):
        # retrieve LHS parameters
        # the parameter space boundary is set by config.lhs_mimax, which is the
        # prior boundaries for flat prior and +- 4 sigma for Gaussian prior
        lhs_params = get_lhs_samples(config.n_dim, config.n_lhs)
    else:
        lhs_params = None
    lhs_params = comm.bcast(lhs_params, root=0)
    params_list = lhs_params
else:
    next_training_samples = np.load(pjoin(config.traindir, f'samples_{label}_{n}.npy'))
    params_list = get_params_list(next_training_samples, config.param_labels)
    
#current_iter_samples, current_iter_data_vectors, current_iter_sigma8 = get_data_vectors(params_list, comm, rank, label_train)
train_samples, train_data_vectors, train_sigma8 = get_data_vectors(params_train, comm, rank, label_train)

    
#train_samples      = current_iter_samples
#train_data_vectors = current_iter_data_vectors

print("finished computing data vectors")


#valid_samples, valid_data_vectors, valid_sigma8 = get_data_vectors(params_valid, comm, rank, label_valid)

# ============ Clean training data & save ====================
if(rank==0):
    # ================== Chi_sq cut ==========================
    def get_chi_sq_cut(train_data_vectors):
        chi_sq_list = []
        for dv in train_data_vectors:
            delta_dv = (dv - config.dv_lkl)[config.mask_lkl]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
            chi_sq_list.append(chi_sq)
        chi_sq_arr = np.array(chi_sq_list)
        print(f'chi2 difference [{np.nanmin(chi_sq_arr)}, {np.nanmax(chi_sq_arr)}]')
        select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
        return select_chi_sq
    # ===============================================
    select_chi_sq_train = get_chi_sq_cut(train_data_vectors)
    selected_obj_train = np.sum(select_chi_sq_train)
    total_obj_train    = len(train_data_vectors)
    print(f'[calculate_dv.py] Select {selected_obj_train} training sample out of {total_obj_train}!')
    #select_chi_sq_valid = get_chi_sq_cut(valid_data_vectors)
    #selected_obj_valid = np.sum(select_chi_sq_valid)
    #total_obj_valid    = len(valid_data_vectors)
    #print(f'[calculate_dv.py] Select {selected_obj_valid} training sample out of {total_obj_valid}!')
    # ===============================================
    train_data_vectors = train_data_vectors[select_chi_sq_train]
    train_samples      = train_samples[select_chi_sq_train]
    #train_sigma8       = train_sigma8[select_chi_sq_train]
    #valid_data_vectors = valid_data_vectors[select_chi_sq_valid]
    #valid_samples      = valid_samples[select_chi_sq_valid]
    #valid_sigma8       = valid_sigma8[select_chi_sq_valid]
    # ========================================================
    np.save(pjoin(config.traindir, f'data_vectors_{label_train}.npy'),train_data_vectors)
    np.save(pjoin(config.traindir, f'samples_{label_train}.npy'), train_samples)
    #np.save(pjoin(config.traindir, f'data_vectors_{label_valid}.npy'),valid_data_vectors)
    #np.save(pjoin(config.traindir, f'samples_{label_valid}.npy'), valid_samples)
    # ======================================================== 
    print(f'data vectors saved!')
MPI.Finalize