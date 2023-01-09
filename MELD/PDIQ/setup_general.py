#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld.remd import ladder, adaptor, master_runner
from meld import comm, vault
from meld import system
from meld import parse
from meld.system.restraints import LinearRamp,ConstantRamp
import glob
from restraints import *

# number of replicas
N_REPLICAS = 30
# number of steps (units of exchange period)
N_STEPS = 20000
# controles frequence of output
BLOCK_SIZE = 100


def gen_state_templates(index, templates):
    n_templates = len(templates)
    a = system.ProteinMoleculeFromPdbFile(templates[index%n_templates])
    b = system.SystemBuilder(forcefield="ff14sbside")
    c = b.build_system_from_molecules([a])
    pos = c._coordinates
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.SystemState(pos, vel, alpha, energy, [0,0,0])


# MAIN routine
def setup_system():
    # create the system starting from coordinates in template.pdb
    templates = glob.glob('TEMPLATES/*.pdb')
    p = system.ProteinMoleculeFromPdbFile(templates[0])
    b = system.SystemBuilder(forcefield="ff14sbside")
    s = b.build_system_from_molecules([p])
 
    # Create temperature ladder
    s.temperature_scaler = system.GeometricTemperatureScaler(0.0, 0.5, 300., 500.)

    # set the reseidue indicies for the protein, pep1, pep2
    pro_res = range(1,86)
    pep1_res = range(86,100)
    pro_center = (32,'CG')
    #We have a peptide two residues longer than initially simulated
    NTER1 = range(89,94) 
    CTER1 = range(96,101)
    PROT_NTER = [38,48,69,72]
    PROT_CTER = [2,26,30,72,76,80]

    # Restraints
    const_scaler = s.restraints.create_scaler('constant')
    
    # Keep protein close to starting conformation
    rest = make_cartesian_collections(s, const_scaler, pro_res)
    s.restraints.add_as_always_active_list(rest)

    #Peptide one and two: one restraint to be in the pocket and one to be in the shell have to be enforced
   
    dist_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.0, alpha_max=0.6, factor=4.0)
    
    # setup hydrophobic restraints between 3 important residues in peptide and any hydrophobic in protein
    # satisfy only 5
    hydroph_restraints = create_hydrophobes(s,ContactsPerHydroph=1.3,scaler=dist_scaler,group_1=np.array([88,92,95]),group_2=np.array(range(1,86)),CO=False)
    s.restraints.add_selectively_active_collection(hydroph_restraints, 5)
    
    # Restrain peptide within reasonable distance from protein
    scaler3 = s.restraints.create_scaler('constant')
    conf_rest = []
    for index in NTER1:
        conf_rest.append(s.restraints.create_restraint('distance', scaler3,ramp=LinearRamp(0,100,0,1), 
                                                       r1=0.0, r2=0.0, r3=7.0, r4=8.0, k=250.0,
                                                       atom_1_res_index=index, atom_1_name='CA', 
                                                       atom_2_res_index=pro_center[0], atom_2_name=pro_center[1]))
    s.restraints.add_as_always_active_list(conf_rest)

    # create the options
    options = system.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.remove_com = False
    options.use_big_timestep = False # MD timestep (3.3 fs)
    options.use_bigger_timestep = True # MD timestep (4.0 fs)
    options.cutoff = 1.8 # cutoff in nm

    options.use_amap = False # correction to FF12SB
    options.amap_beta_bias = 1.0
    options.timesteps = 11111 # number of MD steps per exchange
    options.minimize_steps = 20000 # init minimization steps

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner, sets up replica exchange details
    l = ladder.NearestNeighborLadder(n_trials=48*48)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)
    remd_runner = master_runner.MasterReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)

    # create and save the initial states
    # create and save the initial states, initialize each replica with a different template
    states = [gen_state_templates(i,templates) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms

# RUN THE SETUP
setup_system()
