import numpy as np
from meld.remd import ladder, adaptor, master_runner
from meld import comm, vault
from meld import system
from meld import parse
from meld.system.restraints import LinearRamp,ConstantRamp


hydrophobes = 'AILMFPWV'
hydrophobes_res = ['ALA','ILE','LEU','MET','PHE','PRO','TRP','VAL']


def non_interacting(g,s,i,j,name_i,name_j,scaler=None):
    g.append(s.restraints.create_restraint('distance', scaler,ramp=LinearRamp(0,100,0,1), r1=0.0, r2=3.0, r3=100.0, r4=101.0, k=250.0,
            atom_1_res_index=i, atom_1_name=name_i, atom_2_res_index=j, atom_2_name=name_j))

def exclude_restraint(g,s,i,j,name_i,name_j,scaler=None):
    g.append(s.restraints.create_restraint('distance', scaler,ramp=LinearRamp(0,100,0,1), r1=4.0, r2=5.0, r3=7.0, r4=8.0, k=250.0,
            atom_1_res_index=i, atom_1_name=name_i, atom_2_res_index=j, atom_2_name=name_j))

def append_restraint(g,s,i,j,name_i,name_j,scaler=None):
    try:
        g.append(s.restraints.create_restraint('distance', scaler,ramp=LinearRamp(0,100,0,1), r1=0.0, r2=0.0, r3=0.8, r4=1.0, k=250.0,
            atom_1_res_index=i, atom_1_name=name_i, atom_2_res_index=j, atom_2_name=name_j))
    except:
        pass


# Restraints on Protein CA
def make_cartesian_collections(s, scaler, residues, delta=0.35, k=250.):
    cart = []
    backbone = ['CA']
    #Residues are 1 based
    #index of atoms are 1 base
    for i in residues:
        for b in backbone:
            atom_index = s.index_of_atom(i,b) - 1
            x,y,z = s.coordinates[atom_index]/10.
            rest = s.restraints.create_restraint('cartesian',scaler, res_index=i, atom_name=b,
                x=x, y=y, z=z, delta=delta,force_const=k)
            cart.append(rest)
    return cart


def create_contacts(s,scaler=None,group1=[],group2=[]):
    #Group1 will be any heavy atom in protein 
    #Group2 will be backbone atoms in peptide
    #Should have two contacts working out of all possibilities.
    scaler = scaler if scaler else s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    contact_restraints = []
    #Heavy atoms for each residue
    atoms = {"ALA":['N','C','O','CA','CB'],
             "VAL":['N','C','O','CA','CB','CG1','CG2'],
             "LEU":['N','C','O','CA','CB','CG','CD1','CD2'],
             "ILE":['N','C','O','CA','CB','CG1','CG2','CD1'],
             "PHE":['N','C','O','CA','CB','CG','CD1','CE1','CZ','CE2','CD2'],
             "TRP":['N','C','O','CA','CB','CG','CD1','NE1','CE2','CZ2','CH2','CZ3','CE3','CD2'],
             "MET":['N','C','O','CA','CB','CG','SD','CE'],
             "PRO":['N','C','O','CD','CG','CB','CA'],
             "ASP":['N','C','O','CA','CB','CG','OD1','OD2'],
             "GLU":['N','C','O','CA','CB','CG','CD','OE1','OE2'],
             "LYS":['N','C','O','CA','CB','CG','CD','CE','NZ'],
             "ARG":['N','C','O','CA','CB','CG','CD','NE','CZ','NH1','NH2'],
             "HIS":['N','C','O','CA','CB','CG','ND1','CE1','NE2','CD2'],
             "HID":['N','C','O','CA','CB','CG','ND1','CE1','NE2','CD2'],
             "HIE":['N','C','O','CA','CB','CG','ND1','CE1','NE2','CD2'],
             "HIP":['N','C','O','CA','CB','CG','ND1','CE1','NE2','CD2'],
             "GLY":['N','C','O','CA'],
             "SER":['N','C','O','CA','CB','OG'],
             "THR":['N','C','O','CA','CB','CG2','OG1'],
             "CYS":['N','C','O','CA','CB','SG'],
             "CYX":['N','C','O','CA','CB','SG'],
             "TYR":['N','C','O','CA','CB','CG','CD1','CE1','CZ','OH','CE2','CD2'],
             "ASN":['N','C','O','CA','CB','CG','OD1','ND2'],
             "GLN":['N','C','O','CA','CB','CG','CD','OE1','NE2'],
             "BCK":['N','C','O','CA']}

    #s.residue_numbers and s.residue_names have as many instances as atoms
    #we first use zip to create tuples and then set to create unique (disordered) lists of tuples
    #the sorted organizes them. Finally the dict creates an instance (one based) of residue to residue name
    sequence = [(i,j) for i,j in zip(s.residue_numbers,s.residue_names)]
    sequence = sorted(set(sequence))
    sequence = dict(sequence)

    for index_i in group1:
        for index_j in group2:
            res_i = sequence[index_i]
            res_j = sequence[index_j]
            atoms_i = atoms[res_i]
            atoms_j = atoms['BCK']

            local_contact = []
            for a_i in atoms_i:
                for a_j in atoms_j:
                    #EvFold paper defines contact as 5angstrong any atom contact. We are only using heavy atoms
                    #Hence, should allow to get closer.
                    local_contact.append(s.restraints.create_restraint('distance', scaler, LinearRamp(0,100,0,1),r1=0.0, r2=0.0, r3=0.50, r4=0.65, k=250.0,
                atom_1_res_index=index_i, atom_1_name=a_i, atom_2_res_index=index_j, atom_2_name=a_j))

            contact_restraints.append(s.restraints.create_restraint_group(local_contact,1))

    return(contact_restraints)

    #all_rest = len(contact_restraints)
    #active = int( 2 )
    #s.restraints.add_selectively_active_collection(contact_restraints, active)

def get_dist_restraints(filename, s, scaler):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])
            name_i = cols[1]
            j = int(cols[2])
            name_j = cols[3]
            dist = float(cols[4]) / 10.

            rest = s.restraints.create_restraint('distance', scaler,LinearRamp(0,100,0,1),
                                                 r1=0.0, r2=0.0, r3=dist, r4=dist+0.2, k=250,
                                                 atom_1_res_index=i, atom_2_res_index=j,
                                                 atom_1_name=name_i, atom_2_name=name_j)
            rest_group.append(rest)
    return dists

def create_hydrophobes(s,ContactsPerHydroph=1.3,scaler=None,group_1=np.array([]),group_2=np.array([]),CO=True):
    atoms = {"ALA":['CA','CB'],
             "VAL":['CA','CB','CG1','CG2'],
             "LEU":['CA','CB','CG','CD1','CD2'],
             "ILE":['CA','CB','CG1','CG2','CD1'],
             "PHE":['CA','CB','CG','CD1','CE1','CZ','CE2','CD2'],
             "TRP":['CA','CB','CG','CD1','NE1','CE2','CZ2','CH2','CZ3','CE3','CD2'],
             "MET":['CA','CB','CG','SD','CE'],
             "PRO":['CD','CG','CB','CA']}
    #Groups should be 1 centered
    n_res = s.residue_numbers[-1]
    group_1 = group_1 if group_1.size else np.array(range(n_res))+1
    group_2 = group_2 if group_2.size else np.array(range(n_res))+1
    scaler = scaler if scaler else s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    #Get a list of names and residue numbers, if just use names might skip some residues that are two
    #times in a row
    #make list 1 centered
    sequence = [(i,j) for i,j in zip(s.residue_numbers,s.residue_names)]
    sequence = sorted(set(sequence))
    sequence = dict(sequence)

    #Get list of groups with only residues that are hydrophobs
    group_1 = [ res for res in group_1 if (sequence[res] in hydrophobes_res) ]
    group_2 = [ res for res in group_2 if (sequence[res] in hydrophobes_res) ]


    pairs = []
    hydroph_restraints = []
    for i in group_1:
        for j in group_2:

            # don't put the same pair in more than once
            if ( (i,j) in pairs ) or ( (j,i) in pairs ):
                continue

            if ( i ==j ):
                continue

            if (abs(i-j)< 7):
                continue
            pairs.append( (i,j) )
           
            atoms_i = atoms[sequence[i]]
            atoms_j = atoms[sequence[j]]
   
            local_contact = []
            for a_i in atoms_i:
                for a_j in atoms_j:
                    if CO:
                        tmp_scaler = scaler(abs(i-j), 'hydrophobic')
                    else:
                        tmp_scaler = scaler
                    local_contact.append(s.restraints.create_restraint('distance', tmp_scaler, LinearRamp(0,100,0,1),r1=0.0, r2=0.0, r3=0.50, r4=0.70, k=250.0,
            atom_1_res_index=i, atom_1_name=a_i, atom_2_res_index=j, atom_2_name=a_j))

            hydroph_restraints.append(s.restraints.create_restraint_group(local_contact,1))
    all_rest = len(hydroph_restraints)
    return(hydroph_restraints)
    #active = int( ContactsPerHydroph * len(group_1) )
    #s.restraints.add_selectively_active_collection(hydroph_restraints, active)



