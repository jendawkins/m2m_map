import subprocess
from main_MAP import *
my_str = '''
#!/bin/bash
#BSUB -J cdiff
#BSUB -o output/m2m.out
#BSUB -e output/m2m.err

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q normal

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here

cd /PHShome/jjd65/m2m_map

python3 ./main_MAP.py -learn {0} -priors {1} -case {2} -N_met {3} -N_bug {4} -N_nuisance {5} -L {6} -K {7}
'''

N_met = 20
N_bug = 22
learn = 'all'
pid_list = []
for case in [1, 2]:
    for L in [2,4,6]:
        for K in [2,4,6]:
            for priors in ['all','none']:
                if case == 1:
                    n_nuisance = 0
                    f = open('m2m.lsf', 'w')
                    f.write(my_str.format(learn, priors, case, N_met, N_bug, n_nuisance, L, K))
                    f.close()
                    os.system('bsub < {}'.format('m2m.lsf'))
                else:
                    for n_nuisance in [2,4,8,16,18]:
                        cmd = my_str.format(learn, priors, case, N_met, N_bug, n_nuisance, L, K)
                        f = open('m2m.lsf', 'w')
                        f.write(my_str.format(learn, priors, case, N_met, N_bug, n_nuisance))
                        f.close()
                        os.system('bsub < {}'.format('m2m.lsf'))