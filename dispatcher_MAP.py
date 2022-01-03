import subprocess
from main_MAPv0 import *
my_str = '''
#!/bin/bash
#BSUB -J m2m
#BSUB -o m2m_L{6}_K{7}_mvar{8}.out
#BSUB -e m2m_L{6}_K{7}_mvar{8}.err

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

#BSUB -q bigmem

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
rm *.err
rm *.out
python3 ./main_MAP.py -learn {0} -priors {1} -N_met {2} -N_bug {3} -L {4} -K {5} -meas_var {6} -prior_meas_var {7} -seed {8} -rep_clust {9}
'''

N_met = 20
N_bug = 20
learn = 'all'
pid_list = []
prior_meas_var = 4
meas_var = 0.001
for seed in [0,1]:
    # for meas_var in [0.01, 1]:
    for L, K in [(2,2),(3,3),(6,6)]:
        for repeat_clusters in [0, 1]:
            for priors in ['all','none']:
                f = open('m2m.lsf', 'w')
                f.write(my_str.format(learn, priors, N_met, N_bug, L, K, meas_var, prior_meas_var, seed, repeat_clusters))
                f.close()
                os.system('bsub < {}'.format('m2m.lsf'))