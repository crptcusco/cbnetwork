#!/bin/bash
#SBATCH --nodes=3            # Numero de Nós
#SBATCH --ntasks-per-node=16 # Numero de tarefas por Nó
#SBATCH --ntasks=384         # Numero de tarefas
#SBATCH -p cpu         # Fila (partition) a ser utilizada
#SBATCH -J CRPT              # Nome job
#SBATCH --time=01:00:00
#SBATCH -e log/slurm-%j.err
#SBATCH -o log/slurm-%j.out

mkdir -p log tmp
NETINFO=log/netinfo.$SLURM_JOBID.log

#Exibe os nos alocados para o Job
echo $SLURM_JOB_NODELIST
# nodeset -e $SLURM_JOB_NODELIST  # Might have compatibility issues

echo -n Entering in:
pwd
cd $SLURM_SUBMIT_DIR

echo $SLURM_SUBMIT_HOST >> $NETINFO
ip addr >> $NETINFO

echo Loading modules ...
#Language, applications, and other configurations
#module load python/3.12.1
#module load python/3.9.12
#module load python/3.9.6
module load python/3.8.2
#module load python/3.9.1
#module load python/3.7.2
#module load python/3.6.9
module load minisat/2.2.0


echo Creating the virtual enviroment ...
# Try creating the virtual environment using venv (assuming Python 3.7+)
python3 -m venv venv
# Path /prj/deephash/carlos.tovar/venv
source venv/bin/activate  # Activate virtual environment

# Install libraries within the virtual environment
pip install --upgrade pip
pip install --upgrade typeguard
pip install parsl satispy networkx pandas matplotlib # Assuming you have internet access on the cluster

# Now the script can access the installed libraries
cd /scratch/deephash/carlos.tovar/cbnetwork/hpc/sdumont

echo Starting Test Script
export PYTHONPATH=/scratch/deephash/carlos.tovar/cbnetwork/:$PYTHONPATH
echo PYTHONPATH
python3 s_generate_cbn.py
#python3 s_square.py
#python3 s_example_valor_pi.py
