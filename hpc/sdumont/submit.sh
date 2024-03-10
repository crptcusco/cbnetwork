#!/bin/bash
#SBATCH --nodes=3            #Numero de Nós
#SBATCH --ntasks-per-node=24 #Numero de tarefas por Nó
#SBATCH --ntasks=384         #Numero de tarefas
#SBATCH -p cpu_share         #Fila (partition) a ser utilizada
#SBATCH -J CRPT              #Nome job
#SBATCH --time=01:00:00
#SBATCH -e log/slurm-%j.err
#SBATCH -o log/slurm-%j.out

mkdir -p log tmp
NETINFO=log/netinfo.$SLURM_JOBID.log

#Exibe os nos alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

echo -n Entering in:
pwd
cd $SLURM_SUBMIT_DIR

echo $SLURM_SUBMIT_HOST >> $NETINFO
ip addr >> $NETINFO

echo Loading modules
#Language, applications, and other configurations
module load python/3.9.12
module load minisat/2.2.0

echo Creating the virtual enviroment
python3 -m venv venv
source venv/bin/activate
pip install parsl satispy

#acessa o diretório onde o script está localizado
cd /scratch/deephash/carlos.tovar/cbnetwork/hpc/sdumont

#executa o script
echo Starting Test Script
python3 test_valor_pi.py