#!/bin/bash
# submit-job.sh - Submit a job to the SLURM cluster
# Usage: ./submit-job.sh "<command>" [--gpu] [--name <job-name>]
# Example: ./submit-job.sh "flwr run" --gpu
# Evaluation: ./submit-job.sh "evaluate.py" --gpu

set -e
REPO_DIR="$HOME/latebloomers"

# Check if coldstart directory exists
[ ! -d "$REPO_DIR" ] && { echo "Error: Directory $REPO_DIR does not exist"; exit 1; }

[ $# -lt 1 ] && { echo "Usage: $0 \"<command>\" [--gpu] [--name <job-name>]"; exit 1; }

COMMAND=$1; shift
NAME_SUFFIX=""
CPUS=8; MEM=46G; QOS=cpu_qos; PARTITION_ARGS="#SBATCH --partition=cpu"; VENV_NAME="hackathon-venv-cpu"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) CPUS=6; MEM=120G; QOS=gpu_qos; PARTITION_ARGS="#SBATCH --gres=gpu:1
#SBATCH --partition=gpu"; VENV_NAME="hackathon-venv"; shift ;;
    --name) NAME_SUFFIX="$2"; shift 2 ;;
    *) shift ;;
  esac
done

[ -z "$NAME_SUFFIX" ] && NAME_SUFFIX="$(date +%H%M%S)"

# Use /home/root/logs for root user, otherwise use team's logs directory
ACTUAL_USER=$(id -un)
if [ "$ACTUAL_USER" = "root" ]; then
  LOG_DIR="/home/root/logs"
else
  LOG_DIR="/home/${ACTUAL_USER}/logs"
fi
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --output ${LOG_DIR}/job%j_${NAME_SUFFIX}.out
#SBATCH --job-name ${NAME_SUFFIX}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --qos=${QOS}
#SBATCH --time=00:15:00
${PARTITION_ARGS}

set -e
echo "=== Job \${SLURM_JOB_ID} started at \$(date) on \${SLURMD_NODENAME} ==="

export SCRATCH_DIR=/scratch/\${USER}/\${SLURM_JOB_ID}
mkdir -p "\$SCRATCH_DIR"

cleanup() {
  local exit_code=\$?
  echo "=== Copying best model ==="
  MODELS_SCRATCH="\$SCRATCH_DIR/repo/models"
  if [ -d "\$MODELS_SCRATCH" ]; then
    BEST=\$(ls \$MODELS_SCRATCH/*.pt 2>/dev/null | sed 's/.*_auroc\\([0-9]\\{4\\}\\)\\.pt/\\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "\$BEST" ] && cp "\$BEST" /home/${USER}/models/ && echo "  \$(basename \$BEST)"
  fi
  rm -rf "\$SCRATCH_DIR"
  echo "=== Job \${SLURM_JOB_ID} finished with exit code \$exit_code ==="
  exit \$exit_code
}
trap cleanup EXIT

rsync -a "${REPO_DIR}/" "\$SCRATCH_DIR/repo/" && cd "\$SCRATCH_DIR/repo"
source /home/$USER/${VENV_NAME}/bin/activate

# Export environment
export JOB_NAME="job\${SLURM_JOB_ID}_${NAME_SUFFIX}"
export DATASET_DIR="/home/${USER}/xray-data"
export JOB_SCRATCH="\${SLURM_TMPDIR:-\${TMPDIR:-/tmp}}/job-\${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="\$JOB_SCRATCH/miopen-cache"
export MIOPEN_USER_DB_PATH="\$JOB_SCRATCH/miopen-db"
export XDG_CACHE_HOME="\$JOB_SCRATCH/xdg-cache"

mkdir -p /home/${USER}/models "\$JOB_SCRATCH" "\$MIOPEN_CUSTOM_CACHE_DIR" "\$MIOPEN_USER_DB_PATH" "\$XDG_CACHE_HOME"

echo "=== Running: ${COMMAND} ==="
${COMMAND}
EOF
)

JOB_NAME="job${JOB_ID}_${NAME_SUFFIX}"
echo "$USER scheduled job $JOB_ID ($JOB_NAME) > $COMMAND"
echo "Logs: ${LOG_DIR}/${JOB_NAME}.out"