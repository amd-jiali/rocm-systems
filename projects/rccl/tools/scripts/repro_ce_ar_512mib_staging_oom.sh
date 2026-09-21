#!/bin/bash
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH --job-name=ce-ar-oom-repro
#SBATCH -o /home/jialili/logs/nccl-sync-v2-31/ce_ar_oom_repro_%j.out
#SBATCH -e /home/jialili/logs/nccl-sync-v2-31/ce_ar_oom_repro_%j.err
#SBATCH -x cv350-rck-g03-e14-18,cv350-rck-g03-c14-08

# Restore 512 MiB eager CE AllReduce staging, then run CeMPI_AllReduce.LargeMessage.
# See docs/repro-ce-ar-512mib-staging-oom.md. Validated as Slurm job 41798.

set -u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rcclx
module load rocm/7.0.2.2
export ROCM_PATH=/opt/COE_modules/rocm/rocm-7.0.2.2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RDIR=$(cd "$SCRIPT_DIR/../.." && pwd)
WT=$(cd "$RDIR/../.." && pwd)
BDIR=${BDIR:-$RDIR/build/utfix}
MPI_PATH=${MPI_PATH:-$HOME/mpich/install}
OUT=$HOME/logs/nccl-sync-v2-31/ce-ar-oom/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT" "$OUT/rank-logs"

echo "=== job ${SLURM_JOB_ID:-none} on ${SLURM_JOB_NODELIST:-$(hostname)} ==="
echo "=== started $(date) host=$(hostname) ==="
echo "WT=$WT RDIR=$RDIR BDIR=$BDIR"
git -C "$WT" log -1 --oneline
git -C "$WT" status -sb -- projects/rccl/src/include/ce_coll.h projects/rccl/src/ce_coll.cc
echo "=== nodes ==="; scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null || hostname

export LD_LIBRARY_PATH=$BDIR:$BDIR/test/unit/plugins:$MPI_PATH/lib:${LD_LIBRARY_PATH:-}
export HSA_NO_SCRATCH_RECLAIM=1
export RSMI_MUTEX_THREAD_ONLY=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_HCA=${NCCL_IB_HCA:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-104}
export RCCL_MSCCL_ENABLE=0
export RCCL_IB_QPS_PER_P2P=1
export NCCL_IB_QPS_PER_CONNECTION=4
export FI_PROVIDER=tcp
export FI_TCP_IFACE=fenic0
ulimit -c 0

echo "########## BUILD (rccl + rccl-UnitTestsMPI) ##########"
mkdir -p "$BDIR"
if [ ! -f "$BDIR/CMakeCache.txt" ]; then
  cmake -S "$RDIR" -B "$BDIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DENABLE_MPI_TESTS=ON \
    -DENABLE_HOST_API_TESTS=ON \
    -DMPI_PATH="$MPI_PATH" \
    -DGPU_TARGETS=gfx950 \
    -DENABLE_WARP_SPEED=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PLUGIN_EXAMPLES=OFF \
    -DROCM_PATH="$ROCM_PATH"
fi
if ! cmake --build "$BDIR" -j "$(nproc)" --target rccl rccl-UnitTestsMPI; then
  echo "BUILD FAILED"; exit 1
fi
MPI_UT=$BDIR/test/rccl-UnitTestsMPI
ls -lh --time-style=long-iso "$MPI_UT"

echo "########## RUN CeMPI_AllReduce.LargeMessage ##########"
export NCCL_CTA_POLICY=2
export NCCL_CUMEM_ENABLE=1
export RCCL_CE_ALLREDUCE=1
export NCCL_MAX_P2P_NCHANNELS=8
export NCCL_DEBUG=INFO
export RCCL_TEST_LOG_DIR="$OUT/rank-logs"

srun --mpi=pmi2 --kill-on-bad-exit=0 --export=ALL \
  -N 1 --ntasks-per-node=8 \
  "$MPI_UT" --gtest_filter=CeMPI_AllReduce.LargeMessage \
  > "$OUT/LargeMessage.log" 2>&1
rc=$?
echo "gtest exit=$rc"
tail -80 "$OUT/LargeMessage.log"
rg -n "out of memory|Init CE, rank|HIP failure|Which is: 1|FAILED" \
  "$OUT/LargeMessage.log" "$OUT/rank-logs"/* 2>/dev/null | head -80
echo "=== finished $(date) ==="
exit 0
