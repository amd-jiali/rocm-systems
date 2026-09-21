# Reproduce CE AllReduce 512 MiB staging OOM

**Do not merge.** This branch restores develop-style CE AllReduce staging so the
v2.31 gfx950 unit-test failure can be re-run. The NCCL 2.31.2 sync default
(`NCCL_CE_AR_STAGING_BYTES = 16 MiB`, lazy alloc) is the workaround, not this.

## What fails

`CeMPI_AllReduce.LargeMessage` is an **8 MiB** AllReduce. Eligibility is still
256 MiB. Staging is **always** `2 × NCCL_CE_AR_MAX_MSG_BYTES = 512 MiB` of
uncached RDMA VMM, allocated in `ncclCeInit` when `RCCL_CE_ALLREDUCE=1`.

After comm init (~2–3 GiB P2P/shareable maps), `hipMemCreate(512 MiB)` returns
HIP OOM. Idle HBM is still hundreds of GiB; an isolated 512 MiB alloc succeeds.
This is late contiguous VMM, not a 256 MiB-message reject.

## Validated

- Job **41798**, 2026-09-21, node `cv350-rck-g03-e09-18` (gfx950), 1×8
- `ncclAllReduce` → `1` (`ncclUnhandledCudaError`)
- Rank 0: `Init CE` then `allocator.cc:76 HIP failure 'out of memory'`
- Call stack: `ncclCeEnsureAllReduceStaging` (`ce_coll.cc` ~1980) from `ncclCeInit` (~220)

## Run

Build `rccl-UnitTestsMPI` (`ENABLE_MPI_TESTS=ON`, `GPU_TARGETS=gfx950`) then:

```bash
sbatch projects/rccl/tools/scripts/repro_ce_ar_512mib_staging_oom.sh
```

Or inside an exclusive 1-node allocation:

```bash
srun --mpi=pmi2 --kill-on-bad-exit=0 --export=ALL \
  -N 1 --ntasks-per-node=8 \
  env NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 RCCL_CE_ALLREDUCE=1 \
      NCCL_MAX_P2P_NCHANNELS=8 NCCL_DEBUG=INFO \
  ./rccl-UnitTestsMPI --gtest_filter=CeMPI_AllReduce.LargeMessage
```

Expect FAIL + `HIP failure 'out of memory'` on every rank.

To sweep slot size, change `ncclCeAllReduceMaxChunkBytes` to divide
`NCCL_CE_AR_STAGING_BYTES` instead of `NCCL_CE_AR_MAX_MSG_BYTES` and set
`NCCL_CE_AR_STAGING_BYTES` to 16/64/128/256 MiB. Keep alloc lazy if you only
want the AllReduce path, not AlltoAll paying at CE init.
