/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    size_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    size_t offset;
    size_t dataOffset;
    int nelem;
    int rankDest;

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg);

    for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
      /////////////// begin AllGather steps ///////////////
      nelem = min(chunkCount, partCount - elemOffset);
      dataOffset = partOffset + elemOffset;

      // step 0: push data to next GPU
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;

      if (inputBuf + dataOffset == outputBuf + offset) { // In place
        prims.directSend(dataOffset, offset, nelem);
      } else {
        prims.directCopySend(dataOffset, offset, nelem);
      }

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = dataOffset + rankDest * count;

        prims.directRecvCopySend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = ringRanks[1];
      offset = dataOffset + rankDest * count;

      // Final wait/copy.
      prims.directRecv(offset, nelem);
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncScaledAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncScaledAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncScaledAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};