#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/un.h>
#include <stdint.h>
#include <sys/syscall.h>

#include "../include/cuda-helper.h"
#include "../include/hijack.h"
#include "../include/nvml-helper.h"

extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];
extern char pid_path[];

typedef void (*atomic_fn_ptr)(int, void *);

#define GPU_MAX_NUM 8

static int g_block_x = 1, g_block_y = 1, g_block_z = 1;

static long long g_current_rate[GPU_MAX_NUM] = {};
static long long g_rate_counter[GPU_MAX_NUM] = {};
static int g_active_gpu[GPU_MAX_NUM] = {};
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];

const size_t g_spare_memory = 1ull << 30;

static void activate_rate_watcher();
static void *rate_watcher(void *);
static void rate_estimator(const long long);
static uint32_t g_block_locker = 0;

static void initialization(CUdevice);

/** dynamic rate control */

const char *cuda_error(CUresult code, const char **p) {
  CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, code, p);

  return *p;
}

static ssize_t rio_writen(int fd, void *usrbuf, size_t n) {
  size_t nleft = n;
  ssize_t nwritten;
  char *bufp = usrbuf;

  while (nleft > 0) {
	  if ((nwritten = write(fd, bufp, nleft)) <= 0) {
	    if (errno == EINTR)
		    nwritten = 0;
	    else
		    return -1;
	  }
	  nleft -= nwritten;
	  bufp += nwritten;
  }
  return n;
}


static int open_clientfd(CUdevice device) {
  char SOCKET_PATH[108];
  struct sockaddr_un addr;
  int ret;
  int clientfd;

  /* Create local socket. */

  clientfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (clientfd == -1) {
    LOGGER(FATAL, "socket failed: %s\n", strerror(errno));
  }

  /*
  * For portability clear the whole structure, since some
  * implementations have additional (nonstandard) fields in
  * the structure.
  */

  memset(&addr, 0, sizeof(addr));

  /* Connect socket to socket address. */

  char uuid_str[37] = {};
  for (int i = 0; i < 16; ++i) {
    unsigned char byte = g_uuid[device].bytes[i];
    uuid_str[2*i] = byte / 16;
    uuid_str[2*i+1] = byte % 16;
  }
  for (int i = 0; i < 32; ++i)
    uuid_str[i] = uuid_str[i] >= 10 ? uuid_str[i] - 10 + 'a' : uuid_str[i] + '0';
  uuid_str[32] = 0;

  sprintf(SOCKET_PATH, "/etc/gsharing/rate_%s.sock", uuid_str);
  if (access(SOCKET_PATH, 0) != 0)
    return -1;

  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path));

  ret = connect(clientfd, (const struct sockaddr *) &addr, sizeof(addr));
  if (ret == -1) {
    LOGGER(4, "connect failed: %s\n", strerror(errno));
    if (close(clientfd) < 0)
      LOGGER(FATAL, "open_clientfd: close failed: %s\n", strerror(errno));
    return -1;
  }
  return clientfd;
}


static inline void rate_estimator(const long long kernel_size) {
  CUdevice device = 0;
  const CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxGetDevice error\n");
  }

  if (!g_active_gpu[device])
    initialization(device);
  
  __sync_add_and_fetch_8(&g_rate_counter[device], kernel_size);
}


static void *rate_monitor(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 5000;
  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  struct timespec req = unit_time, rem;

  LOGGER(4, "[%d] rate_monitor start\n", device);

  g_rate_counter[device] = 0;
  while (g_active_gpu[device] > 0) {
    int ret = nanosleep(&req, &rem);
    if (ret < 0) {
      if (errno == EINTR) {
        req = rem;
        continue;
      }
      else LOGGER(FATAL, "nanosleep error: %s\n", strerror(errno));
    }
    else
      req = unit_time;
    
    g_current_rate[device] = g_rate_counter[device];

    g_rate_counter[device] = 0;
  }
  return NULL;
}


inline double shift_window(double rate_window[], const int WINDOW_SIZE, double recv_rate) {
  double max_window_rate = 0;

  for (int i = WINDOW_SIZE-1; i > 0; --i) {
    double mean_rate = (rate_window[i] + rate_window[i-1]) / 2;
    max_window_rate = max_window_rate > mean_rate ? max_window_rate : mean_rate;
    rate_window[i] = rate_window[i-1];
  }
  rate_window[0] = recv_rate;

  return max_window_rate;
}


static void *rate_watcher(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 5000;

  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  const struct timespec listen_time = {
    .tv_nsec = 100 * MILLISEC,
  };
  struct timespec req = listen_time, rem;
  const int WINDOW_SIZE = 5;
  double rate_window[WINDOW_SIZE];
  double max_rate = 0;

  while (g_active_gpu[device] > 0) {

    int clientfd;
    int loop_cnt = 0;
    while ((clientfd = open_clientfd(device)) < 0) {
      if (g_active_gpu[device] <= 0) {
        return NULL;
      }

      nanosleep(&listen_time, &rem);
      
      if (loop_cnt < 20) {
        ++loop_cnt;
        continue;
      }
      else
        loop_cnt = 0;

      double rate_counter =  g_current_rate[device];
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, (double)rate_counter);

      double max_delta = (max_window_rate - max_rate) / max_rate;
      
      if (max_delta >= -0.2 && max_delta <= 0.2)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else
        max_rate = max_window_rate;
    }
    
    if (rio_writen(clientfd, (void *)&max_rate, sizeof(double)) != sizeof(double)) {
      LOGGER(4, "rio_writen error\n");
      continue;
    }

    int ret = 0;
    req = unit_time;
    while (g_active_gpu[device] > 0) {

      ret = nanosleep(&req, &rem);
      if (ret < 0) {
        if (errno == EINTR) {
          req = rem;
          continue;
        }
        else LOGGER(FATAL, "nanosleep error: %s\n", strerror(errno));
      }
      else
        req = unit_time;
      double rate_counter = g_current_rate[device];
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, (double)rate_counter);
      double max_delta = (max_window_rate - max_rate) / max_rate;
      
      if (max_delta >= -0.2 && max_delta <= 0.2)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else
        max_rate = max_window_rate;

      if (rio_writen(clientfd, (void *)&rate_counter, sizeof(double)) != sizeof(double)) {
        LOGGER(4, "rio_writen error\n");
        break;
      }
    }

    close(clientfd);

    max_rate *= 0.8;
  }
  return NULL;
}


static int tgs_set_cpu_affinity(pthread_t thread_id, int core_id) {
    int ret;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    ret = pthread_setaffinity_np(thread_id, sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "failed to set cpu affinity");
        return -1;
    } else {
        ret = pthread_getaffinity_np(thread_id, sizeof(cpuset), &cpuset);
        if (ret != 0) {
            fprintf(stderr, "failed to get cpu affinity");
            return -1;
        } else {
            fprintf(stderr, "set returned by pthread_getaffinity_np() contained:");
            int cnt = 0, cpu_in_set = 0;
            for (int i = 0; i < CPU_SETSIZE; i++) {
                if (CPU_ISSET(i, &cpuset)) {
                    cnt++;
                    cpu_in_set = i;
                    fprintf(stderr, "  cpu=%d", i);
                }
            }
            // this should not happen though
            if (cnt != 1 || cpu_in_set != core_id) {
                fprintf(stderr, "failed to set cpu affinity with cpu=%d", core_id);
                return -1;
            }
        }
    }
    return 0;
}


static void activate_rate_watcher(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, rate_watcher, (void *)(uintptr_t)device);
  tgs_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("rate_watcher");
#else
  pthread_setname_np(tid, "rate_watcher");
#endif
}


static void activate_rate_monitor(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, rate_monitor, (void *)(uintptr_t)device);
  tgs_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("rate_monitor");
#else
  pthread_setname_np(tid, "rate_monitor");
#endif
}

static inline void initialization(CUdevice device) {
  g_active_gpu[device] = 1;
  
  fprintf(stderr, "initialize device %d\n", device);

  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid, &g_uuid[device], device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuDeviceGetUuid error\n");
  }

  int gpu_id = 0;
  for (int i = 0; i < 16; ++i) {
    gpu_id += (int)g_uuid[device].bytes[i];
  }
  gpu_id = (gpu_id % 8 + 8) % 8;
  g_gpu_id[device] = gpu_id;

  activate_rate_watcher(device);
  activate_rate_monitor(device);
}

/** hijack entrypoint */
CUresult cuDriverGetVersion(int *driverVersion) {
  CUresult ret;

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDriverGetVersion, driverVersion);
  return ret;
}

CUresult cuInit(unsigned int flag) {
  CUresult ret;

  fprintf(stderr, "[cuInit] thread %lu\n", (unsigned long)syscall(SYS_gettid));

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, flag);
  return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        flags);
  if (ret != CUDA_SUCCESS)
    return ret;

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);
  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  *pPitch = ROUND_UP(WidthInBytes, 128);
  size_t bytesize = ROUND_UP(*pPitch * Height, ElementSizeBytes);
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes) {
  *pPitch = ROUND_UP(WidthInBytes, 128);
  size_t bytesize = ROUND_UP(*pPitch * Height, ElementSizeBytes);
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}


CUresult cuMemFree_v2(CUdeviceptr dptr) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree_v2, dptr);
}


CUresult cuMemFree(CUdeviceptr dptr) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree, dptr);
}

CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;


  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;


  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMipmappedArrayCreate, pHandle,
                        pMipmappedArrayDesc, numMipmapLevels);
  return ret;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes -= g_spare_memory;
  return ret;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes -= g_spare_memory;
  return ret;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*free > g_spare_memory) ? (*free - g_spare_memory) : 0;
  return ret;
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*free > g_spare_memory) ? (*free - g_spare_memory) : 0;
  return ret;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  int leastPriority, greatestPriority;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetStreamPriorityRange, &leastPriority, &greatestPriority);
  if (ret == CUDA_SUCCESS) {
    int priority = (leastPriority + greatestPriority - 1) / 2;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreateWithPriority, phStream, Flags, priority);
    if (ret == CUDA_SUCCESS) {
      return ret;
    }
    else {
      fprintf(stderr, "\n[ERROR] cuStreamCreateWithPriority failed\n");
    }
  }
  else {
    fprintf(stderr, "\n[ERROR] cuCtxGetStreamPriorityRange failed\n");
  }
  
  return CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreate, phStream, Flags);
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                    int priority) {
  int leastPriority, greatestPriority;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetStreamPriorityRange, &leastPriority, &greatestPriority);
  if (ret == CUDA_SUCCESS) {
    leastPriority = (leastPriority + greatestPriority - 1) / 2;
    if (priority > leastPriority) {
      priority = leastPriority;
    }
  }
  else {
    fprintf(stderr, "\n[ERROR] cuCtxGetStreamPriorityRange failed\n");
  }
  return CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreateWithPriority,
                         phStream, flags, priority);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra) {
  rate_estimator(gridDimX * gridDimY * gridDimZ);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel_ptsz, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
  rate_estimator(gridDimX * gridDimY * gridDimZ);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunch(CUfunction f) {
  rate_estimator(1);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  rate_estimator(gridDimX * gridDimY * gridDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel_ptsz, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams) {
  rate_estimator(gridDimX * gridDimY * gridDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  rate_estimator(grid_width * grid_height);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f, grid_width,
                         grid_height);
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  rate_estimator(grid_width * grid_height);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGridAsync, f, grid_width,
                         grid_height, hStream);
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  while (!CAS(&g_block_locker, 0, 1));

  g_block_x = x;
  g_block_y = y;
  g_block_z = z;

  LOGGER(5, "Set block shape: %d, %d, %d", x, y, z);

  while (!CAS(&g_block_locker, 1, 0));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuFuncSetBlockShape, hfunc, x, y,
                         z);
}

CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy_ptds, dst, src,
                         ByteCount);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy, dst, src, ByteCount);
}

CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                            CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyAsync_ptsz, dst, src,
                         ByteCount, hStream);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                       CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyAsync, dst, src, ByteCount,
                         hStream);
}

CUresult cuMemcpyPeer_ptds(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeer_ptds, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount);
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext,
                      size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeer, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount);
}

CUresult cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice, CUcontext dstContext,
                                CUdeviceptr srcDevice, CUcontext srcContext,
                                size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeerAsync_ptsz, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeerAsync, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoD_v2_ptds, dstDevice,
                         srcHost, ByteCount);
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoD_v2, dstDevice,
                         srcHost, ByteCount);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                      size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoD, dstDevice, srcHost,
                         ByteCount);
}

CUresult cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoDAsync_v2_ptsz,
                         dstDevice, srcHost, ByteCount, hStream);
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoDAsync_v2, dstDevice,
                         srcHost, ByteCount, hStream);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoDAsync, dstDevice,
                         srcHost, ByteCount, hStream);
}

CUresult cuMemcpyDtoH_v2_ptds(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH_v2_ptds, dstHost,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH_v2, dstHost,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH, dstHost, srcDevice,
                         ByteCount);
}

CUresult cuMemcpyDtoHAsync_v2_ptsz(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoHAsync_v2_ptsz, dstHost,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoHAsync_v2, dstHost,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoHAsync, dstHost,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoD_v2_ptds, dstDevice,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoD_v2, dstDevice,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                      size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoD, dstDevice, srcDevice,
                         ByteCount);
}

CUresult cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoDAsync_v2_ptsz,
                         dstDevice, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoDAsync_v2, dstDevice,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoDAsync, dstDevice,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DUnaligned_v2_ptds,
                         pCopy);
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DUnaligned_v2, pCopy);
}

CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DUnaligned, pCopy);
}

CUresult cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DAsync_v2_ptsz, pCopy,
                         hStream);
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DAsync_v2, pCopy,
                         hStream);
}

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DAsync, pCopy, hStream);
}

CUresult cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3D_v2_ptds, pCopy);
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3D_v2, pCopy);
}

CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3D, pCopy);
}

CUresult cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DAsync_v2_ptsz, pCopy,
                         hStream);
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DAsync_v2, pCopy,
                         hStream);
}

CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DAsync, pCopy, hStream);
}

CUresult cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeer_ptds, pCopy);
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeer, pCopy);
}

CUresult cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy,
                                  CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeerAsync_ptsz, pCopy,
                         hStream);
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                             CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeerAsync, pCopy,
                         hStream);
}

/*
 *  Context Management
 */

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxCreate_v2, pctx, flags, dev);
  return ret;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxCreate, pctx, flags, dev);
  return ret;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetCurrent, ctx);
  return ret;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPushCurrent_v2, ctx);
  return ret;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPushCurrent, ctx);
  return ret;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
  }
  else fprintf(stderr, "[%d] cuCtxDestroy_v2!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy_v2, ctx);
}

CUresult cuCtxDestroy(CUcontext ctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxDestroy error\n");
  }
  else fprintf(stderr, "[%d] cuCtxDestroy!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy, ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxPopCurrent_v2 error\n");
  }
  else fprintf(stderr, "[%d] cuCtxPopCurrent_v2!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent_v2, pctx);
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxPopCurrent error\n");
  }
  else fprintf(stderr, "[%d] cuCtxPopCurrent!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent, pctx);
}

/*
 *  Primary Context Management
 */

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRetain, pctx,
                         dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxGetDevice error\n");
  }
  else fprintf(stderr, "[%d] cuDevicePrimaryCtxRelease!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRelease, dev);
}