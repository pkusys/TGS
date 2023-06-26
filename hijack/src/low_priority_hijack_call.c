#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/types.h>
#include <assert.h>
#include <sys/un.h>
#include <stdint.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include "../include/cuda-helper.h"
#include "../include/hijack.h"
#include "../include/nvml-helper.h"
#include "../include/nvml-subset.h"

extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];
extern char pid_path[];
static const size_t g_spare_memory = 1ull << 30;
static size_t g_used_memory = 0;

static int g_block_x = 1, g_block_y = 1, g_block_z = 1;
static uint32_t g_block_locker = 0;

#define GPU_MAX_NUM 8

static long long g_rate_counter[GPU_MAX_NUM] = {};
static long long g_rate_limit[GPU_MAX_NUM] = {};
static long long g_rate_control_flag[GPU_MAX_NUM] = {};
static long long g_current_rate[GPU_MAX_NUM] = {};
static int g_active_gpu[GPU_MAX_NUM] = {};
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];

const long long LIMIT_INITIALIZER = 20000;
const long long RATE_MIN = 1000;

#define TGS_SLOW_START 0
#define TGS_CONGESTION_AVOIDANCE 1

static const struct timespec g_cycle = {
    .tv_sec = 0,
    .tv_nsec = TIME_TICK * MILLISEC,
};


struct MemRange {
  CUdeviceptr devPtr;
  size_t count;
  CUdevice device;
  struct MemRange *successor, *precursor;
};

static struct MemRange *list_head = NULL;
static size_t list_size = 0;
static pthread_mutex_t g_map_mutex = PTHREAD_MUTEX_INITIALIZER;

static void activate_rate_watcher();
static void *rate_watcher(void *);
static void rate_limiter(const long long);
static void activate_limit_manager();
static void *limit_manager(void *);
static void init_rate_limit(long long, volatile long long *, int *);
static void *memory_transfer_routine(CUdevice device);

static void initialization();

static const char *cuda_error(CUresult, const char **);

/*
 * memory transfer
 */

void init_list() {
  pthread_mutex_lock(&g_map_mutex);
  list_head = (struct MemRange*)malloc(sizeof(struct MemRange));
  list_head->count = 0;
  list_head->devPtr = -1;
  list_head->device = 0;
  list_head->precursor = NULL;
  list_head->successor = NULL;
  pthread_mutex_unlock(&g_map_mutex);
  LOGGER(4, "list_head: %p\n", list_head);
}


void list_insert(struct MemRange *pos, struct MemRange *item) {
  item->successor = pos->successor;
  item->precursor = pos;
  if (pos->successor)
    pos->successor->precursor = item;
  pos->successor = item;
  ++list_size;
}


void list_delete(struct MemRange *item) {
  if (item->precursor)
    item->precursor->successor = item->successor;
  if (item->successor)
    item->successor->precursor = item->precursor;
  item->precursor = NULL;
  item->successor = NULL;
  --list_size;
}


void allocate_mem(CUdeviceptr devPtr, size_t count, CUdevice device) {
  struct MemRange *item = (struct MemRange *)malloc(sizeof(struct MemRange));
  item->devPtr = devPtr;
  item->count = count;
  item->device = device;
  item->precursor = item->successor = NULL;
  
  if (list_head == NULL)
    init_list();

  pthread_mutex_lock(&g_map_mutex);
  g_used_memory += count;
  list_insert(list_head, item);
  pthread_mutex_unlock(&g_map_mutex);
}


void delete_mem(CUdeviceptr devPtr) {
  if (list_head == NULL)
    init_list();

  int ptr_find = 0;
  
  pthread_mutex_lock(&g_map_mutex);

  for (struct MemRange *it = list_head->successor; it; it = it->successor) {
    if (it->devPtr == devPtr) {
      ptr_find = 1;
      g_used_memory -= it->count;
      list_delete(it);
      break;
    }
  }
  pthread_mutex_unlock(&g_map_mutex);
  
  if (ptr_find == 0) {
  }
}


const char *cuda_error(CUresult code, const char **p) {
  CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, code, p);
  return *p;
}

static ssize_t rio_readn(int fd, void *usrbuf, size_t n) {
  size_t nleft = n;
  ssize_t nread;
  char *bufp = usrbuf;

  while (nleft > 0) {
	  if ((nread = read(fd, bufp, nleft)) < 0) {
	    if (errno == EINTR)
		    nread = 0;
	    else
		    return -1;
	  } 
	  else if (nread == 0)
	    break;
	  nleft -= nread;
	  bufp += nread;
  }
    return (n - nleft);
}


static int open_listenfd(CUdevice device) {
  char SOCKET_PATH[108];
  const int LISTENQ = 8;
  struct sockaddr_un name;
  int ret;
  int listenfd;

  /* Create local socket. */

  listenfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listenfd == -1) {
    fprintf(stderr, "socket failed: %s\n", strerror(errno));
  }

  /*
  * For portability clear the whole structure, since some
  * implementations have additional (nonstandard) fields in
  * the structure.
  */

  memset(&name, 0, sizeof(name));

  /* Bind socket to socket name. */

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

  name.sun_family = AF_UNIX;
  strncpy(name.sun_path, SOCKET_PATH, sizeof(name.sun_path));

  ret = unlink(SOCKET_PATH);
  if (ret == -1) {
    if (!access(SOCKET_PATH, F_OK))
      fprintf(stderr, "unlink failed: %s\n", strerror(errno));
  }

  ret = bind(listenfd, (const struct sockaddr *) &name, sizeof(name));
  if (ret == -1) {
    fprintf(stderr, "bind failed: %s\n", strerror(errno));
  }

  /*
  * Prepare for accepting connections. The backlog size is set
  * to LISTENQ. So while one request is being processed other requests
  * can be waiting.
  */

  ret = listen(listenfd, LISTENQ);
  if (ret == -1) {
    fprintf(stderr, "listen failed: %s\n", strerror(errno));
  }

  return listenfd;
}

static void init_rate_limit(long long initial_value, volatile long long *p_rate_limit, int *p_state) {
  *p_rate_limit = initial_value;
  *p_state = TGS_SLOW_START;
}

static inline long long min(long long a, long long b) {
  return a < b ? a : b;
}

static inline long long max(long long a, long long b) {
  return a > b ? a : b;
}

static const long long update_rate_limit(int *p_state, CUdevice device, double recv_rate, double max_rate, double *p_max_rate) {
  const static long long UPPER_LIMIT = 100000000000000LL;
  const double threshold = 0.03;
  static int sign = 0;
  double delta = (recv_rate - max_rate) / max_rate;
  delta = delta > 0 ? delta : -delta;

  long long rate_limit = g_rate_limit[device];

  switch (*p_state)
  {
  case TGS_SLOW_START:
    if (delta <= threshold) {
      rate_limit = min(rate_limit * 1.5 + 1, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)(1ll << 40))));
    }
    else {
      rate_limit = rate_limit / 1.5;
      sign = -1;
      *p_state = TGS_CONGESTION_AVOIDANCE;
    }
    break;

  case TGS_CONGESTION_AVOIDANCE:
    if ((sign == -1 && delta <= threshold) || (sign == 1 && delta < threshold)) {
      rate_limit += max(max_rate * 0.00025, RATE_MIN);
      rate_limit = min(rate_limit, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL)));
      sign = 1;
    }
    else {
      rate_limit -= max(rate_limit * 0.08, RATE_MIN);
      rate_limit = min(rate_limit, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL)));
      sign = -1;
    }

    if (delta >= 3. * threshold) {
      *p_state = TGS_SLOW_START;
      rate_limit /= 10;
    }
    break;
  }

  static int max_diff_counter = 0;
  if (delta >= 0.12) {
    ++max_diff_counter;
    if (max_diff_counter >= 20) {
      *p_max_rate *= 0.8;
      max_diff_counter = 0;
      *p_state = TGS_SLOW_START;
      rate_limit /= 2;
    }
  }
  else {
    max_diff_counter = 0;
  }

  rate_limit = (rate_limit <= 0) ? 0 : rate_limit;

  g_rate_limit[device] = rate_limit;
  return rate_limit;
}


static void *memory_transfer_routine(CUdevice device) {
  if (list_head == NULL)
    init_list();
  CUresult ret;
  const char *cuda_err_string = NULL;

  CUcontext cuContext;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRetain, &cuContext, device);
  if (unlikely(ret)) {
    LOGGER(FATAL, "cuDevicePrimaryCtxRetain error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }
  
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetCurrent, cuContext);
  if (unlikely(ret)) {
    LOGGER(FATAL, "cuCtxSetCurrent error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }

  pthread_mutex_lock(&g_map_mutex);

  for (struct MemRange *it = list_head->successor; it; it = it->successor) 
    if (it->device == device) {
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemPrefetchAsync, it->devPtr, it->count, it->device, NULL);
      if (ret != CUDA_SUCCESS) {
        const char *error_name = NULL, *error_reason = NULL;
        CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorName, ret, &error_name);
        CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, ret, &error_reason);
        continue;
      }
    }

  pthread_mutex_unlock(&g_map_mutex);

  return NULL;
}


void activate_memory_transfer_routine(CUdevice device) {
  memory_transfer_routine(device);
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

static void *limit_manager(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const int MAXLINE = 4096;
  const static long long UPPER_LIMIT = 100000000000000LL;
  const double alpha = 0.0;
  int listenfd, connfd, ret;
  socklen_t clientlen;
  struct sockaddr_storage clientaddr;
  char client_hostname[MAXLINE], client_port[MAXLINE];

  listenfd = open_listenfd(device);
  clientlen = sizeof(struct sockaddr_storage);

  
  while (1) {
    if ((connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &clientlen)) < 0)
      LOGGER(FATAL, "accept error\n");
    if ((ret = getnameinfo((const struct sockaddr *)&clientaddr, clientlen, client_hostname, MAXLINE,
                           client_port, MAXLINE, 0)) != 0)
      LOGGER(FATAL, "getnameinfo error: %s\n", gai_strerror(ret));
    
    double max_rate = -1;
    if (rio_readn(connfd, (void *)&max_rate, sizeof(double)) != sizeof(double)) {
      continue;
    }

    g_rate_limit[device] = 0;
    g_rate_control_flag[device] = 1;

    double recv_rate = 1.;
    long long cnt = 0;
    int state = -1;

    const int WINDOW_SIZE = 5;
    const int PREWARM_TIME = 0;
    const int PROFILE_TIME = 5;
    double rate_window[WINDOW_SIZE];
    int continue_flag = 0;

profile:
    for (int t = 1; t <= PREWARM_TIME + PROFILE_TIME || continue_flag; ++t) {
      double recv_counter = -1;
      ssize_t n = rio_readn(connfd, (void *)&recv_counter, sizeof(double));
      if (n != sizeof(double)) {
        break;
      }
      if (t <= PREWARM_TIME) {
        continue;
      }

      recv_counter = recv_counter >= 1. ? recv_counter : 1.;
      recv_rate = alpha * recv_rate + (1 - alpha) * recv_counter;
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, recv_rate);
      double max_delta = (max_window_rate - max_rate) / max_rate;
      
      if (max_delta >= -0.05 && max_delta <= 0.05) {
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
        continue_flag = (abs(max_rate - max_window_rate) < 1e-5);
      }
      else {
        if (max_delta > 0.05) {
          double new_max_rate = max_window_rate * 0.975;
          max_rate = max_rate > new_max_rate ? max_rate : new_max_rate;
        }
        else {
          double new_max_rate = max_window_rate * 1.025;
          max_rate = max_rate < new_max_rate ? max_rate : new_max_rate;
        }
        continue_flag = 1;
      }
    }
    fprintf(stderr, "profile max rate: %lld\n", max_rate);

    while (1) {
      double recv_counter = -1;
      ssize_t n = rio_readn(connfd, (void *)&recv_counter, sizeof(double));
      if (n != sizeof(double)) {
        if (n)
          LOGGER(4, "readn error: receive %d byte\n", (int)n);
        break;
      }

      recv_counter = recv_counter >= 1. ? recv_counter : 1.;
      recv_rate = alpha * recv_rate + (1 - alpha) * recv_counter;
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, recv_rate);
      double max_delta = (max_window_rate - max_rate) / max_rate;
      
      if (max_delta >= -0.1 && max_delta <= 0.1)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else if (max_delta > 0.2 || max_delta < -0.2) {
        if (max_delta > 0.2) {
          double new_max_rate = max_window_rate * 0.975;
          max_rate = max_rate > new_max_rate ? max_rate : new_max_rate;
        }
        else {
          double new_max_rate = max_window_rate * 1.025;
          max_rate = max_rate < new_max_rate ? max_rate : new_max_rate;
        }
        fprintf(stderr, "change max rate: %lf\n", max_rate);
        goto profile;
      }

      ++cnt;
      if (cnt == 1) {
        init_rate_limit(LIMIT_INITIALIZER, &g_rate_limit[device], &state);
        continue;
      }

      int num_zero = 0;
      for(int i = 0; i < WINDOW_SIZE; ++i){
        if(rate_window[i] < 1000)
          ++num_zero;
      }
      
      long long rate_limit;
      if(num_zero <= WINDOW_SIZE / 5 * 2 || cnt < 15){
        if(num_zero == 2 && cnt > 15){
          rate_limit = LIMIT_INITIALIZER;
          init_rate_limit(rate_limit, &g_rate_limit[device], &state);
        }
        else
          rate_limit = update_rate_limit(&state, device, recv_rate, (double)max_rate, &max_rate);
      }
      else{
        rate_limit = min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL));
        init_rate_limit(rate_limit, &g_rate_limit[device], &state);
      }

    }
    if ((ret = close(connfd)) < 0)
      LOGGER(FATAL, "close error\n");

    g_rate_limit[device] = 0;
    activate_memory_transfer_routine(device);
    g_rate_control_flag[device] = 0;
  }
}

static int tgs_set_cpu_affinity(pthread_t thread_id, int core_id) {
    int ret;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    ret = pthread_setaffinity_np(thread_id, sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "failed to set cpu affinity, core_id=%d", core_id);
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

static void activate_limit_manager(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, limit_manager, (void *)(uintptr_t)device);
  tgs_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("limit_manager");
#else
  pthread_setname_np(tid, "limit_manager");
#endif
}


static inline int launch_test(const long long kernel_size, const CUdevice device) {
  return g_rate_control_flag[device] == 1 && g_rate_counter[device] > g_rate_limit[device];
}


static inline void rate_limiter(const long long kernel_size) {
  CUdevice device = 0;
  const CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxGetDevice error\n");
  }

  if (!g_active_gpu[device])
    initialization(device);

  while (launch_test(kernel_size, device))
    nanosleep(&g_cycle, NULL);
  __sync_add_and_fetch_8(&g_rate_counter[device], kernel_size);
}


static void *rate_watcher(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 50;
  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  g_rate_counter[device] = 0;
  while (1) {
    nanosleep(&unit_time, NULL);
    
    long long current_rate = g_rate_counter[device];
    g_rate_counter[device] = 0;
    g_current_rate[device] = current_rate;
  }
  return NULL;
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


static inline void initialization(const CUdevice device) {
  g_active_gpu[device] = 1;

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

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxResetPersistingL2Cache);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxResetPersistingL2Cache error\n");
  }
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetLimit, CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxSetLimit error, ret=%d\n", (int)ret);
  }

  activate_rate_watcher(device);
  activate_limit_manager(device);
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

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, flag);
  return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        flags);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }
  
  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);
  
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }
  
  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);
  

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

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
  

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }
  
  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);
  

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes) {
  *pPitch = ROUND_UP(WidthInBytes, 128);
  size_t bytesize = ROUND_UP(*pPitch * Height, ElementSizeBytes);
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }
  
  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);
  

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}


CUresult cuMemFree_v2(CUdeviceptr dptr) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree_v2, dptr);
  if (ret == CUDA_SUCCESS)
    delete_mem(dptr);
  return ret;
}


CUresult cuMemFree(CUdeviceptr dptr) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree, dptr);
  if (ret == CUDA_SUCCESS)
    delete_mem(dptr); 
  return ret;
}


CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArrayCreate_v2");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArrayCreate");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArray3DCreate_v2");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArray3DCreate");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
  CUresult ret;

  LOGGER(FATAL, "call cuMipmappedArrayCreate");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMipmappedArrayCreate, pHandle,
                        pMipmappedArrayDesc, numMipmapLevels);
  return ret;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes = (*bytes > g_spare_memory) ? (*bytes - g_spare_memory) : 0;
  return ret;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes = (*bytes > g_spare_memory) ? (*bytes - g_spare_memory) : 0;
  return ret;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*total > g_spare_memory + g_used_memory) ? (*total - g_spare_memory - g_used_memory) : 0;
  return ret;
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*total > g_spare_memory + g_used_memory) ? (*total - g_spare_memory - g_used_memory) : 0;
  return ret;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  int leastPriority, greatestPriority;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetStreamPriorityRange, &leastPriority, &greatestPriority);
  if (ret == CUDA_SUCCESS) {
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreateWithPriority, phStream, Flags, leastPriority);
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
    greatestPriority = (leastPriority + greatestPriority) / 2;
    if (priority < greatestPriority) {
      priority = greatestPriority;
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
  rate_limiter(gridDimX * gridDimY * gridDimZ);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel_ptsz, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
  rate_limiter(gridDimX * gridDimY * gridDimZ);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunch(CUfunction f) {
  rate_limiter(1);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  rate_limiter(gridDimX * gridDimY * gridDimZ);
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
  rate_limiter(gridDimX * gridDimY * gridDimZ);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  rate_limiter(grid_width * grid_height);
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f, grid_width,
                         grid_height);
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  rate_limiter(grid_width * grid_height);
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
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy_v2, ctx);
}

CUresult cuCtxDestroy(CUcontext ctx) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy, ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent_v2, pctx);
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
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
  return CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRelease, dev);
}
