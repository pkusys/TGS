
#ifndef HIJACK_LIBRARY_H
#define HIJACK_LIBRARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nvml-subset.h"

/**
 * Proc file path for driver version
 */
#define DRIVER_VERSION_PROC_PATH "/proc/driver/nvidia/version"

/**
 * Driver regular expression pattern
 */
#define DRIVER_VERSION_MATCH_PATTERN "([0-9]+)(\\.[0-9]+)+"

/**
 * Max sample pid size
 */
#define MAX_PIDS (1024)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define ROUND_UP(n, base) ((n) % (base) ? (n) + (base) - (n) % (base) : (n))

#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2 * !!(condition)]))

#define CAS(ptr, old, new) __sync_bool_compare_and_swap_8((ptr), (old), (new))
#define UNUSED __attribute__((unused))

#define MILLISEC (1000UL * 1000UL)

#define TIME_TICK (10)
#define FACTOR (32)
#define MAX_UTILIZATION (100)
#define CHANGE_LIMIT_INTERVAL (30)
#define USAGE_THRESHOLD (5)

#define GET_VALID_VALUE(x) (((x) >= 0 && (x) <= 100) ? (x) : 0)
#define CODEC_NORMALIZE(x) (x * 85 / 100)

typedef struct {
  void *fn_ptr;
  char *name;
} entry_t;

typedef struct {
  int major;
  int minor;
} __attribute__((packed, aligned(8))) version_t;

typedef enum {
  INFO = 0,
  ERROR = 1,
  WARNING = 2,
  FATAL = 3,
  VERBOSE = 4,
} log_level_enum_t;

#define LOGGER(level, format, ...)                              \
  ({                                                            \
    char *_print_level_str = getenv("LOGGER_LEVEL");            \
    int _print_level = 3;                                       \
    if (_print_level_str) {                                     \
      _print_level = (int)strtoul(_print_level_str, NULL, 10);  \
      _print_level = _print_level < 0 ? 3 : _print_level;       \
    }                                                           \
    FILE *fptr; \
    fptr = fopen("/tmp/cudalog","a");\
    fprintf(fptr, "%s:%d " format "\n", __FILE__, __LINE__, \
              ##__VA_ARGS__);                                   \
    fclose(fptr);\
    if (level == FATAL) {                                       \
      exit(-1);                                                 \
    }                                                           \
  })

/**
 * Load library and initialize some data
 */
void load_necessary_data();

#ifdef __cplusplus
}
#endif

#endif
