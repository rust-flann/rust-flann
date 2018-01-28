#ifndef WRAPPER_H
#define WRAPPER_H

#include <stdint.h>
#include "defines.h"

#ifdef __cplusplus
extern "C" {
using namespace flann;
#endif

struct FLANNParameters {
  enum flann_algorithm_t algorithm;

  int checks;
  float eps;
  int sorted;
  int max_neighbors;
  int cores;

  int trees;
  int leaf_max_size;

  int branching;
  int iterations;
  enum flann_centers_init_t centers_init;
  float cb_index;

  float target_precision;
  float build_weight;
  float memory_weight;
  float sample_fraction;

  unsigned int table_number_;
  unsigned int key_size_;
  unsigned int multi_probe_level_;

  enum flann_distance_t distance_type;
  int distance_order;

  enum flann_log_level_t log_level;
  long random_seed;
};

typedef void* flann_index_t;

FLANN_EXPORT extern struct FLANNParameters DEFAULT_FLANN_PARAMETERS;

#define FLANN_BUILD_INDEX(T, R)                       \
  FLANN_EXPORT flann_index_t flann_build_index_##T(   \
      T* dataset, int rows, int cols, float* speedup, \
      struct FLANNParameters* flann_params);

#define FLANN_ADD_POINTS(T, R)                                   \
  FLANN_EXPORT int flann_add_points_##T(                         \
      flann_index_t index_ptr, T* points, int rows, int columns, \
      float rebuild_threshold, const struct FLANNParameters* flann_params);

#define FLANN_REMOVE_POINT(T, R)                      \
  FLANN_EXPORT int flann_remove_point_##T(            \
      flann_index_t index_ptr, unsigned int point_id, \
      const struct FLANNParameters* flann_params);

#define FLANN_GET_POINT(T, R)                                                \
  FLANN_EXPORT int flann_get_point_##T(                                      \
      flann_index_t index_ptr, unsigned int point_id, T* point, int columns, \
      const struct FLANNParameters* flann_params);

#define FLANN_VECLEN(T, R)                    \
  FLANN_EXPORT unsigned int flann_veclen_##T( \
      flann_index_t index_ptr, const struct FLANNParameters* flann_params);

#define FLANN_SIZE(T, R)                    \
  FLANN_EXPORT unsigned int flann_size_##T( \
      flann_index_t index_ptr, const struct FLANNParameters* flann_params);

#define FLANN_USED_MEMORY(T, R)           \
  FLANN_EXPORT int flann_used_memory_##T( \
      flann_index_t index_ptr, const struct FLANNParameters* flann_params);

#define FLANN_SAVE_INDEX(T, R)                \
  FLANN_EXPORT int flann_save_index_##T(      \
      flann_index_t index_id, char* filename, \
      const struct FLANNParameters* flann_params);

#define FLANN_LOAD_INDEX(T, R)                        \
  FLANN_EXPORT flann_index_t flann_load_index_##T(    \
      char* filename, T* dataset, int rows, int cols, \
      const struct FLANNParameters* flann_params);

#define FLANN_FIND_NEAREST_NEIGHBORS(T, R)                                 \
  FLANN_EXPORT int flann_find_nearest_neighbors_##T(                       \
      T* dataset, int rows, int cols, T* testset, int trows, int* indices, \
      R* dists, int nn, const struct FLANNParameters* flann_params);

#define FLANN_FIND_NEAREST_NEIGHBORS_INDEX(T, R)                             \
  FLANN_EXPORT int flann_find_nearest_neighbors_index_##T(                   \
      flann_index_t index_id, T* testset, int trows, int* indices, R* dists, \
      int nn, const struct FLANNParameters* flann_params);

#define FLANN_RADIUS_SEARCH(T, R)                                            \
  FLANN_EXPORT int flann_radius_search_##T(                                  \
      flann_index_t index_ptr, T* query, int* indices, R* dists, int max_nn, \
      float radius, const struct FLANNParameters* flann_params);

#define FLANN_FREE_INDEX(T, R)           \
  FLANN_EXPORT int flann_free_index_##T( \
      flann_index_t index_id, const struct FLANNParameters* flann_params);

#define FLANN_COMPUTE_CLUSTER_CENTERS(T, R)                    \
  FLANN_EXPORT int flann_compute_cluster_centers_##T(          \
      T* dataset, int rows, int cols, int clusters, R* result, \
      const struct FLANNParameters* flann_params);

#define FLANN_IMPL(T, R)                   \
  FLANN_BUILD_INDEX(T, R)                  \
  FLANN_ADD_POINTS(T, R)                   \
  FLANN_REMOVE_POINT(T, R)                 \
  FLANN_GET_POINT(T, R)                    \
  FLANN_VECLEN(T, R)                       \
  FLANN_SIZE(T, R)                         \
  FLANN_USED_MEMORY(T, R)                  \
  FLANN_SAVE_INDEX(T, R)                   \
  FLANN_LOAD_INDEX(T, R)                   \
  FLANN_FIND_NEAREST_NEIGHBORS(T, R)       \
  FLANN_FIND_NEAREST_NEIGHBORS_INDEX(T, R) \
  FLANN_RADIUS_SEARCH(T, R)                \
  FLANN_FREE_INDEX(T, R)                   \
  FLANN_COMPUTE_CLUSTER_CENTERS(T, R)

FLANN_IMPL(float, float)
FLANN_IMPL(double, double)
FLANN_IMPL(char, float)
FLANN_IMPL(int16_t, float)
FLANN_IMPL(int32_t, float)
FLANN_IMPL(uint8_t, float)
FLANN_IMPL(uint16_t, float)
FLANN_IMPL(uint32_t, float)

#undef FLANN_BUILD_INDEX
#undef FLANN_ADD_POINTS
#undef FLANN_REMOVE_POINT
#undef FLANN_GET_POINT
#undef FLANN_VECLEN
#undef FLANN_SIZE
#undef FLANN_USED_MEMORY
#undef FLANN_SAVE_INDEX
#undef FLANN_LOAD_INDEX
#undef FLANN_FIND_NEAREST_NEIGHBORS
#undef FLANN_FIND_NEAREST_NEIGHBORS_INDEX
#undef FLANN_RADIUS_SEARCH
#undef FLANN_FREE_INDEX
#undef FLANN_COMPUTE_CLUSTER_CENTERS
#undef FLANN_IMPL

#ifdef __cplusplus
}

#include <flann/flann.hpp>

#endif

#endif  // WRAPPER_H
