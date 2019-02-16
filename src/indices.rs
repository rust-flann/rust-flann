use raw::{self, flann_index_t, FLANNParameters};
use std::os::raw::{c_int, c_uint};
use Indexable;

macro_rules! impl_index {
    (
        $t: ty,
        $r: ty,
        $build_index: ident,
        $add_points: ident,
        $remove_point: ident,
        $get_point: ident,
        $veclen: ident,
        $size: ident,
        $used_memory: ident,
        $find_nearest_neighbors_index: ident,
        $radius_search: ident,
        $free_index: ident,
    ) => {
        unsafe impl Indexable for $t {
            type ResultType = $r;

            #[inline]
            unsafe fn build_index(
                dataset: *mut Self,
                rows: c_int,
                cols: c_int,
                speedup: *mut f32,
                flann_params: *mut FLANNParameters,
            ) -> flann_index_t {
                raw::$build_index(dataset, rows, cols, speedup, flann_params)
            }

            #[inline]
            unsafe fn add_points(
                index_ptr: flann_index_t,
                points: *mut Self,
                rows: c_int,
                columns: c_int,
                rebuild_threshold: f32,
            ) -> c_int {
                raw::$add_points(index_ptr, points, rows, columns, rebuild_threshold)
            }

            #[inline]
            unsafe fn remove_point(index_ptr: flann_index_t, point_id: c_uint) -> c_int {
                raw::$remove_point(index_ptr, point_id)
            }

            #[inline]
            unsafe fn get_point(index_ptr: flann_index_t, point_id: c_uint) -> *mut Self {
                raw::$get_point(index_ptr, point_id)
            }

            #[inline]
            unsafe fn veclen(index_ptr: flann_index_t) -> c_uint {
                raw::$veclen(index_ptr)
            }

            #[inline]
            unsafe fn size(index_ptr: flann_index_t) -> c_uint {
                raw::$size(index_ptr)
            }

            #[inline]
            unsafe fn used_memory(index_ptr: flann_index_t) -> c_int {
                raw::$used_memory(index_ptr)
            }

            #[inline]
            unsafe fn find_nearest_neighbors_index(
                index_id: flann_index_t,
                testset: *mut Self,
                trows: c_int,
                indices: *mut c_int,
                dists: *mut Self::ResultType,
                nn: c_int,
                flann_params: *mut FLANNParameters,
            ) -> c_int {
                raw::$find_nearest_neighbors_index(
                    index_id,
                    testset,
                    trows,
                    indices,
                    dists,
                    nn,
                    flann_params,
                )
            }

            #[inline]
            unsafe fn radius_search(
                index_ptr: flann_index_t,
                query: *mut Self,
                indices: *mut c_int,
                dists: *mut Self::ResultType,
                max_nn: c_int,
                radius: f32,
                flann_params: *mut FLANNParameters,
            ) -> c_int {
                raw::$radius_search(
                    index_ptr,
                    query,
                    indices,
                    dists,
                    max_nn,
                    radius,
                    flann_params,
                )
            }

            #[inline]
            unsafe fn free_index(
                index_id: flann_index_t,
                flann_params: *mut FLANNParameters,
            ) -> c_int {
                raw::$free_index(index_id, flann_params)
            }
        }
    };
}

impl_index!(
    f32,
    f32,
    flann_build_index_float,
    flann_add_points_float,
    flann_remove_point_float,
    flann_get_point_float,
    flann_veclen_float,
    flann_size_float,
    flann_used_memory_float,
    flann_find_nearest_neighbors_index_float,
    flann_radius_search_float,
    flann_free_index_float,
);

impl_index!(
    f64,
    f64,
    flann_build_index_double,
    flann_add_points_double,
    flann_remove_point_double,
    flann_get_point_double,
    flann_veclen_double,
    flann_size_double,
    flann_used_memory_double,
    flann_find_nearest_neighbors_index_double,
    flann_radius_search_double,
    flann_free_index_double,
);

impl_index!(
    u8,
    f32,
    flann_build_index_byte,
    flann_add_points_byte,
    flann_remove_point_byte,
    flann_get_point_byte,
    flann_veclen_byte,
    flann_size_byte,
    flann_used_memory_byte,
    flann_find_nearest_neighbors_index_byte,
    flann_radius_search_byte,
    flann_free_index_byte,
);

impl_index!(
    i32,
    f32,
    flann_build_index_int,
    flann_add_points_int,
    flann_remove_point_int,
    flann_get_point_int,
    flann_veclen_int,
    flann_size_int,
    flann_used_memory_int,
    flann_find_nearest_neighbors_index_int,
    flann_radius_search_int,
    flann_free_index_int,
);
