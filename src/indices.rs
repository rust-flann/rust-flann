use Indexable;
use raw::{self, flann_index_t, FLANNParameters};
use std::os::raw::{c_int, c_uint};

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
                raw::$build_index(
                    dataset,
                    rows,
                    cols,
                    speedup,
                    flann_params,
                )
            }

            #[inline]
            unsafe fn add_points(
                index_ptr: flann_index_t,
                points: *mut Self,
                rows: c_int,
                columns: c_int,
                rebuild_threshold: f32,
                flann_params: *mut FLANNParameters,
            ) -> c_int {
                raw::$add_points(
                    index_ptr,
                    points,
                    rows,
                    columns,
                    rebuild_threshold,
                    flann_params,
                )
            }

            #[inline]
            unsafe fn remove_point(
                index_ptr: flann_index_t,
                point_id: c_uint,
                flann_params: *mut FLANNParameters,
            ) -> c_int {
                raw::$remove_point(index_ptr, point_id, flann_params)
            }

            #[inline]
            unsafe fn get_point(
                index_ptr: flann_index_t,
                point_id: c_uint,
                flann_params: *mut FLANNParameters,
            ) -> *mut Self {
                raw::$get_point(index_ptr, point_id, flann_params)
            }

            #[inline]
            unsafe fn veclen(
                index_ptr: flann_index_t,
                flann_params: *mut FLANNParameters,
            ) -> c_uint {
                raw::$veclen(index_ptr, flann_params)
            }

            #[inline]
            unsafe fn size(
                index_ptr: flann_index_t,
                flann_params: *mut FLANNParameters,
            ) -> c_uint {
                raw::$size(index_ptr, flann_params)
            }

            #[inline]
            unsafe fn used_memory(
                index_ptr: flann_index_t,
                flann_params: *mut FLANNParameters,
            ) -> c_int {
                raw::$used_memory(index_ptr, flann_params)
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
    }
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
    i8,
    f32,
    flann_build_index_char,
    flann_add_points_char,
    flann_remove_point_char,
    flann_get_point_char,
    flann_veclen_char,
    flann_size_char,
    flann_used_memory_char,
    flann_find_nearest_neighbors_index_char,
    flann_radius_search_char,
    flann_free_index_char,
);

impl_index!(
    u8,
    f32,
    flann_build_index_uint8_t,
    flann_add_points_uint8_t,
    flann_remove_point_uint8_t,
    flann_get_point_uint8_t,
    flann_veclen_uint8_t,
    flann_size_uint8_t,
    flann_used_memory_uint8_t,
    flann_find_nearest_neighbors_index_uint8_t,
    flann_radius_search_uint8_t,
    flann_free_index_uint8_t,
);

impl_index!(
    i16,
    f32,
    flann_build_index_int16_t,
    flann_add_points_int16_t,
    flann_remove_point_int16_t,
    flann_get_point_int16_t,
    flann_veclen_int16_t,
    flann_size_int16_t,
    flann_used_memory_int16_t,
    flann_find_nearest_neighbors_index_int16_t,
    flann_radius_search_int16_t,
    flann_free_index_int16_t,
);

impl_index!(
    u16,
    f32,
    flann_build_index_uint16_t,
    flann_add_points_uint16_t,
    flann_remove_point_uint16_t,
    flann_get_point_uint16_t,
    flann_veclen_uint16_t,
    flann_size_uint16_t,
    flann_used_memory_uint16_t,
    flann_find_nearest_neighbors_index_uint16_t,
    flann_radius_search_uint16_t,
    flann_free_index_uint16_t,
);

impl_index!(
    i32,
    f32,
    flann_build_index_int32_t,
    flann_add_points_int32_t,
    flann_remove_point_int32_t,
    flann_get_point_int32_t,
    flann_veclen_int32_t,
    flann_size_int32_t,
    flann_used_memory_int32_t,
    flann_find_nearest_neighbors_index_int32_t,
    flann_radius_search_int32_t,
    flann_free_index_int32_t,
);

impl_index!(
    u32,
    f32,
    flann_build_index_uint32_t,
    flann_add_points_uint32_t,
    flann_remove_point_uint32_t,
    flann_get_point_uint32_t,
    flann_veclen_uint32_t,
    flann_size_uint32_t,
    flann_used_memory_uint32_t,
    flann_find_nearest_neighbors_index_uint32_t,
    flann_radius_search_uint32_t,
    flann_free_index_uint32_t,
);
