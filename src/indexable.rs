use raw::{flann_index_t, FLANNParameters};
use std::os::raw::{c_int, c_uint};

pub unsafe trait Indexable {
    type ResultType;

    unsafe fn build_index(
        dataset: *mut Self,
        rows: c_int,
        cols: c_int,
        speedup: *mut f32,
        flann_params: *mut FLANNParameters,
    ) -> flann_index_t;

    unsafe fn add_points(
        index_ptr: flann_index_t,
        points: *mut Self,
        rows: c_int,
        columns: c_int,
        rebuild_threshold: f32,
        flann_params: *mut FLANNParameters,
    ) -> c_int;

    unsafe fn remove_point(
        index_ptr: flann_index_t,
        point_id: c_uint,
        flann_params: *mut FLANNParameters,
    ) -> c_int;

    unsafe fn get_point(
        index_ptr: flann_index_t,
        point_id: c_uint,
        flann_params: *mut FLANNParameters,
    ) -> *mut Self;

    unsafe fn veclen(index_ptr: flann_index_t, flann_params: *mut FLANNParameters) -> c_uint;

    unsafe fn size(index_ptr: flann_index_t, flann_params: *mut FLANNParameters) -> c_uint;

    unsafe fn used_memory(index_ptr: flann_index_t, flann_params: *mut FLANNParameters) -> c_int;

    unsafe fn find_nearest_neighbors_index(
        index_id: flann_index_t,
        testset: *mut Self,
        trows: c_int,
        indices: *mut c_int,
        dists: *mut Self::ResultType,
        nn: c_int,
        flann_params: *mut FLANNParameters,
    ) -> c_int;

    unsafe fn radius_search(
        index_ptr: flann_index_t,
        query: *mut Self,
        indices: *mut c_int,
        dists: *mut Self::ResultType,
        max_nn: c_int,
        radius: f32,
        flann_params: *mut FLANNParameters,
    ) -> c_int;

    unsafe fn free_index(index_id: flann_index_t, flann_params: *mut FLANNParameters) -> c_int;
}
