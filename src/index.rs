use generic_array::{ArrayLength, GenericArray};
use itertools::{IntoChunks, Itertools};
use raw;
use std::marker::PhantomData;
use FlannError;
use Indexable;
use Neighbor;
use Parameters;

pub struct Index<T: Indexable, N: ArrayLength<T>> {
    index: raw::flann_index_t,
    storage: Vec<Vec<T>>,
    parameters: raw::FLANNParameters,
    rebuild_threshold: f32,
    _phantom: PhantomData<(T, N)>,
}

impl<T: Indexable, N: ArrayLength<T>> Drop for Index<T, N> {
    fn drop(&mut self) {
        unsafe {
            T::free_index(self.index, &mut self.parameters);
        }
    }
}

impl<T: Indexable, N: ArrayLength<T>> Index<T, N> {
    pub fn new<I>(points: I, parameters: Parameters) -> Result<Self, FlannError>
    where
        I: IntoIterator<Item = GenericArray<T, N>>,
    {
        let points_vec: Vec<T> = points.into_iter().flat_map(|p| p.into_iter()).collect();
        if points_vec.is_empty() {
            return Err(FlannError::ZeroInputPoints);
        }
        let mut speedup = 0.0;
        let rebuild_threshold = parameters.rebuild_threshold;
        let mut flann_params = parameters.into();
        let index = unsafe {
            T::build_index(
                points_vec.as_ptr() as *mut T,
                (points_vec.len() / N::to_usize()) as i32,
                N::to_i32(),
                &mut speedup,
                &mut flann_params,
            )
        };
        if index.is_null() {
            return Err(FlannError::FailedToBuildIndex);
        }
        Ok(Self {
            index,
            storage: vec![points_vec],
            parameters: flann_params,
            rebuild_threshold,
            _phantom: PhantomData,
        })
    }

    /// Adds a point to the index.
    pub fn add(&mut self, point: GenericArray<T, N>) {
        let points_vec: Vec<T> = point.into_iter().collect();
        let retval = unsafe {
            T::add_points(
                self.index,
                points_vec.as_ptr() as *mut T,
                1,
                N::to_i32(),
                self.rebuild_threshold,
            )
        };
        self.storage.push(points_vec);
        assert_eq!(retval, 0);
    }

    /// Adds multiple points to the index.
    pub fn add_multiple<I>(&mut self, points: I)
    where
        I: IntoIterator<Item = GenericArray<T, N>>,
    {
        let points_vec: Vec<T> = points.into_iter().flat_map(|p| p.into_iter()).collect();
        if points_vec.is_empty() {
            return;
        }
        let retval = unsafe {
            T::add_points(
                self.index,
                points_vec.as_ptr() as *mut T,
                (points_vec.len() / N::to_usize()) as i32,
                N::to_i32(),
                self.rebuild_threshold,
            )
        };
        self.storage.push(points_vec);
        assert_eq!(retval, 0);
    }

    /// Get the point that corresponds to this index `idx`.
    pub fn get(&self, idx: usize) -> Option<&GenericArray<T, N>> {
        if idx < self.len() {
            let point = unsafe { T::get_point(self.index, idx as u32) };
            assert!(!point.is_null());
            Some(unsafe { &*(point as *const GenericArray<T, N>) })
        } else {
            None
        }
    }

    /// Removes a point at index `idx`.
    pub fn remove(&mut self, idx: usize) {
        let retval = unsafe { T::remove_point(self.index, idx as u32) };
        assert_eq!(retval, 0);
    }

    pub fn len(&self) -> usize {
        unsafe { T::size(self.index) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Performs a search to find only the closest neighbor.
    pub fn find_nearest_neighbor(&mut self, point: &GenericArray<T, N>) -> Neighbor<T::ResultType> {
        let mut index = -1;
        let mut distance_squared = T::ResultType::default();
        let retval = unsafe {
            T::find_nearest_neighbors_index(
                self.index,
                point.as_ptr() as *mut T,
                1,
                &mut index,
                &mut distance_squared,
                1,
                &mut self.parameters,
            )
        };
        assert_eq!(retval, 0);
        Neighbor {
            index: index as usize,
            distance_squared,
        }
    }

    /// Performs k-NN search for `num` neighbors.
    /// If there are less points in the set than `num` it returns that many neighbors.
    pub fn find_nearest_neighbors(
        &mut self,
        num: usize,
        point: &GenericArray<T, N>,
    ) -> impl Iterator<Item = Neighbor<T::ResultType>> {
        let num = num.min(self.len());
        let mut indices: Vec<i32> = vec![-1; num];
        let mut distances_squared: Vec<T::ResultType> = vec![T::ResultType::default(); num];
        let retval = unsafe {
            T::find_nearest_neighbors_index(
                self.index,
                point.as_ptr() as *mut T,
                1,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
                num as i32,
                &mut self.parameters,
            )
        };
        assert_eq!(retval, 0);
        indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .map(|(index, distance_squared)| Neighbor {
                index: index as usize,
                distance_squared,
            })
    }

    /// Performs k-NN search for `num` neighbors.
    /// If there are less points in the set than `num` it returns that many neighbors.
    pub fn find_nearest_neighbors_radius(
        &mut self,
        num: usize,
        radius_squared: f32,
        point: &GenericArray<T, N>,
    ) -> impl Iterator<Item = Neighbor<T::ResultType>> {
        let num = num.min(self.len());
        let mut indices: Vec<i32> = vec![-1; num];
        let mut distances_squared: Vec<T::ResultType> = vec![T::ResultType::default(); num];
        let retval = unsafe {
            T::radius_search(
                self.index,
                point.as_ptr() as *mut T,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
                num as i32,
                radius_squared,
                &mut self.parameters,
            )
        };
        assert!(retval >= 0);
        indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .take(retval as usize)
            .map(|(index, distance_squared)| Neighbor {
                index: index as usize,
                distance_squared,
            })
    }

    /// Performs k-NN search on `num` neighbors for several points.
    ///
    /// If there are less points in the set than `num` it returns that many
    /// neighbors for each point.
    pub fn find_many_nearest_neighbors(
        &mut self,
        num: usize,
        points: &[GenericArray<T, N>],
    ) -> IntoChunks<impl Iterator<Item = Neighbor<T::ResultType>>> {
        let neighbor_from_index_distance = |(index, distance_squared)| Neighbor {
            index: index as usize,
            distance_squared,
        };
        if points.is_empty() {
            let indices: Vec<i32> = Vec::new();
            let distances: Vec<T::ResultType> = Vec::new();
            return indices
                .into_iter()
                .zip(distances.into_iter())
                .map(neighbor_from_index_distance)
                .chunks(num);
        }
        let num = num.min(self.len());
        let mut indices: Vec<i32> = vec![-1; num * points.len()];
        let mut distances_squared: Vec<T::ResultType> =
            vec![T::ResultType::default(); num * points.len()];
        let retval = unsafe {
            T::find_nearest_neighbors_index(
                self.index,
                points.as_ptr() as *mut T,
                points.len() as i32,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
                num as i32,
                &mut self.parameters,
            )
        };
        assert_eq!(retval, 0);
        indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .map(neighbor_from_index_distance)
            .chunks(num)
    }
}
