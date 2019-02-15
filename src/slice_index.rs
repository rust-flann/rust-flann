use itertools::{IntoChunks, Itertools};
use raw;
use Indexable;
use Parameters;

const DEFAULT_REBUILD_THRESHOLD: f32 = 2.0;

#[derive(Copy, Clone, Debug, Fail)]
pub enum FlannError {
    #[fail(
        display = "expected {} dimensions in point, but got {} dimensions",
        expected, got
    )]
    InvalidPointDimensionality { expected: usize, got: usize },
    #[fail(
        display = "expected number divisible by {}, but got {}, which is not",
        expected, got
    )]
    InvalidFlatPointsLen { expected: usize, got: usize },
    #[fail(display = "FLANN failed to build index")]
    FailedToBuildIndex,
    #[fail(display = "input must have at least one point")]
    ZeroInputPoints,
}

#[derive(Copy, Clone, Debug)]
pub struct Neighbor<D> {
    pub index: usize,
    pub distance_squared: D,
}

pub struct SliceIndex<'a, T: Indexable> {
    index: raw::flann_index_t,
    parameters: raw::FLANNParameters,
    pub(crate) point_len: usize,
    _phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T: Indexable> Drop for SliceIndex<'a, T> {
    fn drop(&mut self) {
        unsafe {
            T::free_index(self.index, &mut self.parameters);
        }
    }
}

impl<'a, T: Indexable> SliceIndex<'a, T> {
    /// Makes a new index from points that are already in a slice of memory
    /// in component order where there are `point_len` components.
    ///
    /// This borrows the slice internally in FLANN.
    pub fn new(
        point_len: usize,
        points: &'a [T],
        parameters: Parameters,
    ) -> Result<Self, FlannError> {
        if points.is_empty() {
            return Err(FlannError::ZeroInputPoints);
        }
        if points.len() % point_len != 0 {
            return Err(FlannError::InvalidFlatPointsLen {
                expected: point_len,
                got: points.len(),
            });
        }
        // This stores how much faster FLANN executed compared to linear, which we discard.
        let mut speedup = 0.0;
        let mut flann_params = parameters.into();
        let index = unsafe {
            T::build_index(
                points.as_ptr() as *mut T,
                (points.len() / point_len) as i32,
                point_len as i32,
                &mut speedup,
                &mut flann_params,
            )
        };
        if index.is_null() {
            return Err(FlannError::FailedToBuildIndex);
        }
        Ok(Self {
            index,
            parameters: flann_params,
            point_len,
            _phantom: Default::default(),
        })
    }

    /// Adds a point to the index.
    ///
    /// To prevent the index from becoming unbalanced, it rebuilds after adding
    /// `rebuild_theshold` points. This defaults to `2.0`.
    pub fn add_slice(
        &mut self,
        point: &'a [T],
        rebuild_threshold: impl Into<Option<f32>>,
    ) -> Result<(), FlannError> {
        if point.len() != self.point_len {
            return Err(FlannError::InvalidPointDimensionality {
                expected: self.point_len,
                got: point.len(),
            });
        }
        let retval = unsafe {
            T::add_points(
                self.index,
                point.as_ptr() as *mut T,
                1,
                self.point_len as i32,
                rebuild_threshold
                    .into()
                    .unwrap_or(DEFAULT_REBUILD_THRESHOLD),
            )
        };
        assert_eq!(retval, 0);
        Ok(())
    }

    /// Adds multiple points to the index.
    ///
    /// To prevent the index from becoming unbalanced, it rebuilds after adding
    /// `rebuild_theshold` points. This defaults to `2.0`.
    pub fn add_many_slices(
        &mut self,
        points: &'a [T],
        rebuild_threshold: impl Into<Option<f32>>,
    ) -> Result<(), FlannError> {
        // Don't run FLANN if we add no points.
        if points.is_empty() {
            return Ok(());
        }
        if points.len() % self.point_len != 0 {
            return Err(FlannError::InvalidFlatPointsLen {
                expected: self.point_len,
                got: points.len(),
            });
        }
        let retval = unsafe {
            T::add_points(
                self.index,
                points.as_ptr() as *mut T,
                (points.len() / self.point_len) as i32,
                self.point_len as i32,
                rebuild_threshold
                    .into()
                    .unwrap_or(DEFAULT_REBUILD_THRESHOLD),
            )
        };
        assert_eq!(retval, 0);
        Ok(())
    }

    /// Get the point that corresponds to this index.
    pub fn get(&self, idx: usize) -> Option<&'a [T]> {
        if idx < self.len() {
            let point = unsafe { T::get_point(self.index, idx as u32) };
            assert!(!point.is_null());
            Some(unsafe { std::slice::from_raw_parts(point, self.point_len) })
        } else {
            None
        }
    }

    /// Returns `true` if the point was successfully removed.
    pub fn remove(&mut self, idx: usize) -> bool {
        if idx < self.len() {
            let retval = unsafe { T::remove_point(self.index, idx as u32) };
            assert_eq!(retval, 0);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        unsafe { T::size(self.index) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Performs a search to find only the closest neighbor.
    pub fn find_nearest_neighbor(
        &mut self,
        point: &[T],
    ) -> Result<Neighbor<T::ResultType>, FlannError> {
        if point.len() != self.point_len {
            return Err(FlannError::InvalidPointDimensionality {
                expected: self.point_len,
                got: point.len(),
            });
        }
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
        Ok(Neighbor {
            index: index as usize,
            distance_squared,
        })
    }

    /// Performs k-NN search for `num` neighbors.
    /// If there are less points in the set than `num` it returns that many neighbors.
    pub fn find_nearest_neighbors(
        &mut self,
        num: usize,
        point: &[T],
    ) -> Result<impl Iterator<Item = Neighbor<T::ResultType>>, FlannError> {
        if point.len() != self.point_len {
            return Err(FlannError::InvalidPointDimensionality {
                expected: self.point_len,
                got: point.len(),
            });
        }
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
        Ok(indices.into_iter().zip(distances_squared.into_iter()).map(
            |(index, distance_squared)| Neighbor {
                index: index as usize,
                distance_squared,
            },
        ))
    }

    /// Performs k-NN search for `num` neighbors, limiting the search to `radius` distance.
    /// If there are less points in the set than `num` it returns that many neighbors.
    ///
    /// The returned iterator is sorted by closest to furthest.
    pub fn find_nearest_neighbors_radius(
        &mut self,
        num: usize,
        radius: f32,
        point: &[T],
    ) -> Result<impl Iterator<Item = Neighbor<T::ResultType>>, FlannError> {
        if point.len() != self.point_len {
            return Err(FlannError::InvalidPointDimensionality {
                expected: self.point_len,
                got: point.len(),
            });
        }
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
                radius,
                &mut self.parameters,
            )
        };
        assert!(retval >= 0);
        Ok(indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .take(retval as usize)
            .map(|(index, distance_squared)| Neighbor {
                index: index as usize,
                distance_squared,
            }))
    }

    /// Performs k-NN search for `num` neighbors for several points.
    ///
    /// If there are less points in the set than `num` it returns that many
    /// neighbors for each point.
    ///
    /// The returned iterator contains each point's matches in
    /// `min(num, self.len())` sized chunks. If you want to iterate over the
    /// matches in a logical way you will need to use `.chunks()` from
    /// itertools or collect the neighbors into a `Vec` and then use
    /// `.chunks_exact()`. This will be corrected in a future release.
    pub fn find_many_nearest_neighbors<I, P>(
        &mut self,
        num: usize,
        points: I,
    ) -> Result<IntoChunks<impl Iterator<Item = Neighbor<T::ResultType>>>, FlannError>
    where
        I: IntoIterator<Item = P>,
        P: IntoIterator<Item = T>,
    {
        let mut points_vec = Vec::new();
        for point in points {
            let count = point.into_iter().map(|d| points_vec.push(d)).count();
            if count != self.point_len {
                return Err(FlannError::InvalidPointDimensionality {
                    expected: self.point_len,
                    got: count,
                });
            }
        }
        self.find_many_nearest_neighbors_flat(num, &points_vec)
    }

    /// Performs k-NN search on `num` neighbors for several points.
    ///
    /// If there are less points in the set than `num` it returns that many
    /// neighbors for each point.
    ///
    /// The returned iterator contains each point's matches in
    /// `min(num, self.len())` sized chunks. If you want to iterate over the
    /// matches in a logical way you will need to use `.chunks()` from
    /// itertools or collect the neighbors into a `Vec` and then use
    /// `.chunks_exact()`. This may be corrected in a future release.
    ///
    /// This assumes points are already in a slice of memory
    /// in component order where there are `point_len` components
    /// (as specified in `new` or `new_flat`).
    pub fn find_many_nearest_neighbors_flat(
        &mut self,
        num: usize,
        points: &[T],
    ) -> Result<IntoChunks<impl Iterator<Item = Neighbor<T::ResultType>>>, FlannError> {
        let neighbor_from_index_distance = |(index, distance_squared)| Neighbor {
            index: index as usize,
            distance_squared,
        };
        if points.is_empty() {
            let indices: Vec<i32> = Vec::new();
            let distances: Vec<T::ResultType> = Vec::new();
            return Ok(indices
                .into_iter()
                .zip(distances.into_iter())
                .map(neighbor_from_index_distance)
                .chunks(num));
        }
        if points.len() % self.point_len != 0 {
            return Err(FlannError::InvalidFlatPointsLen {
                expected: self.point_len,
                got: points.len(),
            });
        }
        let num = num.min(self.len());
        let total_points = points.len() / self.point_len;
        let mut indices: Vec<i32> = vec![-1; num * total_points];
        let mut distances_squared: Vec<T::ResultType> =
            vec![T::ResultType::default(); num * total_points];
        let retval = unsafe {
            T::find_nearest_neighbors_index(
                self.index,
                points.as_ptr() as *mut T,
                total_points as i32,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
                num as i32,
                &mut self.parameters,
            )
        };
        assert_eq!(retval, 0);
        Ok(indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .map(neighbor_from_index_distance)
            .chunks(num))
    }
}
