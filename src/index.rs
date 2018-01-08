use generic_array::{ArrayLength, GenericArray};
use Indexable;
use Parameters;
use raw;
use std::marker::PhantomData;
use std::ops::DerefMut;
use std::sync::Mutex;

type Datum<T, N> = GenericArray<T, N>;

pub struct Index<T: Indexable, N: ArrayLength<T>> {
    index: raw::flann_index_t,
    point_memory: Vec<Vec<T>>,
    parameters: Mutex<raw::FLANNParameters>,
    _phantom: PhantomData<(T, N)>,
}

impl<T: Indexable, N: ArrayLength<T>> Drop for Index<T, N> {
    fn drop(&mut self) {
        unsafe {
            T::free_index(
                self.index,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            );
        }
    }
}

impl<T: Indexable, N: ArrayLength<T>> Index<T, N> {
    pub fn new(dataset: &[Datum<T, N>], parameters: Parameters) -> Option<Self> {
        if dataset.is_empty() {
            return None;
        }
        let mut point_memory = Vec::new();
        point_memory.push(
            dataset
                .iter()
                .flat_map(|v| v.iter().cloned())
                .collect::<Vec<T>>(),
        );
        let mut speedup = 0.0;
        let mut flann_params = parameters.into();
        let index = unsafe {
            T::build_index(
                point_memory.last_mut().unwrap().as_mut_ptr(),
                dataset.len() as i32,
                N::to_i32(),
                &mut speedup,
                &mut flann_params,
            )
        };
        if index.is_null() {
            return None;
        }
        Some(Self {
            point_memory,
            index,
            parameters: Mutex::new(flann_params),
            _phantom: PhantomData,
        })
    }

    pub fn add(&mut self, point: &Datum<T, N>, rebuild_threshold: Option<f32>) {
        self.point_memory.push(point.iter().cloned().collect());
        let retval = unsafe {
            T::add_points(
                self.index,
                self.point_memory.last_mut().unwrap().as_mut_ptr(),
                1,
                N::to_i32(),
                rebuild_threshold.unwrap_or(2.0),
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        assert_eq!(retval, 0);
    }

    pub fn add_multiple(&mut self, points: &[Datum<T, N>], rebuild_threshold: Option<f32>) {
        if points.is_empty() {
            return;
        }
        self.point_memory
            .push(points.iter().flat_map(|v| v.iter().cloned()).collect());
        let retval = unsafe {
            T::add_points(
                self.index,
                self.point_memory.last_mut().unwrap().as_mut_ptr(),
                points.len() as i32,
                N::to_i32(),
                rebuild_threshold.unwrap_or(2.0),
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        assert_eq!(retval, 0);
    }

    pub fn get(&self, idx: usize) -> Option<Datum<T, N>> {
        if idx >= self.count() {
            return None;
        }
        let mut point = vec![T::default(); N::to_usize()];
        let retval = unsafe {
            T::get_point(
                self.index,
                idx as u32,
                point.as_mut_ptr(),
                N::to_i32(),
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        if retval == 0 {
            Some(Datum::<T, N>::clone_from_slice(&point))
        } else {
            None
        }
    }

    pub fn remove(&self, idx: usize) {
        unsafe {
            T::remove_point(
                self.index,
                idx as u32,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            );
        }
    }

    pub fn count(&self) -> usize {
        let l = unsafe {
            T::size(
                self.index,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        l as usize
    }

    pub fn find_nearest_neighbor(&self, point: &Datum<T, N>) -> (usize, T::ResultType) {
        let mut data_raw = point.iter().cloned().collect::<Vec<T>>();
        let mut index = 0;
        let mut dist = T::ResultType::default();
        let retval = unsafe {
            T::find_nearest_neighbors_index(
                self.index,
                data_raw.as_mut_ptr(),
                1,
                &mut index,
                &mut dist,
                1,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        assert_eq!(retval, 0);
        (index as usize, dist)
    }

    pub fn search_radius(
        &self,
        point: &Datum<T, N>,
        radius: f32,
        max_nn: usize,
    ) -> Vec<(usize, T::ResultType)> {
        let mut data_raw = point.iter().cloned().collect::<Vec<T>>();
        let mut indices = vec![0; max_nn];
        let mut dists = vec![T::ResultType::default(); max_nn];
        let retval = unsafe {
            T::radius_search(
                self.index,
                data_raw.as_mut_ptr(),
                indices.as_mut_ptr(),
                dists.as_mut_ptr(),
                max_nn as i32,
                radius,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        assert!(retval >= 0);
        indices
            .into_iter()
            .map(|v| v as usize)
            .zip(dists.into_iter())
            .take(retval as usize)
            .collect()
    }
}

static LOCK_FAIL: &'static str = "Failed to acquire lock on parameters field";
