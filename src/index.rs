use generic_array::{ArrayLength, GenericArray};
use Indexable;
use itertools::Itertools;
use Parameters;
use raw;
use std::marker::PhantomData;

type Datum<T, N> = GenericArray<T, N>;

pub struct Index<T: Indexable, N: ArrayLength<T>> {
    index: raw::flann_index_t,
    point_memory: Vec<Vec<T>>,
    points: Vec<Datum<T, N>>,
    parameters: raw::FLANNParameters,
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
    pub fn new(points: Vec<Datum<T, N>>, parameters: Parameters) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let mut point_memory = Vec::new();
        point_memory.push(
            points
                .iter()
                .flat_map(|v| v.iter().cloned())
                .collect::<Vec<T>>(),
        );
        let mut speedup = 0.0;
        let mut flann_params = parameters.into();
        let index = unsafe {
            T::build_index(
                point_memory.last_mut().unwrap().as_mut_ptr(),
                points.len() as i32,
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
            points: points,
            index,
            parameters: flann_params,
            _phantom: PhantomData,
        })
    }

    pub fn add(&mut self, point: Datum<T, N>, rebuild_threshold: Option<f32>) {
        self.point_memory.push(point.iter().cloned().collect());
        self.points.push(point);
        let retval = unsafe {
            T::add_points(
                self.index,
                self.point_memory.last_mut().unwrap().as_mut_ptr(),
                1,
                N::to_i32(),
                rebuild_threshold.unwrap_or(2.0),
                &self.parameters,
            )
        };
        assert_eq!(retval, 0);
    }

    pub fn add_multiple(&mut self, mut points: Vec<Datum<T, N>>, rebuild_threshold: Option<f32>) {
        if points.is_empty() {
            return;
        }
        self.point_memory
            .push(points.iter().flat_map(|v| v.iter().cloned()).collect());
        let l = points.len() as i32;
        self.points.append(&mut points);
        let retval = unsafe {
            T::add_points(
                self.index,
                self.point_memory.last_mut().unwrap().as_mut_ptr(),
                l,
                N::to_i32(),
                rebuild_threshold.unwrap_or(2.0),
                &self.parameters,
            )
        };
        assert_eq!(retval, 0);
    }

    pub fn get(&self, idx: usize) -> Option<&Datum<T, N>> {
        self.points.get(idx)
    }

    pub fn remove(&mut self, idx: usize) {
        self.points.remove(idx);
        unsafe {
            T::remove_point(self.index, idx as u32, &self.parameters);
        }
    }

    pub fn count(&self) -> usize {
        self.points.len()
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
                &self.parameters,
            )
        };
        assert_eq!(retval, 0);
        (index as usize, dist)
    }

    pub fn find_nearest_neighbors(
        &self,
        points: &Vec<Datum<T, N>>,
        mut num: usize,
    ) -> Vec<Vec<(usize, T::ResultType)>> {
        num = num.min(self.count());
        let mut data_raw = points
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect::<Vec<T>>();
        let mut index: Vec<i32> = vec![0; points.len() * num];
        let mut dist: Vec<T::ResultType> = vec![T::ResultType::default(); points.len() * num];
        let retval = unsafe {
            T::find_nearest_neighbors_index(
                self.index,
                data_raw.as_mut_ptr(),
                points.len() as i32,
                index.as_mut_ptr(),
                dist.as_mut_ptr(),
                num as i32,
                &self.parameters,
            )
        };
        assert_eq!(retval, 0);
        izip!(index.into_iter().map(|v| v as usize), dist)
            .chunks(num)
            .into_iter()
            .map(Iterator::collect)
            .collect()
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
                &self.parameters,
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
