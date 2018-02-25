use Indexable;
use itertools::Itertools;
use Parameters;
use raw;

type Datum<T> = Vec<T>;

pub struct VecIndex<T: Indexable> {
    index: raw::flann_index_t,
    point_memory: Vec<Vec<T>>,
    points: Vec<Datum<T>>,
    parameters: raw::FLANNParameters,
    datum_length: usize,
    datum_length_i32: i32,
}

impl<T: Indexable> Drop for VecIndex<T> {
    fn drop(&mut self) {
        unsafe {
            T::free_index(self.index, &mut self.parameters);
        }
    }
}

impl<T: Indexable> VecIndex<T> {
    pub fn new(
        datum_length: usize,
        points: Vec<Datum<T>>,
        parameters: Parameters,
    ) -> Result<Self, &'static str> {
        if points.is_empty() {
            return Err("Initial set of points cannot be empty");
        }
        if points.iter().any(|datum| datum.len() != datum_length) {
            return Err("All points need to have the desired number of coordinates");
        }
        let mut point_memory = Vec::new();
        point_memory.push(
            points
                .iter()
                .flat_map(|v| v.iter().cloned())
                .collect::<Vec<T>>(),
        );
        let datum_length_i32 = datum_length as i32;
        let mut speedup = 0.0;
        let mut flann_params = parameters.into();
        let index = unsafe {
            T::build_index(
                point_memory.last_mut().unwrap().as_mut_ptr(),
                points.len() as i32,
                datum_length_i32,
                &mut speedup,
                &mut flann_params,
            )
        };
        if index.is_null() {
            return Err("FLANN's C bindings failed to initialize index");
        }
        Ok(Self {
            point_memory,
            points: points,
            index,
            parameters: flann_params,
            datum_length,
            datum_length_i32,
        })
    }

    pub fn add(
        &mut self,
        point: Datum<T>,
        rebuild_threshold: Option<f32>,
    ) -> Result<(), &'static str> {
        if point.len() != self.datum_length {
            return Err("Entry doesn't have the desired number of coordinates");
        }
        self.point_memory.push(point.iter().cloned().collect());
        self.points.push(point);
        let retval = unsafe {
            T::add_points(
                self.index,
                self.point_memory.last_mut().unwrap().as_mut_ptr(),
                1,
                self.datum_length_i32,
                rebuild_threshold.unwrap_or(2.0),
                &self.parameters,
            )
        };
        assert_eq!(retval, 0);
        Ok(())
    }

    pub fn add_multiple(
        &mut self,
        mut points: Vec<Datum<T>>,
        rebuild_threshold: Option<f32>,
    ) -> Result<(), &'static str> {
        if points.is_empty() {
            return Ok(());
        }
        if points.iter().any(|datum| datum.len() != self.datum_length) {
            return Err("All points need to have the desired number of coordinates");
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
                self.datum_length_i32,
                rebuild_threshold.unwrap_or(2.0),
                &self.parameters,
            )
        };
        assert_eq!(retval, 0);
        Ok(())
    }

    pub fn get(&self, idx: usize) -> Option<&Datum<T>> {
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

    pub fn find_nearest_neighbor(
        &self,
        point: &Datum<T>,
    ) -> Result<(usize, T::ResultType), &'static str> {
        if point.len() != self.datum_length {
            return Err("Entry doesn't have the desired number of coordinates");
        }
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
        Ok((index as usize, dist))
    }

    pub fn find_nearest_neighbors(
        &self,
        points: &Vec<Datum<T>>,
        mut num: usize,
    ) -> Result<Vec<Vec<(usize, T::ResultType)>>, &'static str> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        if points.iter().any(|datum| datum.len() != self.datum_length) {
            return Err("All points need to have the desired number of coordinates");
        }
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
        Ok(izip!(index.into_iter().map(|v| v as usize), dist)
            .chunks(num)
            .into_iter()
            .map(Iterator::collect)
            .collect())
    }

    pub fn search_radius(
        &self,
        point: &Datum<T>,
        radius: f32,
        max_nn: usize,
    ) -> Result<Vec<(usize, T::ResultType)>, &'static str> {
        if point.len() != self.datum_length {
            return Err("Entry doesn't have the desired number of coordinates");
        }
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
        Ok(indices
            .into_iter()
            .map(|v| v as usize)
            .zip(dists.into_iter())
            .take(retval as usize)
            .collect())
    }
}
