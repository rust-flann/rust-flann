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
        let mut data_raw = dataset
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect::<Vec<T>>();
        let mut speedup = 0.0;
        let mut flann_params = parameters.into();
        let index = unsafe {
            T::build_index(
                data_raw.as_mut_ptr(),
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
            index,
            parameters: Mutex::new(flann_params),
            _phantom: PhantomData,
        })
    }

    pub fn add(&mut self, point: &Datum<T, N>, rebuild_threshold: Option<f32>) {
        let mut data_raw = point.iter().cloned().collect::<Vec<T>>();
        let retval = unsafe {
            T::add_points(
                self.index,
                data_raw.as_mut_ptr(),
                1,
                N::to_i32(),
                rebuild_threshold.unwrap_or(2.0),
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        assert_eq!(retval, 0);
    }

    pub fn add_multiple(&mut self, points: &[Datum<T, N>], rebuild_threshold: Option<f32>) {
        let mut data_raw = points
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect::<Vec<T>>();
        let retval = unsafe {
            T::add_points(
                self.index,
                data_raw.as_mut_ptr(),
                points.len() as i32,
                N::to_i32(),
                rebuild_threshold.unwrap_or(2.0),
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        assert_eq!(retval, 0);
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        unsafe {
            T::get_point(
                self.index,
                idx as u32,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            ).as_ref()
        }
    }

    pub fn remove(&self, idx: usize) {
        unsafe {
            T::get_point(
                self.index,
                idx as u32,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            );
        }
    }

    pub fn len(&self) -> usize {
        let l = unsafe {
            T::veclen(
                self.index,
                self.parameters.lock().expect(LOCK_FAIL).deref_mut(),
            )
        };
        l as usize
    }
}

static LOCK_FAIL: &'static str = "Failed to acquire lock on parameters field";
