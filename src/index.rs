use generic_array::{ArrayLength, GenericArray};
use Indexable;
use Parameters;
use raw;
use std::marker::PhantomData;

type Datum<T, N> = GenericArray<T, N>;

pub struct Index<T: Indexable, N: ArrayLength<T>> {
    index: raw::flann_index_t,
    flann_parameters: raw::FLANNParameters,
    _phantom: PhantomData<(T, N)>,
}

impl<T: Indexable, N: ArrayLength<T>> Drop for Index<T, N> {
    fn drop(&mut self) {
        unsafe {
            T::free_index(self.index, &mut self.flann_parameters);
        }
    }
}

impl<T: Indexable, N: ArrayLength<T>> Index<T, N> {
    pub fn new(dataset: &[Datum<T, N>], parameters: Parameters) -> Self {
        let mut flann_parameters = parameters.into();
        let mut data_raw = dataset
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect::<Vec<T>>();
        let mut speedup = 0.0;
        let index = unsafe {
            T::build_index(
                data_raw.as_mut_ptr(),
                N::to_i32(),
                dataset.len() as i32,
                &mut speedup,
                &mut flann_parameters,
            )
        };
        Self {
            index,
            flann_parameters,
            _phantom: PhantomData,
        }
    }
}
