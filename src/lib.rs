#![deny(warnings)]

#[allow(unknown_lints, unused_imports)]
#[macro_use]
extern crate generic_array;
#[cfg_attr(test, macro_use)]
extern crate itertools;
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;
#[macro_use]
extern crate failure;
pub extern crate flann_sys as raw;

mod enums;
mod index;
mod indexable;
mod indices;
mod parameters;
mod slice_index;
#[cfg(test)]
mod tests;
mod vec_index;

pub use enums::{Algorithm, CentersInit, Checks, DistanceType, LogLevel};
pub use generic_array::typenum;
pub use index::Index;
pub use indexable::Indexable;
pub use parameters::Parameters;
pub use slice_index::SliceIndex;
pub use vec_index::VecIndex;

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
