#![deny(warnings)]

#[allow(unknown_lints, unused_imports)]
#[macro_use]
extern crate generic_array;
#[macro_use]
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
#[cfg(test)]
mod tests;
mod vec_index;

pub use enums::{Algorithm, CentersInit, Checks, DistanceType, LogLevel};
pub use generic_array::typenum;
pub use index::Index;
pub use indexable::Indexable;
pub use parameters::Parameters;
pub use vec_index::VecIndex;
