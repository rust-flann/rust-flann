#![deny(warnings)]

#[allow(unknown_lints, useless_attribute, unused_imports)]
#[macro_use]
extern crate generic_array;
#[macro_use]
extern crate itertools;
pub extern crate flann_sys as raw;

mod enums;
mod index;
mod indexable;
mod indices;
mod parameters;
mod vec_index;
#[cfg(test)]
mod tests;

pub use enums::{Algorithm, CentersInit, Checks, DistanceType, LogLevel};
pub use generic_array::typenum;
pub use index::Index;
pub use vec_index::VecIndex;
pub use indexable::Indexable;
pub use parameters::Parameters;
