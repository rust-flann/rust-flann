#[deny(warnings)]
extern crate generic_array;
pub extern crate flann_sys as raw;

mod enums;
mod index;
mod indexable;
mod indices;
mod parameters;

pub use enums::{Algorithm, CentersInit, Checks, DistanceType, LogLevel};
pub use generic_array::typenum;
pub use index::Index;
pub use indexable::Indexable;
pub use parameters::Parameters;
