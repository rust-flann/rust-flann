use enums::{Algorithm, CentersInit, Checks, LogLevel};
use raw;
use std::os::raw::c_long;

#[derive(Debug, Clone)]
pub struct Parameters {
    pub algorithm: Algorithm,
    pub checks: Checks,
    pub eps: f32,
    pub sorted: i32,
    pub max_neighbors: i32,
    pub cores: i32,
    pub trees: i32,
    pub leaf_max_size: i32,
    pub branching: i32,
    pub iterations: i32,
    pub centers_init: CentersInit,
    pub cb_index: f32,
    pub target_precision: f32,
    pub build_weight: f32,
    pub memory_weight: f32,
    pub sample_fraction: f32,
    pub table_number: u32,
    pub key_size: u32,
    pub multi_probe_level: u32,
    pub log_level: LogLevel,
    pub random_seed: c_long,
}

impl Default for Parameters {
    fn default() -> Parameters {
        Parameters::from_raw(unsafe { raw::DEFAULT_FLANN_PARAMETERS })
            .expect("Illegal default FLANN parameters in C bindings")
    }
}

impl Parameters {
    pub fn from_raw(v: raw::FLANNParameters) -> Result<Parameters, String> {
        Ok(Parameters {
            algorithm: Algorithm::from_raw(v.algorithm)
                .ok_or_else(|| format!("Illegal algorithm enum value: {}", v.algorithm))?,
            checks: Checks::from_raw(v.checks),
            eps: v.eps,
            sorted: v.sorted,
            max_neighbors: v.max_neighbors,
            cores: v.cores,
            trees: v.trees,
            leaf_max_size: v.leaf_max_size,
            branching: v.branching,
            iterations: v.iterations,
            centers_init: CentersInit::from_raw(v.centers_init)
                .ok_or_else(|| format!("Illegal centers init enum value: {}", v.centers_init))?,
            cb_index: v.cb_index,
            target_precision: v.target_precision,
            build_weight: v.build_weight,
            memory_weight: v.memory_weight,
            sample_fraction: v.sample_fraction,
            table_number: v.table_number_,
            key_size: v.key_size_,
            multi_probe_level: v.multi_probe_level_,
            log_level: LogLevel::from_raw(v.log_level)
                .ok_or_else(|| format!("Illegal log level enum value: {}", v.log_level))?,
            random_seed: v.random_seed,
        })
    }
}

impl<'a> Into<raw::FLANNParameters> for &'a Parameters {
    fn into(self) -> raw::FLANNParameters {
        raw::FLANNParameters {
            algorithm: self.algorithm.as_raw(),
            checks: self.checks.as_raw(),
            eps: self.eps,
            sorted: self.sorted,
            max_neighbors: self.max_neighbors,
            cores: self.cores,
            trees: self.trees,
            leaf_max_size: self.leaf_max_size,
            branching: self.branching,
            iterations: self.iterations,
            centers_init: self.centers_init.as_raw(),
            cb_index: self.cb_index,
            target_precision: self.target_precision,
            build_weight: self.build_weight,
            memory_weight: self.memory_weight,
            sample_fraction: self.sample_fraction,
            table_number_: self.table_number,
            key_size_: self.key_size,
            multi_probe_level_: self.multi_probe_level,
            log_level: self.log_level.as_raw(),
            random_seed: self.random_seed,
        }
    }
}

impl Into<raw::FLANNParameters> for Parameters {
    fn into(self) -> raw::FLANNParameters {
        (&self).into()
    }
}
