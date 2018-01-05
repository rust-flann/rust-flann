#[deny(warnings)]
pub extern crate flann_sys as raw;

macro_rules! convertable_enum {
    ($name: ident; $type: ty; $($key: ident = $value: path,)*) => {
        #[derive(Clone, Copy, Debug)]
        pub enum $name {
            $($key,)*
        }

        #[allow(unreachable_patterns)]
        impl $name {
            pub fn from_raw(v: $type) -> Option<$name> {
                match v {
                    $($value => Some($name::$key),)*
                    _ => None
                }
            }

            pub fn as_raw(&self) -> $type {
                match *self {
                    $($name::$key => $value,)*
                }
            }
        }
    }
}

convertable_enum!(Algorithm; u32;
    Linear = raw::flann_algorithm_t_FLANN_INDEX_LINEAR,
    KDTree = raw::flann_algorithm_t_FLANN_INDEX_KDTREE,
    KMeans = raw::flann_algorithm_t_FLANN_INDEX_KMEANS,
    Composite = raw::flann_algorithm_t_FLANN_INDEX_COMPOSITE,
    KDTreeSingle = raw::flann_algorithm_t_FLANN_INDEX_KDTREE_SINGLE,
    Hierarchical = raw::flann_algorithm_t_FLANN_INDEX_HIERARCHICAL,
    Lsh = raw::flann_algorithm_t_FLANN_INDEX_LSH,
    Saved = raw::flann_algorithm_t_FLANN_INDEX_SAVED,
    Autotuned = raw::flann_algorithm_t_FLANN_INDEX_AUTOTUNED,
);

convertable_enum!(CentersInit; u32;
    Random = raw::flann_centers_init_t_FLANN_CENTERS_RANDOM,
    Gonzales = raw::flann_centers_init_t_FLANN_CENTERS_GONZALES,
    KMeansPP = raw::flann_centers_init_t_FLANN_CENTERS_KMEANSPP,
);

convertable_enum!(LogLevel; u32;
    None = raw::flann_log_level_t_FLANN_LOG_NONE,
    Fatal = raw::flann_log_level_t_FLANN_LOG_FATAL,
    Error = raw::flann_log_level_t_FLANN_LOG_ERROR,
    Warn = raw::flann_log_level_t_FLANN_LOG_WARN,
    Info = raw::flann_log_level_t_FLANN_LOG_INFO,
    Debug = raw::flann_log_level_t_FLANN_LOG_DEBUG,
);

convertable_enum!(DistanceType; u32;
    Euclidean = raw::flann_distance_t_FLANN_DIST_EUCLIDEAN,
    L2 = raw::flann_distance_t_FLANN_DIST_L2,
    Manhattan = raw::flann_distance_t_FLANN_DIST_MANHATTAN,
    L1 = raw::flann_distance_t_FLANN_DIST_L1,
    Minkowski = raw::flann_distance_t_FLANN_DIST_MINKOWSKI,
    Max = raw::flann_distance_t_FLANN_DIST_MAX,
    HistIntersect = raw::flann_distance_t_FLANN_DIST_HIST_INTERSECT,
    Hellinger = raw::flann_distance_t_FLANN_DIST_HELLINGER,
    ChiSquare = raw::flann_distance_t_FLANN_DIST_CHI_SQUARE,
    KullbackLeibler = raw::flann_distance_t_FLANN_DIST_KULLBACK_LEIBLER,
    Hamming = raw::flann_distance_t_FLANN_DIST_HAMMING,
    HammingLut = raw::flann_distance_t_FLANN_DIST_HAMMING_LUT,
    HammingPopcnt = raw::flann_distance_t_FLANN_DIST_HAMMING_POPCNT,
    L2Simple = raw::flann_distance_t_FLANN_DIST_L2_SIMPLE,
);

#[derive(Clone, Copy, Debug)]
pub enum Checks {
    Unlimited,
    Autotuned,
    Exact(i32),
}

impl Checks {
    pub fn from_raw(v: i32) -> Checks {
        match v {
            raw::flann_checks_t_FLANN_CHECKS_UNLIMITED => Checks::Unlimited,
            raw::flann_checks_t_FLANN_CHECKS_AUTOTUNED => Checks::Autotuned,
            v => Checks::Exact(v),
        }
    }

    pub fn as_raw(&self) -> i32 {
        match *self {
            Checks::Unlimited => raw::flann_checks_t_FLANN_CHECKS_UNLIMITED,
            Checks::Autotuned => raw::flann_checks_t_FLANN_CHECKS_AUTOTUNED,
            Checks::Exact(v) => v,
        }
    }
}

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
    pub random_seed: i64,
    pub distance_type: DistanceType,
    pub distance_order: i32,
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
            centers_init: CentersInit::from_raw(v.centers_init).ok_or_else(|| {
                format!("Illegal centers init enum value: {}", v.centers_init)
            })?,
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
            distance_type: DistanceType::from_raw(v.distance_type).ok_or_else(|| {
                format!("Illegal distanc etype enum value: {}", v.log_level)
            })?,
            distance_order: v.distance_order,
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
            distance_type: self.distance_type.as_raw(),
            distance_order: self.distance_order,
        }
    }
}

impl Into<raw::FLANNParameters> for Parameters {
    fn into(self) -> raw::FLANNParameters {
        (&self).into()
    }
}
