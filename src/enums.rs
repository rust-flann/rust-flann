use raw;

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

convertable_enum!(Algorithm; raw::flann_algorithm_t;
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

convertable_enum!(CentersInit; raw::flann_centers_init_t;
    Random = raw::flann_centers_init_t_FLANN_CENTERS_RANDOM,
    Gonzales = raw::flann_centers_init_t_FLANN_CENTERS_GONZALES,
    KMeansPP = raw::flann_centers_init_t_FLANN_CENTERS_KMEANSPP,
);

convertable_enum!(LogLevel; raw::flann_log_level_t;
    None = raw::flann_log_level_t_FLANN_LOG_NONE,
    Fatal = raw::flann_log_level_t_FLANN_LOG_FATAL,
    Error = raw::flann_log_level_t_FLANN_LOG_ERROR,
    Warn = raw::flann_log_level_t_FLANN_LOG_WARN,
    Info = raw::flann_log_level_t_FLANN_LOG_INFO,
    Debug = raw::flann_log_level_t_FLANN_LOG_DEBUG,
);

convertable_enum!(DistanceType; raw::flann_distance_t;
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

    pub fn as_raw(self) -> i32 {
        match self {
            Checks::Unlimited => raw::flann_checks_t_FLANN_CHECKS_UNLIMITED,
            Checks::Autotuned => raw::flann_checks_t_FLANN_CHECKS_AUTOTUNED,
            Checks::Exact(v) => v,
        }
    }
}
