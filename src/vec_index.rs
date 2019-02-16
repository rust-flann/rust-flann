use slice_index::SliceIndex;
use FlannError;
use Indexable;
use Parameters;

pub struct VecIndex<T: Indexable + 'static> {
    storage: Vec<Vec<T>>,
    slice_index: Option<SliceIndex<'static, T>>,
}

impl<T: Indexable> Drop for VecIndex<T> {
    fn drop(&mut self) {
        // We absolutely must destroy the index before our storage because
        // we are basically lying about the lifetime of the index using unsafe.
        // Be careful when changing this!
        self.slice_index.take();
    }
}

impl<T: Indexable> std::ops::Deref for VecIndex<T> {
    type Target = SliceIndex<'static, T>;

    fn deref(&self) -> &Self::Target {
        self.slice_index.as_ref().unwrap()
    }
}

impl<T: Indexable> std::ops::DerefMut for VecIndex<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice_index.as_mut().unwrap()
    }
}

impl<T: Indexable> VecIndex<T> {
    pub fn new<I, P>(
        point_len: usize,
        points: I,
        parameters: Parameters,
    ) -> Result<Self, FlannError>
    where
        I: IntoIterator<Item = P>,
        P: IntoIterator<Item = T>,
    {
        let mut points_vec = Vec::new();
        for point in points {
            let count = point.into_iter().map(|d| points_vec.push(d)).count();
            if count != point_len {
                return Err(FlannError::InvalidPointDimensionality {
                    expected: point_len,
                    got: count,
                });
            }
        }
        if points_vec.is_empty() {
            return Err(FlannError::ZeroInputPoints);
        }
        let index = SliceIndex::new(
            point_len,
            unsafe { std::mem::transmute(&points_vec[..]) },
            parameters,
        )?;
        Ok(Self {
            storage: vec![points_vec],
            slice_index: Some(index),
        })
    }

    /// Adds a point to the index.
    pub fn add(&mut self, point: Vec<T>) -> Result<(), FlannError> {
        self.slice_index
            .as_mut()
            .unwrap()
            .add_slice(unsafe { std::mem::transmute(&point[..]) })?;
        self.storage.push(point);
        Ok(())
    }

    /// Adds multiple points to the index.
    pub fn add_many<I, P>(&mut self, points: I) -> Result<(), FlannError>
    where
        I: IntoIterator<Item = P>,
        P: IntoIterator<Item = T>,
    {
        let mut points_vec = Vec::new();
        for point in points {
            let count = point.into_iter().map(|d| points_vec.push(d)).count();
            if count != self.point_len {
                return Err(FlannError::InvalidPointDimensionality {
                    expected: self.point_len,
                    got: count,
                });
            }
        }
        self.add_many_slices(unsafe { std::mem::transmute(&points_vec[..]) })?;
        self.storage.push(points_vec);
        Ok(())
    }
}
