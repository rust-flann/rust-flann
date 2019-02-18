extern crate itertools;
#[macro_use]
extern crate assert_approx_eq;
extern crate flann;
#[macro_use]
extern crate generic_array;

use flann::*;
use generic_array::GenericArray;

type BitIndex = Index<u8, typenum::U1>;

fn index<I>(features: I) -> BitIndex
where
    I: IntoIterator<Item = GenericArray<u8, typenum::U1>>,
{
    let mut parameters = Parameters::default();
    parameters.algorithm = Algorithm::Lsh;
    Index::new(features, parameters).unwrap()
}

#[test]
fn builds_and_adds() {
    let mut index = index(vec![
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ]);
    assert_eq!(index.len(), 5);
    index.add(Default::default());
    assert_eq!(index.len(), 6);
    index.add_multiple(vec![]);
    assert_eq!(index.len(), 6);
    index.add_multiple(vec![
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ]);
    assert_eq!(index.len(), 10);
}

#[test]
fn get_accesses_right_item() {
    let mut index = index((0..5).map(|i| arr![u8; i]).collect::<Vec<_>>());

    index.add(arr![u8; 5]);

    index.add_multiple(vec![]);

    index.add_multiple((6..10).map(|i| arr![u8; i]).collect::<Vec<_>>());

    for i in 0..10 {
        assert_eq!(index.get(i), Some(&arr![u8; i]));
    }
    assert_eq!(index.get(10), None);
}

#[test]
fn nearest_neighbors_is_correct_hamming_distance() {
    let data = (0..8).map(|i| arr![u8; i]).collect::<Vec<_>>();

    let mut index = index(data.clone());

    for (ix, neighbor) in data
        .iter()
        .map(|d| index.find_nearest_neighbor(d))
        .enumerate()
    {
        assert_approx_eq!(neighbor.distance_squared, 0.0);
        assert_eq!(neighbor.index, ix);
    }

    for (ix, mut neighbors) in data
        .iter()
        .map(|d| index.find_nearest_neighbors(4, d))
        .enumerate()
    {
        let neighbor0 = neighbors.next().unwrap();
        assert_eq!(neighbor0.index, ix);
        assert_approx_eq!(neighbor0.distance_squared, 0.0);

        // The next 3 neighbors should have a hamming distance of 1.
        for adjacent_neighbor in (&mut neighbors).take(3) {
            // Verify that the hamming distance is 1 in the data.
            assert!((data[adjacent_neighbor.index][0] ^ data[ix][0]).count_ones() == 1);
            // Verify that the hamming distance is 1 in the returned metric.
            assert_approx_eq!(adjacent_neighbor.distance_squared, 1.0);
        }

        assert!(neighbors.next().is_none());
    }
}
