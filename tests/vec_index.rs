#[macro_use]
extern crate itertools;
#[macro_use]
extern crate assert_approx_eq;
extern crate flann;

use flann::*;

#[test]
fn builds_and_adds() {
    let mut index: VecIndex<f32> =
        VecIndex::new(3, vec![vec![0.0; 3]; 5], Parameters::default()).unwrap();
    assert_eq!(index.len(), 5);
    index.add(vec![0.0; 3]).unwrap();
    assert_eq!(index.len(), 6);
    index.add_many_slices(&[]).unwrap();
    assert_eq!(index.len(), 6);
    index.add_many(vec![vec![0.0; 3]; 4]).unwrap();
    assert_eq!(index.len(), 10);
    index.add_many_slices(&[0.0; 3 * 4]).unwrap();
    assert_eq!(index.len(), 14);
}

#[test]
fn get_accesses_right_item() {
    let mut index: VecIndex<f32> = VecIndex::new(
        3,
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0],
        ],
        Parameters::default(),
    )
    .unwrap();

    index.add_slice(&[16.0, 17.0, 18.0]).unwrap();

    index.add_many_slices(&[]).unwrap();

    index
        .add_many(vec![
            vec![19.0, 20.0, 21.0],
            vec![22.0, 23.0, 24.0],
            vec![25.0, 26.0, 27.0],
            vec![28.0, 29.0, 30.0],
        ])
        .unwrap();

    for (real, test) in index
        .get(0)
        .unwrap()
        .iter()
        .zip(&[1.0f32, 2.0f32, 3.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(1)
        .unwrap()
        .iter()
        .zip(&[4.0f32, 5.0f32, 6.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(2)
        .unwrap()
        .iter()
        .zip(&[7.0f32, 8.0f32, 9.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(3)
        .unwrap()
        .iter()
        .zip(&[10.0f32, 11.0f32, 12.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(4)
        .unwrap()
        .iter()
        .zip(&[13.0f32, 14.0f32, 15.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(5)
        .unwrap()
        .iter()
        .zip(&[16.0f32, 17.0f32, 18.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(6)
        .unwrap()
        .iter()
        .zip(&[19.0f32, 20.0f32, 21.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(7)
        .unwrap()
        .iter()
        .zip(&[22.0f32, 23.0f32, 24.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(8)
        .unwrap()
        .iter()
        .zip(&[25.0f32, 26.0f32, 27.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    for (real, test) in index
        .get(9)
        .unwrap()
        .iter()
        .zip(&[28.0f32, 29.0f32, 30.0f32][..])
    {
        assert_approx_eq!(real, test);
    }
    assert!(index.get(10).is_none());
}

#[test]
fn nearest_neighbor_returns_correct_item() {
    let mut index: VecIndex<f32> = VecIndex::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ],
        Parameters::default(),
    )
    .unwrap();

    assert_eq!(
        index
            .find_nearest_neighbor(&[-1.0, -1.0, -1.0])
            .unwrap()
            .index,
        0
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&[-1.0, -1.0, 2.0])
            .unwrap()
            .index,
        1
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&[-1.0, 2.0, -1.0])
            .unwrap()
            .index,
        2
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&[-1.0, 2.0, 2.0])
            .unwrap()
            .index,
        3
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&[2.0, -1.0, -1.0])
            .unwrap()
            .index,
        4
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&[2.0, -1.0, 2.0])
            .unwrap()
            .index,
        5
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&[2.0, 2.0, -1.0])
            .unwrap()
            .index,
        6
    );
    assert_eq!(
        index.find_nearest_neighbor(&[2.0, 2.0, 2.0]).unwrap().index,
        7
    );
}

#[test]
fn nearest_neighbors_returns_correct_item() {
    type Point2 = VecIndex<f32>;
    let data = vec![
        vec![413.0, 800.0],
        vec![256.0, 755.0],
        vec![843.0, 586.0],
        vec![922.0, 823.0],
        vec![724.0, 789.0],
        vec![252.0, 39.0],
        vec![350.0, 369.0],
        vec![339.0, 247.0],
        vec![212.0, 653.0],
        vec![881.0, 714.0],
    ];
    let mut index = Point2::new(2, vec![vec![0.0; 2]], Parameters::default()).unwrap();
    for point in data.clone() {
        index.add(point).unwrap();
    }
    let nearest_neighbors = index.find_many_nearest_neighbors(3, data).unwrap();

    let indices = [
        [1, 2, 9],
        [2, 9, 1],
        [3, 10, 5],
        [4, 10, 5],
        [5, 10, 4],
        [6, 8, 0],
        [7, 8, 9],
        [8, 7, 6],
        [9, 2, 1],
        [10, 4, 3],
    ];
    let distances_squared = [
        [0.0, 26674.0, 62010.0],
        [0.0, 12340.0, 26674.0],
        [0.0, 17828.0, 55370.0],
        [0.0, 13562.0, 40360.0],
        [0.0, 30274.0, 40360.0],
        [0.0, 50833.0, 65025.0],
        [0.0, 15005.0, 99700.0],
        [0.0, 15005.0, 50833.0],
        [0.0, 12340.0, 62010.0],
        [0.0, 13562.0, 17828.0],
    ];
    for (neighbors, indices, distances_squared) in izip!(
        (&nearest_neighbors).into_iter(),
        indices.iter(),
        distances_squared.iter()
    ) {
        for (neighbor, &index, distance_squared) in izip!(neighbors, indices, distances_squared) {
            assert_eq!(neighbor.index, index);
            assert_approx_eq!(neighbor.distance_squared, distance_squared);
        }
    }
}

#[test]
fn nearest_neighbors_get_truncated() {
    type Point2 = VecIndex<f32>;
    let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    let mut index = Point2::new(2, vec![vec![0.0; 2]], Parameters::default()).unwrap();
    let res = index.find_many_nearest_neighbors(4, data).unwrap();
    let mut res = (&res).into_iter();

    let mut first = res.next().unwrap();
    assert_eq!(first.next().unwrap().index, 0);
    assert!(first.next().is_none());
    assert_eq!(res.next().unwrap().next().unwrap().index, 0);
    assert_eq!(res.next().unwrap().next().unwrap().index, 0);
    assert!(res.next().is_none());
}

#[test]
fn search_radius_returns_correct_item() {
    let mut index: VecIndex<f32> = VecIndex::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ],
        Parameters::default(),
    )
    .unwrap();

    let mut indices = index
        .find_nearest_neighbors_radius(10, 1.1, &[0.0, 0.0, -1.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0]);

    let mut indices = index
        .find_nearest_neighbors_radius(10, 1.1, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4]);

    let mut indices = index
        .find_nearest_neighbors_radius(10, 10.0, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let mut indices = index
        .find_nearest_neighbors_radius(0, 10.0, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![]);

    let mut indices = index
        .find_nearest_neighbors_radius(10, 2.1, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4, 5, 6]);

    let mut indices = index
        .find_nearest_neighbors_radius(10, 3.1, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4, 5, 6, 7]);

    let mut indices = index
        .find_nearest_neighbors_radius(10, 4.1, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 4, 5, 6, 7]);

    let mut indices = index
        .find_nearest_neighbors_radius(10, 6.1, &[2.0, 0.0, 0.0])
        .unwrap()
        .map(|v| v.index)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
