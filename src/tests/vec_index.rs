use super::*;

#[test]
fn builds_and_adds() {
    let mut index = VecIndex::<f32>::new(
        3,
        vec![
            vec![0.0; 3],
            vec![0.0; 3],
            vec![0.0; 3],
            vec![0.0; 3],
            vec![0.0; 3],
        ],
        Parameters::default(),
    ).unwrap();
    assert_eq!(index.count(), 5);
    index.add(vec![0.0; 3], None).unwrap();
    assert_eq!(index.count(), 6);
    index.add_multiple(vec![], None).unwrap();
    assert_eq!(index.count(), 6);
    index
        .add_multiple(
            vec![vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]],
            None,
        )
        .unwrap();
    assert_eq!(index.count(), 10);
}

#[test]
fn get_accesses_right_item() {
    let mut index = VecIndex::<f32>::new(
        3,
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0],
        ],
        Parameters::default(),
    ).unwrap();

    index.add(vec![16.0, 17.0, 18.0], None).unwrap();

    index.add_multiple(vec![], None).unwrap();

    index
        .add_multiple(
            vec![
                vec![19.0, 20.0, 21.0],
                vec![22.0, 23.0, 24.0],
                vec![25.0, 26.0, 27.0],
                vec![28.0, 29.0, 30.0],
            ],
            None,
        )
        .unwrap();

    assert_eq!(index.get(0), Some(&vec![1.0, 2.0, 3.0]));
    assert_eq!(index.get(1), Some(&vec![4.0, 5.0, 6.0]));
    assert_eq!(index.get(2), Some(&vec![7.0, 8.0, 9.0]));
    assert_eq!(index.get(3), Some(&vec![10.0, 11.0, 12.0]));
    assert_eq!(index.get(4), Some(&vec![13.0, 14.0, 15.0]));
    assert_eq!(index.get(5), Some(&vec![16.0, 17.0, 18.0]));
    assert_eq!(index.get(6), Some(&vec![19.0, 20.0, 21.0]));
    assert_eq!(index.get(7), Some(&vec![22.0, 23.0, 24.0]));
    assert_eq!(index.get(8), Some(&vec![25.0, 26.0, 27.0]));
    assert_eq!(index.get(9), Some(&vec![28.0, 29.0, 30.0]));
    assert_eq!(index.get(10), None);
}

#[test]
fn nearest_neighbor_returns_correct_item() {
    let index = VecIndex::<f32>::new(
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
    ).unwrap();

    assert_eq!(
        index
            .find_nearest_neighbor(&vec![-1.0, -1.0, -1.0])
            .unwrap()
            .0,
        0
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&vec![-1.0, -1.0, 2.0])
            .unwrap()
            .0,
        1
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&vec![-1.0, 2.0, -1.0])
            .unwrap()
            .0,
        2
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&vec![-1.0, 2.0, 2.0])
            .unwrap()
            .0,
        3
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&vec![2.0, -1.0, -1.0])
            .unwrap()
            .0,
        4
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&vec![2.0, -1.0, 2.0])
            .unwrap()
            .0,
        5
    );
    assert_eq!(
        index
            .find_nearest_neighbor(&vec![2.0, 2.0, -1.0])
            .unwrap()
            .0,
        6
    );
    assert_eq!(
        index.find_nearest_neighbor(&vec![2.0, 2.0, 2.0]).unwrap().0,
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
    for v in data.clone() {
        index.add(v, None).unwrap();
    }
    let res = index.find_nearest_neighbors(&data, 3).unwrap();

    assert_eq!(res.len(), 10);

    // indices: [
    //     [1, 2, 9],
    //     [2, 9, 1],
    //     [3, 10, 5],
    //     [4, 5, 10],
    //     [5, 4, 10],
    //     [6, 8, 0],
    //     [7, 8, 9],
    //     [8, 7, 6],
    //     [9, 2, 1],
    //     [10, 3, 5],
    // ]
    // distances: [
    //     [0.0, 26674.0, 62010.0],
    //     [0.0, 12340.0, 26674.0],
    //     [0.0, 17828.0, 55370.0],
    //     [0.0, 27490.0, 46628.0],
    //     [0.0, 27490.0, 30274.0],
    //     [0.0, 50833.0, 65025.0],
    //     [0.0, 15005.0, 99700.0],
    //     [0.0, 15005.0, 50833.0],
    //     [0.0, 12340.0, 62010.0],
    //     [0.0, 17828.0, 30274.0],
    // ]
    assert_eq!(res[0][0].0, 1);
    assert_eq!(res[0][1].0, 2);
    assert_eq!(res[0][2].0, 9);
    assert_eq!(res[1][0].0, 2);
    assert_eq!(res[1][1].0, 9);
    assert_eq!(res[1][2].0, 1);
    assert_eq!(res[2][0].0, 3);
    assert_eq!(res[2][1].0, 10);
    assert_eq!(res[2][2].0, 5);

    assert_eq!(res[0][0].1, 0f32);
    assert_eq!(res[0][1].1, 26674f32);
    assert_eq!(res[0][2].1, 62010f32);
}

#[test]
fn nearest_neighbors_get_truncated() {
    type Point2 = VecIndex<f32>;
    let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    let index = Point2::new(2, vec![vec![0.0; 2], vec![0.0; 2]], Parameters::default()).unwrap();
    let res = index.find_nearest_neighbors(&data, 4).unwrap();

    assert_eq!(res.len(), 3);
    assert_eq!(res[0].len(), 2);
    assert_eq!(res[1].len(), 2);
    assert_eq!(res[2].len(), 2);
}

#[test]
fn search_radius_returns_correct_item() {
    let index = VecIndex::<f32>::new(
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
    ).unwrap();

    let mut indices = index
        .search_radius(&vec![0.0, 0.0, -1.0], 1.1, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 1.1, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 10.0, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 10.0, 0)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 2.1, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4, 5, 6]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 3.1, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 4.1, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&vec![2.0, 0.0, 0.0], 6.1, 10)
        .unwrap()
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
