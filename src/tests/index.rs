use super::*;

#[test]
fn builds_and_adds() {
    let mut index = Index::<f32, typenum::U3>::new(
        &[
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
        ],
        Parameters::default(),
    ).unwrap();
    assert_eq!(index.count(), 5);
    index.add(&Default::default(), None);
    assert_eq!(index.count(), 6);
    index.add_multiple(&[], None);
    assert_eq!(index.count(), 6);
    index.add_multiple(
        &[
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
        ],
        None,
    );
    assert_eq!(index.count(), 10);
}

#[test]
fn get_accesses_right_item() {
    let mut index = Index::<f32, typenum::U3>::new(
        &[
            arr![f32; 1, 2, 3],
            arr![f32; 4, 5, 6],
            arr![f32; 7, 8, 9],
            arr![f32; 10, 11, 12],
            arr![f32; 13, 14, 15],
        ],
        Parameters::default(),
    ).unwrap();

    index.add(&arr![f32; 16, 17, 18], None);

    index.add_multiple(&[], None);

    index.add_multiple(
        &[
            arr![f32; 19, 20, 21],
            arr![f32; 22, 23, 24],
            arr![f32; 25, 26, 27],
            arr![f32; 28, 29, 30],
        ],
        None,
    );

    assert_eq!(index.get(0), Some(arr![f32; 1, 2, 3]));
    assert_eq!(index.get(1), Some(arr![f32; 4, 5, 6]));
    assert_eq!(index.get(2), Some(arr![f32; 7, 8, 9]));
    assert_eq!(index.get(3), Some(arr![f32; 10, 11, 12]));
    assert_eq!(index.get(4), Some(arr![f32; 13, 14, 15]));
    assert_eq!(index.get(5), Some(arr![f32; 16, 17, 18]));
    assert_eq!(index.get(6), Some(arr![f32; 19, 20, 21]));
    assert_eq!(index.get(7), Some(arr![f32; 22, 23, 24]));
    assert_eq!(index.get(8), Some(arr![f32; 25, 26, 27]));
    assert_eq!(index.get(9), Some(arr![f32; 28, 29, 30]));
    assert_eq!(index.get(10), None);
}

#[test]
fn nearest_neighbor_returns_correct_item() {
    let index = Index::<f32, typenum::U3>::new(
        &[
            arr![f32; 0, 0, 0],
            arr![f32; 0, 0, 1],
            arr![f32; 0, 1, 0],
            arr![f32; 0, 1, 1],
            arr![f32; 1, 0, 0],
            arr![f32; 1, 0, 1],
            arr![f32; 1, 1, 0],
            arr![f32; 1, 1, 1],
        ],
        Parameters::default(),
    ).unwrap();

    assert_eq!(index.find_nearest_neighbor(&arr![f32; -1, -1, -1]).0, 0);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; -1, -1, 2]).0, 1);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; -1, 2, -1]).0, 2);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; -1, 2, 2]).0, 3);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; 2, -1, -1]).0, 4);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; 2, -1, 2]).0, 5);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; 2, 2, -1]).0, 6);
    assert_eq!(index.find_nearest_neighbor(&arr![f32; 2, 2, 2]).0, 7);
}

#[test]
fn search_radius_returns_correct_item() {
    let index = Index::<f32, typenum::U3>::new(
        &[
            arr![f32; 0, 0, 0],
            arr![f32; 0, 0, 1],
            arr![f32; 0, 1, 0],
            arr![f32; 0, 1, 1],
            arr![f32; 1, 0, 0],
            arr![f32; 1, 0, 1],
            arr![f32; 1, 1, 0],
            arr![f32; 1, 1, 1],
        ],
        Parameters::default(),
    ).unwrap();

    let mut indices = index
        .search_radius(&arr![f32; 0, 0, -1], 1.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 1.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 10.0, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 10.0, 0)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 2.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4, 5, 6]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 3.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 4.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 5.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 4, 5, 6, 7]);

    let mut indices = index
        .search_radius(&arr![f32; 2, 0, 0], 6.1, 10)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<usize>>();
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}
