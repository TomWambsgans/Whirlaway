pub fn max_degree_per_vars_prod(subs: &[Vec<usize>]) -> Vec<usize> {
    let n_vars = subs.iter().map(|s| s.len()).max().unwrap_or_default();
    let mut res = vec![0; n_vars];
    for i in 0..subs.len() {
        for j in 0..subs[i].len() {
            res[j] += subs[i][j];
        }
    }
    res
}

pub fn max_degree_per_vars_sum(subs: &[Vec<usize>]) -> Vec<usize> {
    let n_vars = subs.iter().map(|s| s.len()).max().unwrap();
    let mut res = vec![0; n_vars];
    for i in 0..subs.len() {
        for j in 0..subs[i].len() {
            res[j] = res[j].max(subs[i][j]);
        }
    }
    res
}
