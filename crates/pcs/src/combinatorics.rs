#[derive(Debug, Clone)]
pub struct TreeOfVariables {
    pub vars_per_polynomial: Vec<usize>,
    pub root: TreeOfVariablesInner,
}

#[derive(Debug, Clone)]
pub enum TreeOfVariablesInner {
    Polynomial(usize),
    Composed {
        left: Box<TreeOfVariablesInner>,
        right: Box<TreeOfVariablesInner>,
    },
}

impl TreeOfVariables {
    pub fn total_vars(&self) -> usize {
        self.root.total_vars(&self.vars_per_polynomial)
    }
}

impl TreeOfVariablesInner {
    pub fn total_vars(&self, vars_per_polynomial: &[usize]) -> usize {
        match self {
            TreeOfVariablesInner::Polynomial(i) => vars_per_polynomial[*i],
            TreeOfVariablesInner::Composed { left, right } => {
                1 + left
                    .total_vars(vars_per_polynomial)
                    .max(right.total_vars(vars_per_polynomial))
            }
        }
    }
}

impl TreeOfVariables {
    pub fn compute_optimal(vars_per_polynomial: Vec<usize>) -> Self {
        let n = vars_per_polynomial.len();
        assert!(n > 0);

        let polynomial_indices: Vec<usize> = (0..n).collect();
        let all_trees = Self::generate_all_trees(&polynomial_indices);

        let mut best_tree = None;
        let mut min_vars = usize::MAX;

        for tree in all_trees {
            let total = tree.total_vars(&vars_per_polynomial);
            if total < min_vars {
                min_vars = total;
                best_tree = Some(tree);
            }
        }

        Self {
            root: best_tree.unwrap(),
            vars_per_polynomial,
        }
    }

    fn generate_all_trees(indices: &[usize]) -> Vec<TreeOfVariablesInner> {
        if indices.len() == 1 {
            return vec![TreeOfVariablesInner::Polynomial(indices[0])];
        }
        let mut trees = Vec::new();
        for split_point in 1..indices.len() {
            let left_indices = &indices[0..split_point];
            let right_indices = &indices[split_point..];

            let left_trees = Self::generate_all_trees(left_indices);
            let right_trees = Self::generate_all_trees(right_indices);

            for left_tree in &left_trees {
                for right_tree in &right_trees {
                    trees.push(TreeOfVariablesInner::Composed {
                        left: Box::new(left_tree.clone()),
                        right: Box::new(right_tree.clone()),
                    });
                }
            }
        }
        trees
    }
}

#[cfg(test)]
#[test]
fn test_tree_of_variables() {
    let vars_per_polynomial = vec![2, 3, 1, 4, 7, 2, 7];
    let tree = TreeOfVariables::compute_optimal(vars_per_polynomial.clone());
    dbg!(&tree, tree.total_vars());
}
