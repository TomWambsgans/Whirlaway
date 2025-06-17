use p3_field::Field;

/// outputs the vector [1, base, base^2, base^3, ...] of length len.
pub fn powers<F: Field>(base: F, len: usize) -> Vec<F> {
    base.powers().take(len).collect()
}
