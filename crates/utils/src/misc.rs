#[inline(always)]
pub const fn log2_up(x: usize) -> u32 {
    // rounded up
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }
}

pub fn switch_endianness(mut x: usize, n: usize) -> usize {
    let mut y = 0;
    for _ in 0..n {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}
