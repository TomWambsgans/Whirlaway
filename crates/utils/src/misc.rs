pub const fn log2_up(x: usize) -> usize {
    // rounded up
    (if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }) as usize
}

pub fn count_ending_zero_bits(buff: &[u8]) -> usize {
    let mut count = 0;
    for byte in buff.iter().rev() {
        for i in 0..8 {
            if byte & (1 << i) != 0 {
                return count;
            }
            count += 1;
        }
    }
    count
}
