#[inline(always)]
pub const fn log2_up(x: usize) -> usize {
    if x == 0 {
        0
    } else {
        usize::BITS as usize - (x - 1).leading_zeros() as usize
    }
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
