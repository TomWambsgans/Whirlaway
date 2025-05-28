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
    for &byte in buff.iter().rev() {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.trailing_zeros() as usize;
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_ending_zero_bits_empty() {
        let data: [u8; 0] = [];
        assert_eq!(count_ending_zero_bits(&data), 0);
    }

    #[test]
    fn test_count_ending_zero_bits_all_zeros() {
        let data = [0x00, 0x00, 0x00];
        assert_eq!(count_ending_zero_bits(&data), 24);
    }

    #[test]
    fn test_count_ending_zero_bits_single_one_bit_at_end() {
        let data = [0b00000001];
        assert_eq!(count_ending_zero_bits(&data), 0);
    }

    #[test]
    fn test_count_ending_zero_bits_single_one_bit_at_most_significant() {
        let data = [0b10000000];
        assert_eq!(count_ending_zero_bits(&data), 7);
    }

    #[test]
    fn test_count_ending_zero_bits_multiple_bytes_trailing_zeros() {
        // 0xAB at front, two zeros at end → 16 trailing zero bits
        let data = [0xAB, 0x00, 0x00];
        assert_eq!(count_ending_zero_bits(&data), 16);
    }

    #[test]
    fn test_count_ending_zero_bits_mixed_bytes() {
        let data = [0b11110000, 0b00001111, 0b00000000];
        // last byte 0, middle byte stops at no trailing zero
        assert_eq!(count_ending_zero_bits(&data), 8);
    }
}
