#[inline(always)]
pub const fn log2_up(x: usize) -> usize {
    if x == 0 {
        0
    } else {
        usize::BITS as usize - (x - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_up_zero() {
        assert_eq!(log2_up(0), 0);
    }

    #[test]
    fn test_log2_up_powers_of_two() {
        assert_eq!(log2_up(1), 0); // 2^0
        assert_eq!(log2_up(2), 1); // 2^1
        assert_eq!(log2_up(4), 2); // 2^2
        assert_eq!(log2_up(8), 3); // 2^3
        assert_eq!(log2_up(16), 4); // 2^4
        assert_eq!(log2_up(1024), 10); // 2^10
    }

    #[test]
    fn test_log2_up_non_powers_of_two() {
        assert_eq!(log2_up(3), 2); // between 2^1 and 2^2 → ceil is 2
        assert_eq!(log2_up(5), 3); // between 2^2 and 2^3 → ceil is 3
        assert_eq!(log2_up(9), 4); // between 2^3 and 2^4 → ceil is 4
        assert_eq!(log2_up(17), 5); // between 2^4 and 2^5 → ceil is 5
        assert_eq!(log2_up(1025), 11); // just over 2^10 → ceil is 11
    }

    #[test]
    fn test_log2_up_large_values() {
        let max = usize::MAX;
        let expected = usize::BITS as usize;
        assert_eq!(log2_up(max), expected);
    }
}
