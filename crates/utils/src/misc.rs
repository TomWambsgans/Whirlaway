use std::collections::BTreeSet;
use std::hash::{DefaultHasher, Hash, Hasher};

#[inline(always)]
pub const fn log2_up(x: usize) -> usize {
    if x == 0 {
        0
    } else {
        usize::BITS as usize - (x - 1).leading_zeros() as usize
    }
}

#[inline(always)]
pub const fn log2_down(x: usize) -> usize {
    if x == 0 {
        0
    } else {
        usize::BITS as usize - x.leading_zeros() as usize - 1
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

/// Deduplicates AND orders a vector
pub fn dedup<T: Ord>(v: impl IntoIterator<Item = T>) -> Vec<T> {
    Vec::from_iter(BTreeSet::from_iter(v))
}

pub fn default_hash<H: Hash>(h: H) -> u64 {
    let mut hasher = DefaultHasher::new();
    h.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_down_zero() {
        assert_eq!(log2_down(0), 0);
    }

    #[test]
    fn test_log2_down_powers_of_two() {
        assert_eq!(log2_down(1), 0);
        assert_eq!(log2_down(2), 1);
        assert_eq!(log2_down(4), 2);
        assert_eq!(log2_down(8), 3);
        assert_eq!(log2_down(16), 4);
        assert_eq!(log2_down(1024), 10);
    }

    #[test]
    fn test_log2_down_non_powers_of_two() {
        assert_eq!(log2_down(3), 1); // between 2 and 4 → floor is 1
        assert_eq!(log2_down(5), 2); // between 4 and 8 → floor is 2
        assert_eq!(log2_down(9), 3); // between 8 and 16 → floor is 3
        assert_eq!(log2_down(17), 4); // between 16 and 32 → floor is 4
        assert_eq!(log2_down(1023), 9); // just below 1024 → floor is 9
    }

    #[test]
    fn test_log2_down_large_values() {
        let max = usize::MAX;
        let expected = usize::BITS as usize - 1;
        assert_eq!(log2_down(max), expected);
    }

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
