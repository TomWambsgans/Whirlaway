/// Target single-thread workload size for `T`.
/// Should ideally be a multiple of a cache line (64 bytes)
/// and close to the L1 cache size (32 KB).
pub const fn workload_size<T: Sized>() -> usize {
    const CACHE_SIZE: usize = 1 << 15;
    CACHE_SIZE / size_of::<T>()
}

/// Compute the largest factor of `n` that is ≤ sqrt(n).
/// Assumes `n` is of the form `2^k * {1,3,9}`.
pub fn sqrt_factor(n: usize) -> usize {
    // Count the number of trailing zeros in `n`, i.e., the power of 2 in `n`
    let twos = n.trailing_zeros();

    // Divide `n` by the highest power of 2 to extract the base component
    let base = n >> twos;

    // Determine the largest factor ≤ sqrt(n) based on the extracted `base`
    match base {
        // Case: `n` is purely a power of 2 (base = 1)
        // The largest factor ≤ sqrt(n) is 2^(twos/2)
        1 => 1 << (twos / 2),

        // Case: `n = 2^k * 3`
        3 => {
            if twos == 0 {
                // sqrt(3) ≈ 1.73, so the largest integer factor ≤ sqrt(3) is 1
                1
            } else {
                // - If `twos` is even: The largest factor is `3 * 2^((twos - 1) / 2)`
                // - If `twos` is odd: The largest factor is `2^((twos / 2))`
                if twos % 2 == 0 {
                    3 << ((twos - 1) / 2)
                } else {
                    2 << (twos / 2)
                }
            }
        }

        // Case: `n = 2^k * 9`
        9 => {
            if twos == 1 {
                // sqrt(9 * 2^1) = sqrt(18) ≈ 4.24, largest factor ≤ sqrt(18) is 3
                3
            } else {
                // - If `twos` is even: The largest factor is `3 * 2^(twos / 2)`
                // - If `twos` is odd: The largest factor is `4 * 2^(twos / 2)`
                if twos % 2 == 0 {
                    3 << (twos / 2)
                } else {
                    4 << (twos / 2)
                }
            }
        }

        // If `base` is not in {1,3,9}, `n` is not in the expected form
        _ => panic!("n is not in the form 2^k * {{1,3,9}}"),
    }
}

/// Least common multiple.
///
/// Note that lcm(0,0) will panic (rather than give the correct answer 0).
pub const fn lcm(a: usize, b: usize) -> usize {
    a * (b / gcd(a, b))
}

/// Greatest common divisor.
pub const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
