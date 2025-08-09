use std::ops::Range;

pub fn transmute_slice<Before, After>(slice: &[Before]) -> &[After] {
    let new_len = slice.len() * std::mem::size_of::<Before>() / std::mem::size_of::<After>();
    assert_eq!(
        slice.len() * std::mem::size_of::<Before>(),
        new_len * std::mem::size_of::<After>()
    );
    assert_eq!(slice.as_ptr() as usize % std::mem::align_of::<After>(), 0);
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const After, new_len) }
}

pub fn shift_range(range: Range<usize>, shift: usize) -> Range<usize> {
    Range {
        start: range.start + shift,
        end: range.end + shift,
    }
}

pub fn diff_to_next_power_of_two(n: usize) -> usize {
    n.next_power_of_two() - n
}

pub fn left_mut<A>(slice: &mut [A]) -> &mut [A] {
    assert!(slice.len() % 2 == 0);
    let mid = slice.len() / 2;
    &mut slice[..mid]
}

pub fn right_mut<A>(slice: &mut [A]) -> &mut [A] {
    assert!(slice.len() % 2 == 0);
    let mid = slice.len() / 2;
    &mut slice[mid..]
}

pub fn from_end<A>(slice: &[A], n: usize) -> &[A] {
    assert!(n <= slice.len());
    &slice[slice.len() - n..]
}

#[macro_export]
macro_rules! assert_eq_many {
    ($first:expr, $($rest:expr),+ $(,)?) => {
        {
            let first_val = $first;
            $(
                assert_eq!(first_val, $rest,
                    "assertion failed: `(left == right)`\n  left: `{:?}`,\n right: `{:?}`",
                    first_val, $rest);
            )+
        }
    };
}
