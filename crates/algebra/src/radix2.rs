//! This module defines `Radix2EvaluationDomain`, an `EvaluationDomain`
//! for performing various kinds of polynomial arithmetic on top of
//! fields that are FFT-friendly.
//!
//! `Radix2EvaluationDomain` supports FFTs of size at most `2^F::TWO_ADICITY`.

use std::fmt;

use p3_field::TwoAdicField;

/// Factor that determines if a the degree aware FFT should be called.
///
/// Defines a domain over which finite field (I)FFTs can be performed. Works
/// only for fields that have a large multiplicative subgroup of size that is
/// a power-of-2.
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Radix2EvaluationDomain<F: TwoAdicField> {
    /// The size of the domain.
    pub size: u64,
    /// `log_2(self.size)`.
    pub log_size_of_group: u32,
    /// Size of the domain as a field element.
    pub size_as_field_element: F,
    /// Inverse of the size in the field.
    pub size_inv: F,
    /// A generator of the subgroup.
    pub group_gen: F,
    /// Inverse of the generator of the subgroup.
    pub group_gen_inv: F,
    /// Offset that specifies the coset.
    pub offset: F,
    /// Inverse of the offset that specifies the coset.
    pub offset_inv: F,
    /// Constant coefficient for the vanishing polynomial.
    /// Equals `self.offset^self.size`.
    pub offset_pow_size: F,
}

impl<F: TwoAdicField> fmt::Debug for Radix2EvaluationDomain<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Radix-2 multiplicative subgroup of size {}", self.size)
    }
}

impl<F: TwoAdicField> Radix2EvaluationDomain<F> {
    /// Construct a domain that is large enough for evaluations of a polynomial
    /// having `num_coeffs` coefficients.
    pub fn new(num_coeffs: usize) -> Option<Self> {
        let size = if num_coeffs.is_power_of_two() {
            num_coeffs
        } else {
            num_coeffs.checked_next_power_of_two()?
        } as u64;
        let log_size_of_group = size.trailing_zeros();

        // libfqfft uses > https://github.com/scipr-lab/libfqfft/blob/e0183b2cef7d4c5deb21a6eaf3fe3b586d738fe0/libfqfft/evaluation_domain/domains/basic_radix2_domain.tcc#L33
        if log_size_of_group > F::TWO_ADICITY as u32 {
            return None;
        }

        // Compute the generator for the multiplicative subgroup.
        // It should be the 2^(log_size_of_group) root of unity.
        let group_gen = F::two_adic_generator(log_size_of_group as usize);
        // Check that it is indeed the 2^(log_size_of_group) root of unity.
        debug_assert_eq!(group_gen.exp_u64(size), F::ONE);
        let size_as_field_element = F::from_u64(size);
        let size_inv = size_as_field_element.try_inverse()?;

        Some(Self {
            size,
            log_size_of_group,
            size_as_field_element,
            size_inv,
            group_gen,
            group_gen_inv: group_gen.try_inverse()?,
            offset: F::ONE,
            offset_inv: F::ONE,
            offset_pow_size: F::ONE,
        })
    }

    pub fn get_coset(&self, offset: F) -> Option<Self> {
        Some(Self {
            offset,
            offset_inv: offset.try_inverse()?,
            offset_pow_size: offset.exp_u64(self.size),
            ..*self
        })
    }

    pub fn compute_size_of_domain(num_coeffs: usize) -> Option<usize> {
        let size = num_coeffs.checked_next_power_of_two()?;
        if size.trailing_zeros() > F::TWO_ADICITY as u32 {
            None
        } else {
            Some(size)
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        usize::try_from(self.size).unwrap()
    }

    #[inline]
    pub fn log_size_of_group(&self) -> u64 {
        self.log_size_of_group as u64
    }

    #[inline]
    pub fn size_inv(&self) -> F {
        self.size_inv
    }

    #[inline]
    pub fn group_gen(&self) -> F {
        self.group_gen
    }

    #[inline]
    pub fn group_gen_inv(&self) -> F {
        self.group_gen_inv
    }

    #[inline]
    pub fn coset_offset(&self) -> F {
        self.offset
    }

    #[inline]
    pub fn coset_offset_inv(&self) -> F {
        self.offset_inv
    }

    #[inline]
    pub fn coset_offset_pow_size(&self) -> F {
        self.offset_pow_size
    }

    /// Returns the `i`-th element of the domain.
    pub fn element(&self, i: usize) -> F {
        let mut result = self.group_gen().exp_u64(i as u64);
        if !self.coset_offset().is_one() {
            result *= self.coset_offset()
        }
        result
    }
}
