use algebra::ntt::Radix2EvaluationDomain;
use p3_field::TwoAdicField;

#[derive(Debug, Clone)]
pub struct Domain<F: TwoAdicField> {
    pub base_domain: Option<Radix2EvaluationDomain<F>>, // The domain (in the base
    // field) for the initial FFT
    pub backing_domain: Radix2EvaluationDomain<F>,
}

impl<F: TwoAdicField> Domain<F> {
    pub fn new(degree: usize, log_rho_inv: usize) -> Option<Self> {
        let size = degree * (1 << log_rho_inv);
        let base_domain = Radix2EvaluationDomain::new(size)?;
        let backing_domain = Self::to_extension_domain(&base_domain);

        Some(Self {
            backing_domain,
            base_domain: Some(base_domain),
        })
    }

    pub fn size(&self) -> usize {
        self.backing_domain.size()
    }

    pub fn scale(&self, power: usize) -> Self {
        Self {
            backing_domain: self.scale_generator_by(power),
            base_domain: None, // Set to zero because we only care for the initial
        }
    }

    fn to_extension_domain(domain: &Radix2EvaluationDomain<F>) -> Radix2EvaluationDomain<F> {
        let group_gen = F::from(domain.group_gen());
        let group_gen_inv = F::from(domain.group_gen_inv());
        let size = domain.size() as u64;
        let log_size_of_group = domain.log_size_of_group() as u32;
        let size_as_field_element = F::from(domain.size_as_field_element);
        let size_inv = F::from(domain.size_inv());
        let offset = F::from(domain.coset_offset());
        let offset_inv = F::from(domain.coset_offset_inv());
        let offset_pow_size = F::from(domain.coset_offset_pow_size());
        Radix2EvaluationDomain {
            size,
            log_size_of_group,
            size_as_field_element,
            size_inv,
            group_gen,
            group_gen_inv,
            offset,
            offset_inv,
            offset_pow_size,
        }
    }

    // Takes the underlying backing_domain = <w>, and computes the new domain
    // <w^power> (note this will have size |L| / power)
    fn scale_generator_by(&self, power: usize) -> Radix2EvaluationDomain<F> {
        let starting_size = self.size();
        assert_eq!(starting_size % power, 0);
        let new_size = starting_size / power;
        let log_size_of_group = new_size.trailing_zeros();
        let size_as_field_element = F::from_u64(new_size as u64);

        let group_gen = self.backing_domain.group_gen.exp_u64(power as u64);
        let group_gen_inv = group_gen.inverse();

        let offset = self.backing_domain.offset.exp_u64(power as u64);
        let offset_inv = self.backing_domain.offset_inv.exp_u64(power as u64);
        let offset_pow_size = offset.exp_u64(new_size as u64);

        Radix2EvaluationDomain {
            size: new_size as u64,
            log_size_of_group,
            size_as_field_element,
            size_inv: size_as_field_element.inverse(),
            group_gen,
            group_gen_inv,
            offset,
            offset_inv,
            offset_pow_size,
        }
    }
}
