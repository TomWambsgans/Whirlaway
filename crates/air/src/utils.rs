use algebra::pols::{MultilinearDevice, MultilinearHost, MultilinearsSlice, MultilinearsVec};
use arithmetic_circuit::{ArithmeticCircuit, TransparentPolynomial};
use cuda_bindings::{cuda_air_columns_down, cuda_air_columns_up};
use fiat_shamir::{FsError, FsParticipant};
use p3_field::{ExtensionField, Field, PrimeField};
use rayon::prelude::*;
use utils::log2_up;

use crate::{AirSettings, table::AirTable};

pub(crate) fn matrix_up_lde<F: Field>(log_length: usize) -> TransparentPolynomial<F> {
    /*
        Matrix UP:

       (1 0 0 0 ... 0 0 0)
       (0 1 0 0 ... 0 0 0)
       (0 0 1 0 ... 0 0 0)
       (0 0 0 1 ... 0 0 0)
       ...      ...   ...
       (0 0 0 0 ... 1 0 0)
       (0 0 0 0 ... 0 1 0)
       (0 0 0 0 ... 0 1 0)

       Square matrix of size self.n_columns x sef.n_columns
       As a multilinear polynomial in 2 * log_length variables:
       - self.n_columns first variables -> encoding the row index
       - self.n_columns last variables -> encoding the column index
    */

    TransparentPolynomial::eq_extension_2n_vars(log_length)
        + TransparentPolynomial::eq_extension_n_scalars(&vec![F::ONE; log_length * 2 - 1])
            * (ArithmeticCircuit::Scalar(F::ONE)
                - ArithmeticCircuit::Node(log_length * 2 - 1) * F::TWO)
}

pub(crate) fn matrix_down_lde<F: Field>(log_length: usize) -> TransparentPolynomial<F> {
    /*
        Matrix DOWN:

       (0 1 0 0 ... 0 0 0)
       (0 0 1 0 ... 0 0 0)
       (0 0 0 1 ... 0 0 0)
       (0 0 0 0 ... 0 0 0)
       (0 0 0 0 ... 0 0 0)
       ...      ...   ...
       (0 0 0 0 ... 0 1 0)
       (0 0 0 0 ... 0 0 1)
       (0 0 0 0 ... 0 0 1)

       Square matrix of size self.n_columns x sef.n_columns
       As a multilinear polynomial in 2 * log_length variables:
       - self.n_columns first variables -> encoding the row index
       - self.n_columns last variables -> encoding the column index

       TODO OPTIMIZATIOn:
       the lde currently is in log(table_length)^2, but it could be log(table_length) using a recursive construction
       (However it is not representable as a polynomial in this case, but as a fraction instead)

    */

    TransparentPolynomial::next(log_length)
        + TransparentPolynomial::eq_extension_n_scalars(&vec![F::ONE; log_length * 2])
    // bottom right corner
}

pub(crate) fn columns_up_and_down<F: Field>(columns: &MultilinearsSlice<F>) -> MultilinearsVec<F> {
    match columns {
        MultilinearsSlice::Host(columns) => {
            MultilinearsVec::Host(columns_up_and_down_host(columns))
        }
        MultilinearsSlice::Device(columns) => {
            MultilinearsVec::Device(columns_up_and_down_device(columns))
        }
    }
}
pub(crate) fn columns_up_and_down_device<F: Field>(
    columns: &[&MultilinearDevice<F>],
) -> Vec<MultilinearDevice<F>> {
    let mut res = Vec::new();
    res.extend(cuda_air_columns_up(columns));
    res.extend(cuda_air_columns_down(columns));
    res.into_iter().map(MultilinearDevice::new).collect()
}

pub(crate) fn columns_up_and_down_host<F: Field>(
    columns: &[&MultilinearHost<F>],
) -> Vec<MultilinearHost<F>> {
    let mut res = Vec::with_capacity(columns.len() * 2);
    res.par_extend(columns.par_iter().map(|c| column_up_host(c)));
    res.par_extend(columns.par_iter().map(|c| column_down_host(c)));
    res
}

pub(crate) fn column_up_host<F: Field>(column: &MultilinearHost<F>) -> MultilinearHost<F> {
    let mut up = column.clone();
    up.evals[column.n_coefs() - 1] = up.evals[column.n_coefs() - 2];
    up
}

pub(crate) fn column_down_host<F: Field>(column: &MultilinearHost<F>) -> MultilinearHost<F> {
    let mut down = column.evals[1..].to_vec();
    down.push(*down.last().unwrap());
    MultilinearHost::new(down)
}

impl<F: PrimeField> AirTable<F> {
    pub(crate) fn constraints_batching_pow<
        FS: FsParticipant,
        EF: ExtensionField<F>,
        WhirF: ExtensionField<F>,
    >(
        &self,
        fs: &mut FS,
        settings: &AirSettings<F, EF, WhirF>,
        cuda: bool,
    ) -> Result<(), FsError> {
        fs.challenge_pow(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.constraints.len()))),
            cuda,
        )
    }

    pub(crate) fn zerocheck_pow<
        FS: FsParticipant,
        EF: ExtensionField<F>,
        WhirF: ExtensionField<F>,
    >(
        &self,
        fs: &mut FS,
        settings: &AirSettings<F, EF, WhirF>,
        cuda: bool,
    ) -> Result<(), FsError> {
        fs.challenge_pow(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(self.log_length)),
            cuda,
        )
    }

    pub(crate) fn secondary_sumchecks_batching_pow<
        FS: FsParticipant,
        EF: ExtensionField<F>,
        WhirF: ExtensionField<F>,
    >(
        &self,
        fs: &mut FS,
        settings: &AirSettings<F, EF, WhirF>,
        cuda: bool,
    ) -> Result<(), FsError> {
        fs.challenge_pow(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_witness_columns() * 2))),
            cuda,
        )
    }
}

#[cfg(test)]
mod test {
    use cuda_engine::{CudaFunctionInfo, cuda_init, cuda_load_function, cuda_sync, memcpy_htod};
    use p3_koala_bear::KoalaBear;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    type F = KoalaBear;

    #[test]
    fn test_cuda_air_columns_up_down() {
        cuda_init();
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "multilinears_up",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "multilinears_down",
        ));
        let n_columns = 200;
        let n_vars = 15;
        let rng = &mut StdRng::seed_from_u64(0);
        let columns_host = (0..n_columns)
            .map(|_| MultilinearHost::<F>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let columns_dev = columns_host
            .iter()
            .map(|w| MultilinearDevice::new(memcpy_htod(&w.evals)))
            .collect::<Vec<_>>();
        cuda_sync();
        let up_and_down_host = columns_up_and_down_host(&columns_host.iter().collect::<Vec<_>>());
        let up_and_down_dev = columns_up_and_down_device(&columns_dev.iter().collect::<Vec<_>>());
        let up_and_down_dev_back = up_and_down_dev
            .iter()
            .map(|w| w.transfer_to_host())
            .collect::<Vec<_>>();
        cuda_sync();
        assert_eq!(up_and_down_host, up_and_down_dev_back);
    }
}
