use std::{
    cell::RefCell,
    collections::{BTreeSet, HashSet},
};

use p3_field::{ExtensionField, Field};

use super::TransparentPolynomial;

type StackIndex = usize;

#[derive(Clone, Debug, Hash)]
pub struct CircuitComputation<F> {
    pub instructions: Vec<CircuitInstruction<F>>,
    pub stack_size: usize,
    pub n_vars: usize,
    pub degree_per_var: Vec<usize>,
    pub composition_degree: usize, // if all the avriables where the same X -> represents the degree of X
}

#[derive(Clone, Debug, Hash)]
pub struct CircuitInstruction<F> {
    pub op: CircuitOp,
    pub left: ComputationInput<F>,
    pub right: ComputationInput<F>,
    pub result_location: StackIndex,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CircuitOp {
    Sum,
    Product,
    Sub,
}

#[derive(Clone, Debug, Hash)]
pub enum ComputationInput<F> {
    Node(usize),
    Scalar(F),
    Stack(StackIndex),
}

impl<F: Field> ComputationInput<F> {
    fn eval<EF: ExtensionField<F>>(&self, point: &[EF], stack: &[EF]) -> EF {
        match self {
            ComputationInput::Node(var_index) => point[*var_index],
            ComputationInput::Scalar(scalar) => EF::from(*scalar),
            ComputationInput::Stack(stack_index) => stack[*stack_index],
        }
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, ComputationInput::Scalar(_))
    }
}

impl<F: Field> TransparentPolynomial<F> {
    pub fn fix_computation(&self, optimized: bool) -> CircuitComputation<F> {
        let stack_size = RefCell::new(0);
        let instructions = RefCell::new(Vec::new());
        let n_vars = RefCell::new(Option::<usize>::None);
        let build_instruction =
            |op: CircuitOp, left: ComputationInput<F>, right: ComputationInput<F>| {
                if left.is_scalar() && right.is_scalar() {
                    panic!("Instruction with more than one scalar input -> useless work");
                }
                let mut stack_size_guard = stack_size.borrow_mut();
                instructions.borrow_mut().push(CircuitInstruction {
                    op,
                    left,
                    right,
                    result_location: *stack_size_guard,
                });
                *stack_size_guard += 1;
                ComputationInput::Stack(*stack_size_guard - 1)
            };
        self.parse(
            &|scalar| ComputationInput::Scalar(*scalar),
            &|node| {
                let mut n_vars_guard = n_vars.borrow_mut();
                *n_vars_guard = Some((n_vars_guard.unwrap_or_default()).max(*node + 1));
                ComputationInput::Node(*node)
            },
            &|left, right| build_instruction(CircuitOp::Product, left, right),
            &|left, right| build_instruction(CircuitOp::Sum, left, right),
            &|left, right| build_instruction(CircuitOp::Sub, left, right),
        );

        let stack_size = stack_size.into_inner();
        let mut instructions = instructions.into_inner();
        let n_vars = n_vars.into_inner().unwrap_or_default();
        let degree_per_var = self.max_degree_per_vars(n_vars);
        let composition_degree = self
            .map_node(&|_| TransparentPolynomial::Node(0))
            .max_degree_per_vars(1)[0];

        // trick to speed up computations (to avoid the initial scalar embedding)
        for inst in &mut instructions {
            if matches!(inst.left, ComputationInput::Scalar(_)) && inst.op != CircuitOp::Sub {
                std::mem::swap(&mut inst.left, &mut inst.right);
            }
        }

        let mut res = CircuitComputation {
            instructions,
            stack_size,
            n_vars,
            degree_per_var,
            composition_degree,
        };
        if optimized {
            res = res.optimize_stack_usage();
        }
        res
    }
}

impl<F: Field> CircuitComputation<F> {
    pub fn eval<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        let mut stack = vec![EF::ZERO; self.stack_size];
        for instruction in &self.instructions {
            stack[instruction.result_location] = match &instruction.op {
                CircuitOp::Sum => {
                    instruction.left.eval(point, &stack) + instruction.right.eval(point, &stack)
                }
                CircuitOp::Product => {
                    instruction.left.eval(point, &stack) * instruction.right.eval(point, &stack)
                }
                CircuitOp::Sub => {
                    instruction.left.eval(point, &stack) - instruction.right.eval(point, &stack)
                }
            };
        }
        assert_eq!(stack.len(), self.stack_size);
        stack[self.stack_size - 1]
    }

    pub fn nodes_involved(&self) -> BTreeSet<usize> {
        let mut nodes = BTreeSet::new();
        for instruction in &self.instructions {
            if let ComputationInput::Node(node) = instruction.left {
                nodes.insert(node);
            }
            if let ComputationInput::Node(node) = instruction.right {
                nodes.insert(node);
            }
        }
        nodes
    }

    /// Optimize the stack usage by performing lifetime analysis and a linear‐scan register
    /// allocation that avoids reusing any register in the same instruction when it appears as a stack
    /// input. That is, for each instruction the new result_location (register) is guaranteed to be
    /// different than any register referenced by its left and right operands (when they are stack inputs).
    pub fn optimize_stack_usage(self) -> Self {
        // Phase 1: Build virtual registers and record lifetimes.
        //
        // Each instruction that writes a value will define a “virtual register”. Each such virtual register
        // tracks its definition (instruction index), the original stack slot that was written, and the last
        // instruction in which the value is used.
        #[derive(Clone, Debug)]
        struct VirtualReg {
            def_inst: usize,      // instruction index where it is defined.
            orig_slot: usize,     // original stack slot where it was stored.
            last_use: usize,      // last instruction index where the value is used.
            alloc: Option<usize>, // allocated (optimized) register; initially None.
        }

        let num_instructions = self.instructions.len();
        // For each original stack slot, track which virtual register is the most recent definition.
        let mut current_vreg: Vec<Option<usize>> = vec![None; self.stack_size];
        let mut vregs: Vec<VirtualReg> = Vec::with_capacity(num_instructions);

        // Iterate through instructions to build virtual registers.
        for (i, inst) in self.instructions.iter().enumerate() {
            // For every stack input used as an operand, update the corresponding virtual register's last_use.
            if let ComputationInput::Stack(x) = inst.left {
                if let Some(vr_idx) = current_vreg[x] {
                    vregs[vr_idx].last_use = i;
                }
            }
            if let ComputationInput::Stack(x) = inst.right {
                if let Some(vr_idx) = current_vreg[x] {
                    vregs[vr_idx].last_use = i;
                }
            }
            // Create a new virtual register for the definition.
            let orig_slot = inst.result_location;
            let new_vr_idx = vregs.len();
            vregs.push(VirtualReg {
                def_inst: i,
                orig_slot,
                // If there is no subsequent use, the lifetime is only this instruction.
                last_use: i,
                alloc: None,
            });
            current_vreg[orig_slot] = Some(new_vr_idx);
        }
        // Ensure that the final result (coming from the original stack slot stack_size - 1)
        // lives until after the last instruction.
        if let Some(vr_idx) = current_vreg[self.stack_size - 1] {
            vregs[vr_idx].last_use = num_instructions;
        }

        // Phase 2: Linear-scan register allocation considering the additional constraint.
        //
        // We process instructions in order (which is the same as the order of vreg definitions).
        // In addition to the interference from active registers, when choosing a register for an instruction's
        // definition, we also forbid any register(s) that are used by the operands (if they are stack inputs).
        //
        // To simulate the evolving mapping from original stack slots to their allocated registers,
        // we use `alloc_mapping`: for each original slot, which register allocation is currently there.
        let mut alloc_mapping: Vec<Option<usize>> = vec![None; self.stack_size];
        // `active` holds the indices (in the vregs vector) of virtual registers that have not expired.
        let mut active: Vec<usize> = Vec::new();
        let mut next_alloc = 0;

        // Since our vregs were created in order, we process them in increasing order of def_inst.
        for i in 0..num_instructions {
            // Expire any virtual registers whose lifetime ended before the current instruction.
            active.retain(|&j| vregs[j].last_use > i);

            // Build the "forbidden" set from the instruction's operands.
            let inst = &self.instructions[i];
            let mut forbidden: HashSet<usize> = HashSet::new();
            if let ComputationInput::Stack(x) = inst.left {
                if let Some(reg) = alloc_mapping[x] {
                    forbidden.insert(reg);
                }
            }
            if let ComputationInput::Stack(x) = inst.right {
                if let Some(reg) = alloc_mapping[x] {
                    forbidden.insert(reg);
                }
            }

            // Also get a set of registers used by active (live) virtual registers.
            let used: HashSet<usize> = active
                .iter()
                .map(|&j| {
                    vregs[j]
                        .alloc
                        .expect("active virtual register must have allocation")
                })
                .collect();

            // Choose the smallest available register that is not in `used` nor in `forbidden`.
            let mut candidate = 0;
            loop {
                if used.contains(&candidate) || forbidden.contains(&candidate) {
                    candidate += 1;
                } else {
                    break;
                }
            }
            vregs[i].alloc = Some(candidate);
            alloc_mapping[self.instructions[i].result_location] = Some(candidate);
            active.push(i);
            if candidate >= next_alloc {
                next_alloc = candidate + 1;
            }
        }

        // Ensure the final result comes from the last register (i.e. register number next_alloc - 1).
        let final_vr = vregs
            .iter_mut()
            .filter(|vr| vr.orig_slot == (self.stack_size - 1))
            .max_by_key(|vr| vr.def_inst)
            .expect("Final result must be defined");
        let final_alloc = final_vr.alloc.expect("Final result must have allocation");
        if final_alloc != next_alloc - 1 {
            // Swap allocation: find the virtual register using next_alloc - 1 and swap.
            for vr in vregs.iter_mut() {
                if let Some(a) = vr.alloc {
                    if a == final_alloc {
                        vr.alloc = Some(next_alloc - 1);
                    } else if a == (next_alloc - 1) {
                        vr.alloc = Some(final_alloc);
                    }
                }
            }
        }

        // Phase 3: Reconstruct instructions with new register indices.
        //
        // Now we replay the instructions in order and update any stack operand reference using our new allocations.
        let mut mapping: Vec<Option<usize>> = vec![None; self.stack_size];
        let mut new_instructions = Vec::with_capacity(self.instructions.len());
        // Build a per-instruction vector for the allocated register as computed.
        let mut def_alloc: Vec<Option<usize>> = vec![None; self.instructions.len()];
        for vr in &vregs {
            def_alloc[vr.def_inst] = vr.alloc;
        }

        for (i, inst) in self.instructions.into_iter().enumerate() {
            // Update left operand if it uses a stack.
            let new_left = match inst.left {
                ComputationInput::Stack(x) => {
                    let alloc = mapping[x].expect("Expected mapping for a used stack value");
                    ComputationInput::Stack(alloc)
                }
                other => other,
            };
            // Update right operand if it uses a stack.
            let new_right = match inst.right {
                ComputationInput::Stack(x) => {
                    let alloc = mapping[x].expect("Expected mapping for a used stack value");
                    ComputationInput::Stack(alloc)
                }
                other => other,
            };
            // Get new allocation for the definition.
            let new_alloc = def_alloc[i].expect("Definition allocation not found");

            let new_inst = CircuitInstruction {
                op: inst.op,
                left: new_left,
                right: new_right,
                result_location: new_alloc,
            };
            // Update the mapping for the original slot that is being redefined.
            mapping[inst.result_location] = Some(new_alloc);
            new_instructions.push(new_inst);
        }

        Self {
            instructions: new_instructions,
            stack_size: next_alloc, // the optimized number of registers.
            ..self
        }
    }
}

pub fn max_composition_degree<F: Field>(circuits: &[CircuitComputation<F>]) -> usize {
    circuits
        .iter()
        .map(|circuit| circuit.composition_degree)
        .max_by_key(|&degree| degree)
        .unwrap_or_default()
}

pub fn max_stack_size<F: Field>(circuits: &[CircuitComputation<F>]) -> usize {
    circuits
        .iter()
        .map(|circuit| circuit.stack_size)
        .max_by_key(|&stack_size| stack_size)
        .unwrap_or_default()
}

pub fn all_nodes_involved<F: Field>(circuits: &[CircuitComputation<F>]) -> BTreeSet<usize> {
    let mut all_nodes = BTreeSet::new();
    for circuit in circuits {
        all_nodes.extend(circuit.nodes_involved());
    }
    all_nodes
}

#[cfg(test)]
mod test {
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_optimize_stack_usage() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 8>;

        let circuit = TransparentPolynomial::<F>::random(&mut StdRng::seed_from_u64(0), 10, 100);

        let optimized = circuit.fix_computation(true);
        let non_optimized = circuit.fix_computation(false);

        println!("Non-optimized stack size: {}", non_optimized.stack_size);
        println!("Optimized stack size: {}", optimized.stack_size);

        assert_eq!(
            optimized.stack_size,
            optimized.clone().optimize_stack_usage().stack_size
        );

        for instr in &optimized.instructions {
            if let ComputationInput::Stack(x) = instr.left {
                assert!(x != instr.result_location);
            }
            if let ComputationInput::Stack(x) = instr.right {
                assert!(x != instr.result_location);
            }
        }

        let rng = &mut StdRng::seed_from_u64(0);
        let random_point = (0..10).map(|_| rng.random()).collect::<Vec<EF>>();
        assert_eq!(
            optimized.eval(&random_point),
            non_optimized.eval(&random_point)
        );
    }
}
