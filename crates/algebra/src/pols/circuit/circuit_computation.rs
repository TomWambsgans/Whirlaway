use std::{cell::RefCell, usize};

use p3_field::{ExtensionField, Field};

use super::ArithmeticCircuit;

type StackIndex = usize;

#[derive(Clone, Debug)]
pub struct CircuitComputation<F, N> {
    instructions: Vec<CircuitInstruction<F, N>>,
    stack_size: usize,
}

#[derive(Clone, Debug)]
pub enum CircuitInstruction<F, N> {
    Sum(Vec<ComputationInput<F, N>>),
    Product(Vec<ComputationInput<F, N>>),
}

#[derive(Clone, Debug)]
pub enum ComputationInput<F, N> {
    Node(N),
    Scalar(F),
    Stack(StackIndex),
}

impl<F: Field> ComputationInput<F, usize> {
    fn eval<EF: ExtensionField<F>>(&self, point: &[EF], stack: &[EF]) -> EF {
        match self {
            ComputationInput::Node(var_index) => point[*var_index],
            ComputationInput::Scalar(scalar) => EF::from(*scalar),
            ComputationInput::Stack(stack_index) => stack[*stack_index],
        }
    }
}

impl<F: Field> ArithmeticCircuit<F, usize> {
    pub fn fix_computation(&self) -> CircuitComputation<F, usize> {
        let stack_size = RefCell::new(0);
        let instructions = RefCell::new(Vec::new());
        self.parse(
            &|scalar| ComputationInput::Scalar(scalar.clone()),
            &|node| ComputationInput::Node(node.clone()),
            &|product| {
                assert!(!product.is_empty());
                if product
                    .iter()
                    .filter(|input| matches!(input, ComputationInput::Scalar(_)))
                    .count()
                    > 1
                {
                    tracing::warn!("Product with more than one scalar input");
                }

                instructions
                    .borrow_mut()
                    .push(CircuitInstruction::Product(product));
                let mut stack_size_guard = stack_size.borrow_mut();
                *stack_size_guard += 1;
                ComputationInput::Stack(*stack_size_guard - 1)
            },
            &|sum| {
                assert!(!sum.is_empty());
                if sum
                    .iter()
                    .filter(|input| matches!(input, ComputationInput::Scalar(_)))
                    .count()
                    > 1
                {
                    tracing::warn!("Sum with more than one scalar input");
                }
                instructions.borrow_mut().push(CircuitInstruction::Sum(sum));
                let mut stack_size_guard = stack_size.borrow_mut();
                *stack_size_guard += 1;
                ComputationInput::Stack(*stack_size_guard - 1)
            },
        );

        let mut instructions = instructions.into_inner();
        // trick to speed up computations (to avoid the initial scalar embedding)
        for inst in &mut instructions {
            let (CircuitInstruction::Sum(inputs) | CircuitInstruction::Product(inputs)) = inst;
            if matches!(inputs[0], ComputationInput::Scalar(_)) {
                // swap the first element with the last one
                let len = inputs.len();
                inputs.swap(0, len - 1);
            }
        }

        CircuitComputation {
            instructions,
            stack_size: stack_size.into_inner(),
        }
    }
}

impl<F: Field> CircuitComputation<F, usize> {
    pub fn eval<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        let mut stack = Vec::with_capacity(self.stack_size);
        for instruction in &self.instructions {
            match instruction {
                CircuitInstruction::Sum(inputs) => {
                    let mut sum = inputs[0].eval(point, &stack);
                    for input in &inputs[1..] {
                        match input {
                            ComputationInput::Node(var_index) => {
                                sum += point[*var_index];
                            }
                            ComputationInput::Scalar(scalar) => {
                                sum += *scalar;
                            }
                            ComputationInput::Stack(stack_index) => {
                                sum += stack[*stack_index];
                            }
                        }
                    }
                    stack.push(sum);
                }
                CircuitInstruction::Product(inputs) => {
                    let mut product = inputs[0].eval(point, &stack);
                    for input in &inputs[1..] {
                        match input {
                            ComputationInput::Node(var_index) => {
                                product *= point[*var_index];
                            }
                            ComputationInput::Scalar(scalar) => {
                                product *= *scalar;
                            }
                            ComputationInput::Stack(stack_index) => {
                                product *= stack[*stack_index];
                            }
                        }
                    }
                    stack.push(product);
                }
            }
        }
        assert_eq!(stack.len(), self.stack_size);
        stack[self.stack_size - 1]
    }
}
