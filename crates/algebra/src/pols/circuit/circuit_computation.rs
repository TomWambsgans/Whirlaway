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
    Sum((ComputationInput<F, N>, ComputationInput<F, N>)),
    Product((ComputationInput<F, N>, ComputationInput<F, N>)),
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

    pub fn is_scalar(&self) -> bool {
        matches!(self, ComputationInput::Scalar(_))
    }
}

impl<F: Field> ArithmeticCircuit<F, usize> {
    pub fn fix_computation(&self) -> CircuitComputation<F, usize> {
        let stack_size = RefCell::new(0);
        let instructions = RefCell::new(Vec::new());
        self.parse(
            &|scalar| ComputationInput::Scalar(scalar.clone()),
            &|node| ComputationInput::Node(node.clone()),
            &|left, right| {
                if left.is_scalar() && right.is_scalar() {
                    tracing::warn!("Product with more than one scalar input");
                }
                instructions
                    .borrow_mut()
                    .push(CircuitInstruction::Product((left, right)));
                let mut stack_size_guard = stack_size.borrow_mut();
                *stack_size_guard += 1;
                ComputationInput::Stack(*stack_size_guard - 1)
            },
            &|left, right| {
                if left.is_scalar() && right.is_scalar() {
                    tracing::warn!("Sum with more than one scalar input");
                }
                instructions
                    .borrow_mut()
                    .push(CircuitInstruction::Sum((left, right)));
                let mut stack_size_guard = stack_size.borrow_mut();
                *stack_size_guard += 1;
                ComputationInput::Stack(*stack_size_guard - 1)
            },
        );

        let mut instructions = instructions.into_inner();
        // trick to speed up computations (to avoid the initial scalar embedding)
        for inst in &mut instructions {
            let (CircuitInstruction::Sum((left, right))
            | CircuitInstruction::Product((left, right))) = inst;
            if matches!(left, ComputationInput::Scalar(_)) {
                std::mem::swap(left, right);
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
                CircuitInstruction::Sum((left, right)) => {
                    let eval_left = left.eval(point, &stack);
                    let sum = match right {
                        ComputationInput::Node(var_index) => eval_left + point[*var_index],
                        ComputationInput::Scalar(scalar) => eval_left + *scalar,
                        ComputationInput::Stack(stack_index) => eval_left + stack[*stack_index],
                    };
                    stack.push(sum);
                }
                CircuitInstruction::Product((left, right)) => {
                    let eval_left = left.eval(point, &stack);
                    let product = match right {
                        ComputationInput::Node(var_index) => eval_left * point[*var_index],
                        ComputationInput::Scalar(scalar) => eval_left * *scalar,
                        ComputationInput::Stack(stack_index) => eval_left * stack[*stack_index],
                    };
                    stack.push(product);
                }
            }
        }
        assert_eq!(stack.len(), self.stack_size);
        stack[self.stack_size - 1]
    }
}
