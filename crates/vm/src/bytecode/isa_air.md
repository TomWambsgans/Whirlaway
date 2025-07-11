# ISA, Arithmetization (DRAFT)

## Logup*

If we use logup* (https://eprint.iacr.org/2025/946.pdf), we can avoid committing to the bytecode columns (we only pay a constant cost in the size of the bytecode).


## Memory

Read-only (for now at least)

## Registers

- pc: Program Counter
- fp: Frame Pointer

Contrary to Cairo, no ap register (Allocation Pointer)

## Opcode format

Each opcode = 12 field elements:

- A_arg
- B_arg
- C_arg

### Binary columns

- A_flag: (0: MemoryAfterFp, 1: Constant)
- B_flag: (0: MemoryAfterFp, 1: Fp)
- C_flag: (0: MemoryAfterFp, 1: Constant)

### Instruction selector

- ComputationAdd
- ComputationMul
- MemoryPointer
- Jump
- Poseidon2_16
- Poseidon2_24

## Degree 2 AIR

### Execution columns (committed)

- pc
- fp
- A_addr
- B_addr
- C_addr
- ShouldJump
- A
- B
- C

### Execution columns (not committed, if we use logup*)

- A_value = m[A_addr]
- B_value = m[B_addr]
- C_value = m[C_addr]

### Transition constraints

A_flag * (A - A_arg)
B_flag * (B - fp)
C_flag * (C - C_arg)

(1 - A_flag) * ((fp + A_arg) - A_addr)
(1 - B_flag) * ((fp + B_arg) - B_addr)
(1 - C_flag) * ((fp + C_arg) - C_addr)

(1 - A_flag) * (A - A_value)
(1 - B_flag) * (A - B_value)
(1 - C_flag) * (A - C_value)

#### Add / Mul

ComputationAdd * (A + B - C)
ComputationMul * (A * B - C)

#### MemoryPointerEq

set A_flag = 0, B_flag = 1, C_flag = 0

MemoryPointerEq * ((A_value + B_arg) - B_addr)
MemoryPointerEq * (B_value - C_value)

#### Jumps

ShouldJump - Jump * A
ShouldJump * (next(pc) - C)
ShouldJump * (next(fp) - B)
(A - ShouldJump) * (next(pc) - (pc + 1))
(A - ShouldJump) * (next(fp) - fp)

## Alternative: Degree 3 AIR

### Execution columns (committed)

- pc
- fp
- A_addr
- B_addr
- C_addr

### Execution columns (not committed, if we use logup*)

- A_value = m[A_addr]
- B_value = m[B_addr]
- C_value = m[C_addr]

### Transition constraints

Aliases:
A = (A_flag * A_arg + (1 - A_flag) * A_value)
B = (B_flag * fp + (1 - B_flag) * B_value)
C = (C_flag * C_arg + (1 - C_flag) * C_value)

(1 - A_flag) * ((fp + A_arg) - A_addr)
(1 - B_flag) * ((fp + B_arg) - B_addr)
(1 - C_flag) * ((fp + C_arg) - C_addr)

#### Add / Mul

ComputationAdd * (A + B - C)
ComputationMul * (A * B - C)

#### MemoryPointerEq

set A_flag = 0, B_flag = 1, C_flag = 0

MemoryPointerEq * ((A_value + B_arg) - B_addr)
MemoryPointerEq * (B_value - C_value)

#### Jumps

Jump * A * (next(pc) - C)
Jump * A * (next(fp) - B)
A * (1 - Jump) * (next(pc) - (pc + 1))
A * (1 - Jump) * (next(fp) - fp)

TODO: Poseidon16, Poseidon24

## Memory layout:

[Bytecode][Zero Buffer][Public Input][Private Input][Runtime Memory]
convention: everything alligned by 8
Zero Buffer = 8 field elements