# ISA, Arithmetization (DRAFT)

## Logup*

If we use logup* (https://eprint.iacr.org/2025/946.pdf), we can avoid committing to the 22 bytecode columns (we only pay a constant cost in the size of the bytecode).

## TODO

the current ISA can still be simplified a lot (typically, we never multiply fp x fp, or constant x constant etc)

## Memory

Read-only (for now at least)

## Registers

- pc: Program Counter
- fp: Frame Pointer

Contrary to Cairo, no ap register (Allocation Pointer)

## Opcode format

Each opcode = 22 field elements:

- ArgA
- ArgB
- ArgC

### Binary columns

- A_IsConstant
- B_IsConstant
- C_IsConstant

- A_IsFp
- B_IsFp
- C_IsFp

- A_isMemAfterFp
- B_isMemAfterFp
- C_isMemAfterFp

- A_isDirectMem
- B_isDirectMem
- C_isDirectMem

### Instruction selector

- ComputationAdd
- ComputationMul
- MemoryPointer
- Jump
- Poseidon2_16
- Poseidon2_24
- ExtComputationAdd
- ExtComputationMul

## Execution columns (committed)

- pc
- fp
- A_addr
- B_addr
- C_addr
- ShouldJump
- A
- B
- C

## Execution columns (not committed, if we use logup*)

- A_value = m[A_addr]
- B_value = m[B_addr]
- C_value = m[C_addr]

## AIR transition constraints (degree 2)

ComputationAdd * (A_value + B_value - C_value)
ComputationMul * (A_value * B_value - C_value)

A_IsConstant * (A - ArgA)
B_IsConstant * (B - ArgB)
C_IsConstant * (C - ArgC)

A_IsFp * (A - fp)
B_IsFp * (B - fp)
C_IsFp * (C - fp)

(A_isMemAfterFp + A_isDirectMem) * (A - A_value)
A_isMemAfterFp * (A_address - (fp + ArgA))
A_isDirectMem * (A_address - ArgA)

(B_isMemAfterFp + B_isDirectMem) * (B - B_value)
B_isMemAfterFp * (B_address - (fp + ArgB))
B_isDirectMem * (B_address - ArgB)

(C_isMemAfterFp + C_isDirectMem) * (C - C_value)
C_isMemAfterFp * (C_address - (fp + ArgC))
C_isDirectMem * (C_address - ArgC)

MemoryPointer * (A_address - ArgA)
MemoryPointer * (B_address - A_value)
MemoryPointer * (C - B_value)

ShouldJump - Jump * A
ShouldJump * (next(pc) - B)
ShouldJump * (next(fp) - C)
(A - ShouldJump) * (next(pc) - (pc + 1))
(A - ShouldJump) * (next(fp) - fp)

TODO: Poseidon16, Poseidon24, ExtComputation