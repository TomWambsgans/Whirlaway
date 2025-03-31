/*
    Copyright (C) 2023 MrSpike63

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
Source: https://github.com/MrSpike63/vanity-eth-address
*/

#include <cinttypes>

int main()
{
    return 0;
}

// HASHING TYPE
#define KECCACK256 0
#define SHA3 1

#define HASH_TYPE KECCACK256

__device__ uint64_t rotate(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

__device__ uint64_t swap_endianness(uint64_t x)
{
    return ((x & 0x00000000000000FF) << 56) | ((x & 0x000000000000FF00) << 40) | ((x & 0x0000000000FF0000) << 24) | ((x & 0x00000000FF000000) << 8) | ((x & 0x000000FF00000000) >> 8) | ((x & 0x0000FF0000000000) >> 24) | ((x & 0x00FF000000000000) >> 40) | ((x & 0xFF00000000000000) >> 56);
}

__device__ uint32_t swap_endianness(uint32_t x)
{
    return ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24);
}

__constant__ uint64_t IOTA_CONSTANTS[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};

__device__ void block_permute(uint64_t *block)
{
    uint64_t C[5];
    uint64_t temp1, temp2;

    for (int t = 0; t < 24; t++)
    {
        C[0] = block[0] ^ block[1] ^ block[2] ^ block[3] ^ block[4];
        C[1] = block[5] ^ block[6] ^ block[7] ^ block[8] ^ block[9];
        C[2] = block[10] ^ block[11] ^ block[12] ^ block[13] ^ block[14];
        C[3] = block[15] ^ block[16] ^ block[17] ^ block[18] ^ block[19];
        C[4] = block[20] ^ block[21] ^ block[22] ^ block[23] ^ block[24];

        block[0] ^= C[4] ^ rotate(C[1], 1);
        block[1] ^= C[4] ^ rotate(C[1], 1);
        block[2] ^= C[4] ^ rotate(C[1], 1);
        block[3] ^= C[4] ^ rotate(C[1], 1);
        block[4] ^= C[4] ^ rotate(C[1], 1);
        block[5] ^= C[0] ^ rotate(C[2], 1);
        block[6] ^= C[0] ^ rotate(C[2], 1);
        block[7] ^= C[0] ^ rotate(C[2], 1);
        block[8] ^= C[0] ^ rotate(C[2], 1);
        block[9] ^= C[0] ^ rotate(C[2], 1);
        block[10] ^= C[1] ^ rotate(C[3], 1);
        block[11] ^= C[1] ^ rotate(C[3], 1);
        block[12] ^= C[1] ^ rotate(C[3], 1);
        block[13] ^= C[1] ^ rotate(C[3], 1);
        block[14] ^= C[1] ^ rotate(C[3], 1);
        block[15] ^= C[2] ^ rotate(C[4], 1);
        block[16] ^= C[2] ^ rotate(C[4], 1);
        block[17] ^= C[2] ^ rotate(C[4], 1);
        block[18] ^= C[2] ^ rotate(C[4], 1);
        block[19] ^= C[2] ^ rotate(C[4], 1);
        block[20] ^= C[3] ^ rotate(C[0], 1);
        block[21] ^= C[3] ^ rotate(C[0], 1);
        block[22] ^= C[3] ^ rotate(C[0], 1);
        block[23] ^= C[3] ^ rotate(C[0], 1);
        block[24] ^= C[3] ^ rotate(C[0], 1);

        temp1 = block[8];
        block[8] = rotate(block[1], 36);
        block[1] = rotate(block[15], 28);
        block[15] = rotate(block[18], 21);
        block[18] = rotate(block[13], 15);
        block[13] = rotate(block[7], 10);
        block[7] = rotate(block[11], 6);
        block[11] = rotate(block[2], 3);
        block[2] = rotate(block[5], 1);
        block[5] = rotate(block[6], 44);
        block[6] = rotate(block[21], 20);
        block[21] = rotate(block[14], 61);
        block[14] = rotate(block[22], 39);
        block[22] = rotate(block[4], 18);
        block[4] = rotate(block[10], 62);
        block[10] = rotate(block[12], 43);
        block[12] = rotate(block[17], 25);
        block[17] = rotate(block[23], 8);
        block[23] = rotate(block[19], 56);
        block[19] = rotate(block[3], 41);
        block[3] = rotate(block[20], 27);
        block[20] = rotate(block[24], 14);
        block[24] = rotate(block[9], 2);
        block[9] = rotate(block[16], 55);
        block[16] = rotate(temp1, 45);

        temp1 = block[0];
        temp2 = block[5];
        block[0] ^= (~block[5] & block[10]);
        block[5] ^= (~block[10] & block[15]);
        block[10] ^= (~block[15] & block[20]);
        block[15] ^= (~block[20] & temp1);
        block[20] ^= (~temp1 & temp2);

        temp1 = block[1];
        temp2 = block[6];
        block[1] ^= (~block[6] & block[11]);
        block[6] ^= (~block[11] & block[16]);
        block[11] ^= (~block[16] & block[21]);
        block[16] ^= (~block[21] & temp1);
        block[21] ^= (~temp1 & temp2);

        temp1 = block[2];
        temp2 = block[7];
        block[2] ^= (~block[7] & block[12]);
        block[7] ^= (~block[12] & block[17]);
        block[12] ^= (~block[17] & block[22]);
        block[17] ^= (~block[22] & temp1);
        block[22] ^= (~temp1 & temp2);

        temp1 = block[3];
        temp2 = block[8];
        block[3] ^= (~block[8] & block[13]);
        block[8] ^= (~block[13] & block[18]);
        block[13] ^= (~block[18] & block[23]);
        block[18] ^= (~block[23] & temp1);
        block[23] ^= (~temp1 & temp2);

        temp1 = block[4];
        temp2 = block[9];
        block[4] ^= (~block[9] & block[14]);
        block[9] ^= (~block[14] & block[19]);
        block[14] ^= (~block[19] & block[24]);
        block[19] ^= (~block[24] & temp1);
        block[24] ^= (~temp1 & temp2);

        block[0] ^= IOTA_CONSTANTS[t];
    }
}

#define KECCAK_RATE 136

#if HASH_TYPE == KECCACK256
#define DOMAIN_SEPARATOR 0x01
#endif
#if HASH_TYPE == SHA3
#define DOMAIN_SEPARATOR 0x06
#endif

// Helper inline function (or macro) to compute the lane index
__device__ inline size_t getLaneIndex(size_t i)
{
    size_t li = i / 8;
    return (li % 5) * 5 + (li / 5);
}

__device__ void keccak256(const uint8_t *input, size_t length, uint8_t *output)
{
    // 1. Initialize block state to zero
    uint64_t block[25];
#pragma unroll 25
    for (int i = 0; i < 25; i++)
    {
        block[i] = 0ULL;
    }

    size_t offset = 0;

    // 2. Absorb data in full 136-byte (KECCAK_RATE) chunks
    while (length >= KECCAK_RATE)
    {
        for (int i = 0; i < KECCAK_RATE; i++)
        {
            size_t lane_index = getLaneIndex(i);
            size_t byte_offset = i % 8;
            block[lane_index] ^= static_cast<uint64_t>(input[offset + i]) << (8 * byte_offset);
        }

        // Permute the state
        block_permute(block);

        offset += KECCAK_RATE;
        length -= KECCAK_RATE;
    }

    // 3. Absorb leftover bytes (< 136)
    for (size_t i = 0; i < length; i++)
    {
        size_t lane_index = getLaneIndex(i);
        size_t byte_offset = i % 8;
        block[lane_index] ^= static_cast<uint64_t>(input[offset + i]) << (8 * byte_offset);
    }

    // 4. Domain separation: set the special byte right after the input
    {
        size_t lane_index = getLaneIndex(length);
        size_t byte_offset = length % 8;
        block[lane_index] ^= static_cast<uint64_t>(DOMAIN_SEPARATOR) << (8 * byte_offset);
    }

    // 5. Append the 0x80 bit at the end of the 136-byte block
    //    This is the standard SHA-3 padding that sets the highest bit
    //    in the last position within the rate portion.
    block[8] ^= 0x8000000000000000ULL;

    // 6. One final permutation
    block_permute(block);

    // 7. Squeeze out 32 bytes (= 256 bits)
    for (int i = 0; i < 32; i++)
    {
        size_t lane_index = getLaneIndex(i);
        size_t byte_offset = i % 8;
        output[i] = static_cast<uint8_t>((block[lane_index] >> (8 * byte_offset)) & 0xFF);
    }
}

// Kernel that processes multiple inputs in parallel
extern "C" __global__ void batch_keccak256(
    const uint8_t *inputs,            // Packed input data
    const uint32_t num_inputs,          // Number of inputs to process
    const uint32_t input_length,        // Length of each input
    const uint32_t input_packed_length, // Length of each input
    uint8_t *outputs                  // Output buffer for hashes
)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_inputs)
    {
        // Calculate offset in the input buffer
        const uint8_t *input = inputs + (idx * input_packed_length);

        // Calculate offset in the output buffer (32 bytes per hash)
        uint8_t *output = outputs + (idx * 32);

        // Compute the hash

        keccak256(input, input_length, output);
    }
}
