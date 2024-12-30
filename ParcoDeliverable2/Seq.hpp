#ifndef PARCO_SEQ
#define PARCO_SEQ

#include "Defs.hpp"

/// <summary>
/// Computes whether the matrix is symmetric or not
/// </summary>
/// <param name="M"></param>
/// <param name="N"></param>
/// <returns>You know what this is</returns>
bool checkSym(const MatType* M, u32 N);

/// <summary>
/// Transpose matrix and places result in
/// memory pointed by T
/// </summary>
/// <param name="M"></param>
/// <param name="T"></param>
/// <param name="N"></param>
void matTranspose(MatType const* M, MatType* T, uint32_t N);

#endif // !PARCO_SEQ
