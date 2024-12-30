#ifndef PARCO_UTILS
#define PARCO_UTILS

#include "Defs.hpp"

/// <summary>
/// Init random number generation
/// </summary>
void InitRandom();

/// <summary>
/// Allocates N*N floats and inits
/// them with random numbers
/// </summary>
/// <param name="N">Size of matrix side</param>
/// <returns>Pointer to memory</returns>
MatType* CreateMatrix(u32 N);

/// <summary>
/// Prints the matrix to the screen
/// </summary>
/// <param name="M">Matrix</param>
/// <param name="N">Size of matrix side</param>
void PrintMatrix(const MatType* M, u32 N);

/// <summary>
/// Check if T is the transpose of M
/// (and the opposite would still hold)
/// </summary>
/// <param name="M">Left matrix</param>
/// <param name="T">Right matrix</param>
/// <param name="N">Side of matrix</param>
/// <returns>True if one is the transpose of the other</returns>
bool checkTranspose(const MatType* M, const MatType* T, u32 N);

/// <summary>
/// Allocates and inits a matrix distributed on all
/// processes.
/// Each process will have N / N_PROC rows.
/// </summary>
/// <param name="N">Side of matrix</param>
/// <param name="world_size">Number of processes</param>
/// <returns>Pointer to part of the matrix</returns>
MatType* CreateDistributedMatrix(u32 N, int world_size);

/// <summary>
/// Checks if two distributed matrices are 
/// the transposition of the other
/// </summary>
/// <param name="M">Left</param>
/// <param name="T">Right</param>
/// <param name="N">Side of matrices</param>
/// <param name="world_size">Number of processes</param>
/// <param name="rank">Current rank</param>
/// <param name="root">Root process</param>
/// <returns>Whether the transposition is ok (only the root process will know this)</returns>
bool checkDistributedTranspose(const MatType* M, const MatType* T, u32 N, int world_size, int rank, 
	int root);

/// <summary>
/// Outputs distributed matrix on the
/// root's output stream
/// </summary>
/// <param name="M"></param>
/// <param name="N"></param>
/// <param name="world_size"></param>
/// <param name="rank"></param>
/// <param name="root"></param>
void PrintDistributedMatrix(const MatType* M, u32 N, int world_size, int rank,
	int root);

/// <summary>
/// Creates a distributed symmetric matrix
/// by setting each element x,y 
/// to its distance from the center of the matrix
/// </summary>
/// <param name="N"></param>
/// <param name="world_size"></param>
/// <returns></returns>
MatType* CreateDistributedSymmetric(u32 N, int world_size);

/// <summary>
/// Same but on only one process
/// </summary>
/// <param name="N"></param>
/// <returns></returns>
MatType* CreateSymmetric(u32 N);

u32 StringToU32(const char* str, u32 default_val);

#endif // !PARCO_UTILS
