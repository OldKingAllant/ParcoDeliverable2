#ifndef PARCO_PARALLEL
#define PARCO_PARALLEL

#include "Defs.hpp"

/*
For the subsequent algorithms, we assume that the original
matrix is not distributed and only exists on the root process
*/

/// <summary>
/// Check if matrix is symmetryc by using MPI
/// </summary>
/// <param name="M">Square matrix</param>
/// <param name="N">Size of the sides</param>
/// <param name="comm_size">Number of processes</param>
/// <param name="root">Root process</param>
/// <returns>Symmetryc or not</returns>
bool checkSymMPI(const MatType* M, u32 N, int comm_size, int root);

/// <summary>
/// Transposes matrix M by dividing the matrix
/// in columns, distributing them as normal
/// floats and then coalesces the values
/// </summary>
/// <param name="M">Source</param>
/// <param name="T">Dest</param>
/// <param name="N">Side of matrix</param>
/// <param name="comm_size">Number of processes</param>
/// <param name="root">Root process</param>
void matTransposeMPI_TYPE(const MatType* M, MatType* T, u32 N, int comm_size, int root);

/// <summary>
/// Bcasts matrix on all the processes, then each process
/// transposes only a block of rows by
/// using a tiled transposition. Results are gathered 
/// in order
/// </summary>
/// <param name="M">Source</param>
/// <param name="T">Dest</param>
/// <param name="N">Side of matrix</param>
/// <param name="comm_size">Number of processes</param>
/// <param name="root">Root process</param>
void matTransposeMPI_BLOCK(const MatType* M, MatType* T, u32 N, int comm_size, int root);

/// <summary>
/// Bcasts matrix on all the processes, then each process
/// transposes only a block of rows by an
/// using oblivious transposition. Results are gathered 
/// in order
/// </summary>
/// <param name="M">Source</param>
/// <param name="T">Dest</param>
/// <param name="N">Side of matrix</param>
/// <param name="comm_size">Number of processes</param>
/// <param name="root">Root process</param>
//void matTransposeMPI_OBLIVIOUS(const MatType* M, MatType* T, u32 N, int comm_size, int root);

/// <summary>
/// Divides matrix in blocks, distributes them on all the
/// processes. Each process uses an in-place oblivious
/// transpose, by mirroring each block on its own
/// axis. After that, the blocks are gathered 
/// using gatherv
/// </summary>
/// <param name="M">Source</param>
/// <param name="T">Dest</param>
/// <param name="N">Side of matrix</param>
/// <param name="comm_size">Number of processes</param>
/// <param name="root">Root process</param>
void matTransposeMPI_ROOT(const MatType* M, MatType* T, u32 N, int comm_size, int root);

/*
For the last algorithms, the matrix is distributed by blocks of
rows on all processes
*/

/// <summary>
/// Transposes distributed matrix on all processes
/// </summary>
/// <param name="M"></param>
/// <param name="T"></param>
/// <param name="N"></param>
/// <param name="comm_size"></param>
/// <param name="root"></param>
void matTransposeMPI_DISTRIBUTED(const MatType* M, MatType* T, u32 N, int comm_size, int root);

/// <summary>
/// Check if a distributed matrix
/// is symmetric
/// </summary>
/// <param name="M"></param>
/// <param name="N"></param>
/// <param name="comm_size"></param>
/// <param name="root"></param>
/// <returns>You know what this is</returns>
bool checkSymMPI_DISTRIBUTED(const MatType* M, u32 N, int comm_size, int root);

#endif