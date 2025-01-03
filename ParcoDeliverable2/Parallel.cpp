#include "Parallel.hpp"

#include <mpi.h>
#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <cmath>

#include "Utils.hpp"

void matTransposeMPI_TYPE(const MatType* M, MatType* T, u32 N, int comm_size, int root) {
	const u32 COLUMNS_PROC = N / comm_size;
	const u32 ELEM_PROC = N * COLUMNS_PROC;

	//A block of columns
	MatType* temp{ new MatType[ELEM_PROC]{} };

	MPI_Datatype column_temp{}, column{};
	MPI_Type_vector(N, 1, N, MPI_FLOAT, &column_temp); //N floats, each with stride N
	MPI_Type_create_resized(column_temp, 0, sizeof(MatType), &column); //Resize to represent a column
	MPI_Type_commit(&column);

	//Scatter columns to all processes, but store them as normal floats
	MPI_Scatter((void*)M, COLUMNS_PROC, column, (void*)temp, ELEM_PROC, MPI_FLOAT, root,
		MPI_COMM_WORLD);

	//If we gather the values immediately as-is, we get the transposed matrix
	MPI_Gather((void*)temp, ELEM_PROC, MPI_FLOAT, (void*)T, ELEM_PROC,
		MPI_FLOAT, root, MPI_COMM_WORLD);

	delete[] temp;

	MPI_Type_free(&column_temp);
	MPI_Type_free(&column);
}

//Works the same way as in the first deliverable
void Transpose4x4_Aligned(MatType const* src, MatType* dst,
	uint32_t row, uint32_t col, uint32_t N) {
	__m128 row1{}, row2{}, row3{}, row4{};
	__m128 t1{}, t2{}, t3{}, t4{};

	row1 = _mm_load_ps(&src[col * N + row]);
	row2 = _mm_load_ps(&src[(col + 1) * N + row]);
	row3 = _mm_load_ps(&src[(col + 2) * N + row]);
	row4 = _mm_load_ps(&src[(col + 3) * N + row]);

	t1 = _mm_shuffle_ps(row1, row3, 0b00000000);
	t2 = _mm_shuffle_ps(row2, row4, 0b00000000);
	t1 = _mm_blend_ps(t1, t2, 0b1010);

	t2 = _mm_shuffle_ps(row1, row3, 0b00010001);
	t3 = _mm_shuffle_ps(row2, row4, 0b01000100);
	t2 = _mm_blend_ps(t2, t3, 0b1010);

	t3 = _mm_shuffle_ps(row1, row3, 0b00100010);
	t4 = _mm_shuffle_ps(row2, row4, 0b10001000);
	t3 = _mm_blend_ps(t3, t4, 0b1010);

	t4 = _mm_shuffle_ps(row1, row3, 0b00110011);
	row1 = _mm_shuffle_ps(row2, row4, 0b11001100);
	t4 = _mm_blend_ps(t4, row1, 0b1010);

	_mm_store_ps(&dst[row * N + col], t1);
	_mm_store_ps(&dst[(row + 1) * N + col], t2);
	_mm_store_ps(&dst[(row + 2) * N + col], t3);
	_mm_store_ps(&dst[(row + 3) * N + col], t4);
}

/// <summary>
/// Performs in place transposition 
/// of 4x4 block inside a bigger matrix
/// </summary>
/// <param name="M">Matrix</param>
/// <param name="row">Absolute row</param>
/// <param name="col">Absolute column</param>
/// <param name="N">Size of matrix</param>
void Transpose4x4_InPlace(MatType* M,
	uint32_t row, uint32_t col, uint32_t N) {
	__m128 row1{}, row2{}, row3{}, row4{};
	__m128 t1{}, t2{}, t3{}, t4{};

	////////////////////////////////////////

	//First part is the same as for the 
	//normal sse transpose
	row1 = _mm_load_ps(&M[col * N + row]);
	row2 = _mm_load_ps(&M[(col + 1) * N + row]);
	row3 = _mm_load_ps(&M[(col + 2) * N + row]);
	row4 = _mm_load_ps(&M[(col + 3) * N + row]);

	t1 = _mm_shuffle_ps(row1, row3, 0b00000000);
	t2 = _mm_shuffle_ps(row2, row4, 0b00000000);
	t1 = _mm_blend_ps(t1, t2, 0b1010);

	t2 = _mm_shuffle_ps(row1, row3, 0b00010001);
	t3 = _mm_shuffle_ps(row2, row4, 0b01000100);
	t2 = _mm_blend_ps(t2, t3, 0b1010);

	t3 = _mm_shuffle_ps(row1, row3, 0b00100010);
	t4 = _mm_shuffle_ps(row2, row4, 0b10001000);
	t3 = _mm_blend_ps(t3, t4, 0b1010);

	t4 = _mm_shuffle_ps(row1, row3, 0b00110011);
	row1 = _mm_shuffle_ps(row2, row4, 0b11001100);
	t4 = _mm_blend_ps(t4, row1, 0b1010);

	//Now the transposed block resides in the temp
	//variables, we can reuse the rows

	//////////////////////////////////////////

	//Retrieve the other block
	row1 = _mm_load_ps(&M[row * N + col]);
	row2 = _mm_load_ps(&M[(row + 1) * N + col]);
	row3 = _mm_load_ps(&M[(row + 2) * N + col]);
	row4 = _mm_load_ps(&M[(row + 3) * N + col]);

	//Now we can store the other 4 rows
	//safely, without losing data
	_mm_store_ps(&M[row * N + col], t1);
	_mm_store_ps(&M[(row + 1) * N + col], t2);
	_mm_store_ps(&M[(row + 2) * N + col], t3);
	_mm_store_ps(&M[(row + 3) * N + col], t4);

	//Repeat transposition
	t1 = _mm_shuffle_ps(row1, row3, 0b00000000);
	t2 = _mm_shuffle_ps(row2, row4, 0b00000000);
	t1 = _mm_blend_ps(t1, t2, 0b1010);

	t2 = _mm_shuffle_ps(row1, row3, 0b00010001);
	t3 = _mm_shuffle_ps(row2, row4, 0b01000100);
	t2 = _mm_blend_ps(t2, t3, 0b1010);

	t3 = _mm_shuffle_ps(row1, row3, 0b00100010);
	t4 = _mm_shuffle_ps(row2, row4, 0b10001000);
	t3 = _mm_blend_ps(t3, t4, 0b1010);

	t4 = _mm_shuffle_ps(row1, row3, 0b00110011);
	row1 = _mm_shuffle_ps(row2, row4, 0b11001100);
	t4 = _mm_blend_ps(t4, row1, 0b1010);

	_mm_store_ps(&M[col * N + row], t1);
	_mm_store_ps(&M[(col + 1) * N + row], t2);
	_mm_store_ps(&M[(col + 2) * N + row], t3);
	_mm_store_ps(&M[(col + 3) * N + row], t4);
}

void matTransposeMPI_BLOCK(const MatType* M, MatType* T, u32 N, int comm_size, int root) {
	//Allocate space for bcast matrix
	MatType* temp{ new MatType[N * N]{} };
	//Rows that each process must transpose
	const u32 ROWS_PROC = N / comm_size;

	int rank{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Bcast matrix to all processes
	if (rank == root) {
		MPI_Bcast((void*)M, N * N, MPI_FLOAT, root, MPI_COMM_WORLD);
	}
	else {
		MPI_Bcast((void*)T, N * N, MPI_FLOAT, root, MPI_COMM_WORLD);
	}

	//Select correct source pointer
	const MatType* src_mat = (rank == root) ? M : T;
	//Compute start, end rows
	const u32 START = rank * ROWS_PROC;
	const u32 END = START + ROWS_PROC;

	//Tile size for transposition
	const u32 BLOCK_SIZE = 16;

	for (u32 row = START; row < END; row += BLOCK_SIZE) {
		for (u32 col = 0; col < N; col += BLOCK_SIZE) {

			u32 row_limit = std::min(row + BLOCK_SIZE, END);
			u32 col_limit = std::min(col + BLOCK_SIZE, N);

			//Check if the sse transpose would overshoot 
			if ((row_limit - row) % 4 == 0 && (col_limit - col) % 4 == 0) {
				//It doesn't divide 16x16 block in 4x4 blocks and use sse
				for (u32 block_row = row; block_row < row_limit; block_row += 4) {
					for (u32 block_col = col; block_col < col_limit; block_col += 4) {
						Transpose4x4_Aligned(src_mat, temp, block_row,
							block_col, N);
					}
				}
			}
			else {
				//Must use normal transposition
				for (u32 block_row = row; block_row < row_limit; block_row++) {
					for (u32 block_col = col; block_col < col_limit; block_col++) {
						temp[block_row * N + block_col] =
							src_mat[block_col * N + block_row];
					}
				}
			}

		}
	}

	const u32 NUM_ELEMENTS = ROWS_PROC * N;
	const u32 OFFSET = NUM_ELEMENTS * rank;

	//Since the row blocks are transposed in order based on
	//the rank, we can simply gather all blocks
	//and we will still get the correct result
	MPI_Gather((void*)(temp + OFFSET), NUM_ELEMENTS, MPI_FLOAT, (void*)T,
		NUM_ELEMENTS, MPI_FLOAT, root, MPI_COMM_WORLD);

	delete[] temp;
}

//Same as D1
void ObliviousTransposeImpl(MatType const* M, MatType* T, uint32_t N,
	uint32_t N_rem, uint32_t col_offset, uint32_t row_offset) {
	if (N_rem <= 32) {
		//End condition, size is small enough

		if (N_rem % 4 == 0) { //Use sse
			//We trust the caller and expect the matrix to be
			//16-bytes aligned
			for (uint32_t row_idx = 0; row_idx < N_rem; row_idx += 4) {
				for (uint32_t col_idx = 0; col_idx < N_rem; col_idx += 4) {
					Transpose4x4_Aligned(M, T, row_offset + row_idx,
						col_offset + col_idx, N);
				}
			}
		}
		else {
			//Use normal transpose
			for (uint32_t row_idx = 0; row_idx < N_rem; row_idx++) {
				for (uint32_t col_idx = 0; col_idx < N_rem; col_idx++) {
					//(row_offset + row_idx) * N + (col_offset + col_idx)
					T[(row_offset + row_idx) * N + (col_offset + col_idx)] =
						M[(col_idx + col_offset) * N + (row_offset + row_idx)];
				}
			}
		}

	}
	else {
		//Divide and conquer in 4 submatrices
		uint32_t half_size = N_rem / 2;
		ObliviousTransposeImpl(M, T, N, half_size, col_offset, row_offset);
		ObliviousTransposeImpl(M, T, N, half_size, col_offset + half_size, row_offset);

		ObliviousTransposeImpl(M, T, N, half_size, col_offset, row_offset + half_size);
		ObliviousTransposeImpl(M, T, N, half_size, col_offset + half_size, row_offset + half_size);

		if (N_rem & 1) {
			//Size is not even, must transpose last row and column
			for (uint32_t row_idx = 0; row_idx < N_rem; row_idx++) {
				T[(col_offset + N_rem - 1) * N + (row_offset + row_idx)] =
					M[(row_offset + row_idx) * N + (col_offset + N_rem - 1)];

				T[(col_offset + row_idx) * N + (row_offset + N_rem - 1)] =
					M[(row_offset + N_rem - 1) * N + (col_offset + row_idx)];
			}
		}
	}
}

/// <summary>
/// Perform oblivious transposition in-place
/// </summary>
/// <param name="M"></param>
/// <param name="N"></param>
/// <param name="N_rem"></param>
/// <param name="col_offset"></param>
/// <param name="row_offset"></param>
void ObliviousTransposeInPlace(MatType* M, uint32_t N,
	uint32_t N_rem, uint32_t col_offset, uint32_t row_offset) {
	if (col_offset > row_offset) //Check if we are above the diagonal
		return; //leave if yes

	if (N_rem <= 32) {
		//End condition, size is small enough

		if (N_rem % 4 == 0) { 
			//Compute global end positions
			uint32_t end_row = row_offset + N_rem;
			uint32_t end_col = col_offset + N_rem;

			for (uint32_t row_idx = row_offset; row_idx < end_row; row_idx += 4) {
				for (uint32_t col_idx = col_offset; col_idx < end_col; col_idx += 4) {

					//Check if we are below the diagonal (avoid
					//overwriting previously transposed values)
					if (row_idx >= col_idx) {
						//Check if the 4x4 block transpose does overshoot
						//the diagonal
						if (col_idx + 4 > row_idx) {
							uint32_t diag = row_idx;
							//Transpose one by one, from the current row
							//up until 4 rows ahead, from the current column
							//to the diagonal
							for (uint32_t curr_diag = diag; curr_diag < diag + 4; curr_diag++) {
								for (uint32_t sub_col = col_idx; sub_col < curr_diag; sub_col++) {
									std::swap(M[curr_diag * N + sub_col],
										M[sub_col * N + curr_diag]);
								}
							}
							
						}
						else {
							//It doesn't, we can safely use sse
							Transpose4x4_InPlace(M, row_idx, col_idx, N);
						}
					}
					
				}
			}
		}
		else {
			//Use normal transpose
			for (uint32_t row_idx = 0; row_idx < N_rem; row_idx++) {
				for (uint32_t col_idx = 0; col_idx < N_rem; col_idx++) {

					if (row_offset + row_idx >= col_offset + col_idx) {
						std::swap(M[(row_offset + row_idx) * N + (col_offset + col_idx)],
							M[(col_idx + col_offset) * N + (row_offset + row_idx)]);
					}

				}
			}
		}

	}
	else {
		//Divide and conquer in 4 submatrices
		uint32_t half_size = N_rem / 2;
		ObliviousTransposeInPlace(M, N, half_size, col_offset, row_offset);
		ObliviousTransposeInPlace(M, N, half_size, col_offset + half_size, row_offset);

		ObliviousTransposeInPlace(M, N, half_size, col_offset, row_offset + half_size);
		ObliviousTransposeInPlace(M, N, half_size, col_offset + half_size, row_offset + half_size);
	}
}

void matTransposeMPI_ROOT(const MatType* M, MatType* T, u32 N, int comm_size, int root) {
	//TODO: Use in-place transpose for faster execution + handle non-integer sqrt(comm_size)
	
	u32 inflated_comm_size{ u32(comm_size) };
	//Make sure that comm_size is a power of two with integer
	//square root
	if (u32(ceil(sqrt(inflated_comm_size))) != u32(sqrt(inflated_comm_size))) {
		//Given 2^x, if x is odd, sqrt is not integer
		//Get the number as 2^2x
		inflated_comm_size <<= 1;
	}

	//The number of blocks cannot be higher than N
	if (inflated_comm_size > N) {
		//We just use another algorithm
		matTransposeMPI_TYPE(M, T, N, comm_size, root);
		return;
	}

	const u32 NUM_SIDE_BLOCKS{ u32(ceil(sqrt(inflated_comm_size))) };
	const u32 ROOT_NUM_PROC{ u32(sqrt(comm_size)) };
	const u32 ROWS_PROC{ N / NUM_SIDE_BLOCKS };
	
	MatType* temp_matrix{ new MatType[ROWS_PROC * ROWS_PROC]{} };

	MPI_Datatype block{}, temp{};

	int rank{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Create datatype that represents a sub-matrix
	MPI_Type_vector(ROWS_PROC, ROWS_PROC, N, MPI_FLOAT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(MatType), &block);
	MPI_Type_commit(&block);

	//Number of blocks to send to each process
	int* counts{ new int[NUM_SIDE_BLOCKS*NUM_SIDE_BLOCKS] {} };
	//Displacemnts from the start of the matrix, one for
	//each process (in number of elements)
	int* displs{ new int[NUM_SIDE_BLOCKS*NUM_SIDE_BLOCKS] {} };
	//Displacements inside the final transposed matrix,
	//one for each process
	int* gather_displs{ new int[NUM_SIDE_BLOCKS * NUM_SIDE_BLOCKS] {} };

	for (u32 row_block = 0; row_block < NUM_SIDE_BLOCKS; row_block++) {
		for (u32 col_block = 0; col_block < NUM_SIDE_BLOCKS; col_block++) {
			counts[row_block * NUM_SIDE_BLOCKS + col_block] = 1;
			//Blocks are taken incrementally in row major fashion
			displs[row_block * NUM_SIDE_BLOCKS + col_block] =
				(row_block * ROWS_PROC * N) + (col_block * ROWS_PROC);
			//Mirrored blocks are stored in column-major 
			gather_displs[row_block * NUM_SIDE_BLOCKS + col_block] =
				(col_block * ROWS_PROC * N) + (row_block * ROWS_PROC);
		}
	}

	for (u32 curr_block = 0; curr_block < NUM_SIDE_BLOCKS * NUM_SIDE_BLOCKS;
		curr_block += u32(comm_size)) {
		//Scatter blocks to all processes (one for each process)
		MPI_Scatterv((void*)M, counts + curr_block, displs + curr_block, block, 
			(void*)temp_matrix, ROWS_PROC * ROWS_PROC,
			MPI_FLOAT, root, MPI_COMM_WORLD);

		//Perform in-place transpose
		ObliviousTransposeInPlace(temp_matrix, ROWS_PROC,
			ROWS_PROC, 0, 0);

		//Gather mirrored blocks and reorder them as needed
		MPI_Gatherv((void*)temp_matrix, ROWS_PROC * ROWS_PROC, MPI_FLOAT,
			(void*)T, counts + curr_block, gather_displs + curr_block,
			block, root, MPI_COMM_WORLD);
	}

	delete[] counts;
	delete[] displs;
	delete[] gather_displs;
	delete[] temp_matrix;

	MPI_Type_free(&temp);
	MPI_Type_free(&block);
}

void matTransposeMPI_DISTRIBUTED(const MatType* M, MatType* T, u32 N, int comm_size, int root) {
	const u32 ROWS_PROC{ N / u32(comm_size) };
	
	MPI_Datatype block{}, temp{};

	int rank{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Create datatype that represents a sub-matrix
	MPI_Type_vector(ROWS_PROC, ROWS_PROC, N, MPI_FLOAT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(MatType), &block);
	MPI_Type_commit(&block);

	const u32 NUM_BLOCK_PROC{ u32(comm_size) };

	//Divide block of rows and 
	//compute the offset for each block

	int* counts{ new int[NUM_BLOCK_PROC] };
	int* send_disps{ new int[NUM_BLOCK_PROC] };
	int* recv_disps{ new int[NUM_BLOCK_PROC] };

	for (u32 curr_block = 0; curr_block < NUM_BLOCK_PROC; curr_block++) {
		counts[curr_block] = 1;
		send_disps[curr_block] = curr_block * ROWS_PROC;
		recv_disps[curr_block] = curr_block * ROWS_PROC;
	}

	//Using alltoallv mirror each block
	//wrt the diagonal

	MPI_Alltoallv((void*)M, counts, send_disps, block, (void*)T,
		counts, recv_disps, block, MPI_COMM_WORLD);

	//However, each individual block is not (in itself)
	//transposed, so we do it here
	for (u32 curr_block = 0; curr_block < NUM_BLOCK_PROC; curr_block++) {
		ObliviousTransposeInPlace(T + (curr_block * ROWS_PROC), N, ROWS_PROC, 0, 0);
	}

	delete[] counts;
	delete[] send_disps;
	delete[] recv_disps;

	MPI_Type_free(&temp);
	MPI_Type_free(&block);
}

bool checkBlockSymmetry(const MatType* M, const MatType* T, u32 N, u32 row_offset) {
	bool is_mirror{ true };

	static constexpr u32 BLOCK_SIZE = 16;

	for (u32 row = 0; row < N; row += BLOCK_SIZE) {
		for (u32 col = 0; col < N; col += BLOCK_SIZE) {
			u32 row_limit = std::min(row + BLOCK_SIZE, N);
			u32 col_limit = std::min(col + BLOCK_SIZE, N);

			for (u32 block_row = row; block_row < row_limit; block_row++) {
				for (u32 block_col = col; block_col < col_limit; block_col++) {
					//Here we are using & to force the compiler
					//to produce code that always reads from the matrix,
					//irrespective of the value of is_symm
					is_mirror = is_mirror &&
						(M[block_row * row_offset + block_col]
							== T[block_col * row_offset + block_row]);
				}
			}
		}
	}

	return is_mirror;
}

bool checkSymMPI(const MatType* M, u32 N, int comm_size, int root) {
	u32 inflated_comm_size{ u32(comm_size) };
	//Make sure that comm_size is a power of two with integer
	//square root
	if (u32(ceil(sqrt(inflated_comm_size))) != u32(sqrt(inflated_comm_size))) {
		//Given 2^x, if x is odd, sqrt is not integer
		//Get the number as 2^2x
		inflated_comm_size <<= 1;
	}

	if (inflated_comm_size > N) {
		bool is_symm = true;

		for (u32 row = 0; row < N; row++) {
			for (u32 col = row + 1; col < N; col++) {
				is_symm = bool(u32(is_symm) & u32(M[row * N + col] == M[col * N + row]));
			}
		}

		return is_symm;
	}

	const u32 NUM_SIDE_BLOCKS{ u32(ceil(sqrt(inflated_comm_size))) };
	const u32 ROWS_PROC{ N / NUM_SIDE_BLOCKS };
	const u32 BLOCK_SIZE{ ROWS_PROC * ROWS_PROC };

	MatType* block_1{ new MatType[BLOCK_SIZE]{} };
	MatType* block_2{ new MatType[BLOCK_SIZE]{} };

	int rank{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Datatype block{}, temp{};

	MPI_Type_vector(ROWS_PROC, ROWS_PROC, N, MPI_FLOAT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(MatType), &block);
	MPI_Type_commit(&block);

	int* counts{ new int[NUM_SIDE_BLOCKS * NUM_SIDE_BLOCKS] {} };
	int* above_displs{ new int[NUM_SIDE_BLOCKS * NUM_SIDE_BLOCKS] {} };
	int* below_displs{ new int[NUM_SIDE_BLOCKS * NUM_SIDE_BLOCKS] {} };

	//used to limit the effective number of blocks
	//we need to send to each process
	u32 num_blocks_to_send{ 0 };

	for (u32 row_block = 0; row_block < NUM_SIDE_BLOCKS; row_block++) {
		for (u32 col_block = row_block; col_block < NUM_SIDE_BLOCKS; col_block++) {
			counts[num_blocks_to_send] = 1;
			//position of blocks above the diagonal
			above_displs[num_blocks_to_send] =
				(row_block * ROWS_PROC * N) + (col_block * ROWS_PROC);
			//position of transposed blocks
			below_displs[num_blocks_to_send] =
				(col_block * ROWS_PROC * N) + (row_block * ROWS_PROC);

			num_blocks_to_send += 1;
		}
	}

	//used to accumulate on only one process
	bool is_symm{ true };

	for (u32 curr_block_num = 0; curr_block_num < num_blocks_to_send; curr_block_num += 
		u32(comm_size)) {
		//check if the current process has any work to do
		u32 recv_count{ counts[curr_block_num + u32(rank)] > 0 ? BLOCK_SIZE : 0 };

		//scatter blocks above the diagonal (diagonal included)
		MPI_Scatterv((void*)M, counts + curr_block_num, 
			above_displs + curr_block_num, block, (void*)block_1,
			recv_count, MPI_FLOAT, root, MPI_COMM_WORLD);

		//scatter transposed blocks (diagonal included)
		MPI_Scatterv((void*)M, counts + curr_block_num,
			below_displs + curr_block_num, block, (void*)block_2,
			recv_count, MPI_FLOAT, root, MPI_COMM_WORLD);

		bool is_symm_temp{ true };

		//if the current process has received the two blocks,
		//check if they are mirrors of each other
		if (counts[curr_block_num + u32(rank)] > 0) {
			is_symm_temp = checkBlockSymmetry(block_1, block_2, ROWS_PROC,
				ROWS_PROC);
		}

		is_symm = is_symm && is_symm_temp;
	}

	delete[] block_1;
	delete[] block_2;

	delete[] counts;
	delete[] above_displs;
	delete[] below_displs;

	MPI_Type_free(&temp);
	MPI_Type_free(&block);

	bool is_symm_final{ false };

	MPI_Reduce((void*)&is_symm, (void*)&is_symm_final, 1,
		MPI_C_BOOL, MPI_LAND, root, MPI_COMM_WORLD);

	return is_symm_final;
}

bool checkSymMPI_DISTRIBUTED(const MatType* M, u32 N, int comm_size, int root) {
	const u32 ROWS_PROC{ N / u32(comm_size) };

	MatType* temp_matrix{ new MatType[N * ROWS_PROC]{} };

	MPI_Datatype block{}, temp{};

	int rank{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//Create datatype that represents a sub-matrix
	MPI_Type_vector(ROWS_PROC, ROWS_PROC, N, MPI_FLOAT, &temp);
	MPI_Type_create_resized(temp, 0, sizeof(MatType), &block);
	MPI_Type_commit(&block);

	const u32 NUM_BLOCK_PROC{ u32(comm_size) };

	//Same principle for distributed transpose
	int* counts{ new int[NUM_BLOCK_PROC] };
	int* send_disps{ new int[NUM_BLOCK_PROC] };
	int* recv_disps{ new int[NUM_BLOCK_PROC] };

	for (u32 curr_block = 0; curr_block < NUM_BLOCK_PROC; curr_block++) {
		counts[curr_block] = 1;
		send_disps[curr_block] = curr_block * ROWS_PROC;
		recv_disps[curr_block] = curr_block * ROWS_PROC;
	}

	MPI_Alltoallv((void*)M, counts, send_disps, block, (void*)temp_matrix,
		counts, recv_disps, block, MPI_COMM_WORLD);

	//For each received block and original,
	//check if one is the transposition of the other
	bool per_proc_is_symm{ true };

	for (u32 curr_block = 0; curr_block < NUM_BLOCK_PROC; curr_block++) {
		//ObliviousTransposeInPlace(temp_matrix + (curr_block * ROWS_PROC), N, ROWS_PROC, 0, 0);
		per_proc_is_symm = per_proc_is_symm && 
			checkBlockSymmetry(M + curr_block * ROWS_PROC, 
				temp_matrix + curr_block * ROWS_PROC, ROWS_PROC, N);
	}

	/*bool per_proc_is_symm = (memcmp((void*)M, (void*)temp_matrix,
		sizeof(MatType) * N * ROWS_PROC) == 0);*/

	//Accumulate result s.t. if one of the blocks was incorrect,
	//the result is false
	bool result{ false };

	MPI_Reduce((void*)&per_proc_is_symm, (void*)&result, 1, MPI_C_BOOL,
		MPI_LAND, root, MPI_COMM_WORLD);

	delete[] counts;
	delete[] send_disps;
	delete[] recv_disps;

	delete[] temp_matrix;

	MPI_Type_free(&temp);
	MPI_Type_free(&block);

	return result;
}