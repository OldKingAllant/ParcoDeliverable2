#include "Utils.hpp"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <string>

#include <mpi.h>

void InitRandom() {
	srand(unsigned(time(0)));
}

MatType* CreateMatrix(u32 N) {
	MatType* the_matrix{ new MatType[N * N] };

	for (u32 idx = 0; idx < N * N; idx++) {
		the_matrix[idx] = float(rand() % int(VALUE_MAX));
	}

	return the_matrix;
}

void PrintMatrix(const MatType* M, u32 N) {
	for (u32 row = 0; row < N; row++) {
		for (u32 col = 0; col < N; col++) {
			std::cout << M[row * N + col] << " ";
		}
		std::cout << '\n';
	}
	//Use endl here to flush output
	std::cout << std::endl;
}

bool checkTranspose(const MatType* M, const MatType* T, u32 N) {
	bool is_symm = true;

	for (u32 row = 0; row < N; row++) {
		for (u32 col = 0; col < N; col++) {
			//do not use ifs and use a short-circuit logical and
			//(with opt enabled, the compiler will skip further
			//reads when is_symm becomes false)
			is_symm = is_symm && (M[row * N + col] == T[col * N + row]);
		}
	}

	return is_symm;
}

MatType* CreateDistributedMatrix(u32 N, int world_size) {
	//each process will have a block of N / world_size
	//rows
	const u32 rows_per_process{ N / u32(world_size) };

	MatType* the_matrix{ new MatType[N * rows_per_process] };

	for (u32 idx = 0; idx < N * rows_per_process; idx++) {
		the_matrix[idx] = float(rand() % int(VALUE_MAX));
	}

	return the_matrix;
}

bool checkDistributedTranspose(const MatType* M, const MatType* T, u32 N, int world_size, int rank,
	int root) {
	//the matrices are distributed, so we must
	//collect them
	MatType* collected_m{ new MatType[N * N] };
	MatType* collected_t{ new MatType[N * N] };

	const u32 rows_per_process{ N / u32(world_size) };

	//Gather all data

	MPI_Gather((void*)M, N * rows_per_process, MPI_FLOAT, (void*)collected_m,
		N * rows_per_process, MPI_FLOAT, root, MPI_COMM_WORLD);
	MPI_Gather((void*)T, N * rows_per_process, MPI_FLOAT, (void*)collected_t,
		N * rows_per_process, MPI_FLOAT, root, MPI_COMM_WORLD);

	bool result{ false };

	//check only on the root process
	if(rank == root)
		result = checkTranspose(collected_m, collected_t, N);

	delete[] collected_m;
	delete[] collected_t;

	return result;
}

void PrintDistributedMatrix(const MatType* M, u32 N, int world_size, int rank,
	int root) {
	MatType* collected_m{ new MatType[N * N] };

	u32 rows_per_process{ N / u32(world_size) };

	MPI_Gather((void*)M, N * rows_per_process, MPI_FLOAT, (void*)collected_m,
		N * rows_per_process, MPI_FLOAT, root, MPI_COMM_WORLD);

	if(rank == root)
		PrintMatrix(collected_m, N);

	delete[] collected_m;
}

MatType* CreateDistributedSymmetric(u32 N, int world_size) {
	const u32 rows_per_process{ N / u32(world_size) };
	const u32 middle{ N / 2 }; //will be even since N is power of two

	int rank{};
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MatType* the_matrix{ new MatType[N * rows_per_process] };

	for (u32 row = 0; row < rows_per_process; row++) {
		for (u32 col = 0; col < N; col++) {
			//compute absolute row inside the matrix
			const u32 abs_row = (u32(rank) * rows_per_process) + row;
			//compute deltas
			u32 delta_y = u32(std::abs(double(middle - abs_row)));
			u32 delta_x = u32(std::abs(double(middle - col)));
			//use pitagora's theorem and compute the modulo of the vector
			float dist = sqrtf(float(delta_x*delta_x) + float(delta_y*delta_y));
			the_matrix[row * N + col] = dist;
		}
	}

	return the_matrix;
}

MatType* CreateSymmetric(u32 N) {
	const u32 middle{ N / 2 }; //will be even since N is power of two

	MatType* the_matrix{ new MatType[N * N] };

	for (u32 row = 0; row < N; row++) {
		for (u32 col = 0; col < N; col++) {
			u32 delta_y = u32(std::abs(double(middle - row)));
			u32 delta_x = u32(std::abs(double(middle - col)));
			float dist = sqrtf(float(delta_x * delta_x) + float(delta_y * delta_y));
			the_matrix[row * N + col] = dist;
		}
	}

	return the_matrix;
}

u32 StringToU32(const char* str, u32 default_val) {
	std::string in{ str };

	u32 converted_value{};

	try {
		converted_value = u32(std::stoi(in));
	} catch(std::invalid_argument const&) {
		converted_value = default_val;
	}

	return converted_value;
}