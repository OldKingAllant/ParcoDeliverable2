#include "ParcoDeliverable2.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Bench.h"
#include "Utils.hpp"
#include "Seq.hpp"
#include "Parallel.hpp"

static constexpr u32 CONST_N = 4096;

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int curr_rank{}, world_size{};
	MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	u32 N{ CONST_N };

	std::string work_dir{"./"};

	if (argc > 1) {
		N = StringToU32(argv[1], N);
		
		if (argc > 2) {
			work_dir = std::string(argv[2]);
		}
	}

	if ((world_size & (world_size - 1)) && curr_rank == 0) {
		std::cerr << "Expected comm size as a power of two!" << std::endl;
		std::exit(0);
	}

	if (world_size > int(N) && curr_rank == 0) {
		std::cerr << "Too many processes" << std::endl;
		std::exit(0);
	}

	std::ofstream out{};

	if (curr_rank == 0) {
		std::stringstream file_name_builder{};

		file_name_builder <<
			work_dir <<
			"/bench_" <<
			world_size <<
			"_" <<
			N <<
			".txt";

		out.open(file_name_builder.str(), std::ios::out);
	}

	std::cout << "Rank: " << curr_rank << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

	MatType* base_matrix{ nullptr };

	const u32 TOT_SIZE = N * N;

	InitRandom();

	if (curr_rank == 0) {
		base_matrix = CreateMatrix(N);

		MatType* seq_transposed{ new MatType[TOT_SIZE]{} };
		
		Benchmark([=]() { matTranspose(base_matrix, seq_transposed, N); },
			"Base transpose", 10, out, curr_rank, 0);

		if (!checkTranspose(base_matrix, seq_transposed, N))
			std::cout << "Transposition failed" << std::endl;

		delete[] seq_transposed;
	}

	MatType* transposed1{ new MatType[TOT_SIZE]{} };
	MatType* transposed2{ new MatType[TOT_SIZE]{} };
	MatType* transposed3{ new MatType[TOT_SIZE]{} };
	MatType* transposed4{ new MatType[TOT_SIZE]{} };

	/*Benchmark([=]() { matTransposeMPI_TYPE(base_matrix, transposed1, N, world_size, 0); },
		"MPI Column transpose", 10, out, curr_rank, 0);

	if(curr_rank == 0 && !checkTranspose(base_matrix, transposed1, N))
		std::cout << "MPI Column Transposition failed" << std::endl;*/

	Benchmark([=]() { matTransposeMPI_BLOCK(base_matrix, transposed2, N, world_size, 0); },
		"MPI Block transpose", 10, out, curr_rank, 0);

	if (curr_rank == 0 && !checkTranspose(base_matrix, transposed2, N))
		std::cout << "MPI Block Transposition failed" << std::endl;

	/*Benchmark([=]() { matTransposeMPI_OBLIVIOUS(base_matrix, transposed3, N, world_size, 0); },
		"MPI Oblivious transpose", 10, out, curr_rank, 0);

	if (curr_rank == 0 && !checkTranspose(base_matrix, transposed3, N))
		std::cout << "MPI Oblivious Transposition failed" << std::endl;*/

	Benchmark([=]() { matTransposeMPI_ROOT(base_matrix, transposed4, N, world_size, 0); },
		"MPI Root transpose", 10, out, curr_rank, 0);

	if (curr_rank == 0 && !checkTranspose(base_matrix, transposed4, N))
		std::cout << "MPI Root Transposition failed" << std::endl;

	delete[] transposed1;
	delete[] transposed2;
	delete[] transposed3;
	delete[] transposed4;

	/////////////////////////
	//Try distributed matrix
	MatType* distributed_matrix{ CreateDistributedMatrix(N, world_size) };
	MatType* transposed5{ new MatType[(N / world_size) * N] };
	
	Benchmark([=]() { matTransposeMPI_DISTRIBUTED(distributed_matrix, transposed5, N, world_size, 0); },
		"Distributed transpose", 10, out, curr_rank, 0);

	bool is_transposed = checkDistributedTranspose(distributed_matrix, transposed5,
		N, world_size, curr_rank, 0);

	if (!is_transposed && curr_rank == 0) {
		std::cout << "Distributed transpose failed" << std::endl;
	}

	/////////////////////////////////

	MatType* symm1{ nullptr };

	if(curr_rank == 0)
		symm1 = CreateSymmetric(N);

	MatType* symm2 = CreateDistributedSymmetric(N, world_size);

	if (curr_rank == 0) {
		bool is_symm1 = Benchmark([=]() { return checkSym(symm1, N); },
			"Base symm", 10, out, curr_rank, 0);

		if (!is_symm1) {
			std::cout << "Seq. symmetry check failed on symm. matrix" << std::endl;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	bool is_symm2 = Benchmark([=]() { return checkSymMPI(symm1, N, world_size, 0); },
		"MPI symm", 10, out, curr_rank, 0);

	if (!is_symm2 && curr_rank == 0) {
		std::cout << "MPI symmetry check failed on symm. matrix" << std::endl;
	}

	bool is_symm3 = Benchmark([=]() { return checkSymMPI_DISTRIBUTED(symm2, N, world_size, 0); },
		"MPI distributed symm", 10, out, curr_rank, 0);

	if (!is_symm3 && curr_rank == 0) {
		std::cout << "MPI distributed symmetry check failed on symm. matrix" << std::endl;
	}

	if(curr_rank == 0)
		delete[] symm1;

	delete[] symm2;
	delete[] distributed_matrix;
	delete[] transposed5;
	delete[] base_matrix;

	MPI_Finalize();
	return 0;
}
