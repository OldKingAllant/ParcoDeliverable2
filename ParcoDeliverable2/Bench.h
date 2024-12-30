#ifndef PARCO_BENCH
#define PARCO_BENCH

#include <iostream>
#include <fstream>
#include <chrono>

/// <summary>
/// Runs the given function a certain amount
/// of times while registering the time
/// it takes to execute it, then performs
/// the average of the wall clock time
/// and prints it to the console
/// </summary>
/// <typeparam name="Func">Function type</typeparam>
/// <typeparam name="type">Hidden</typeparam>
/// <param name="function">The function to benchmark</param>
/// <param name="name">Benchmark name</param>
/// <param name="repeat">Amount of times that the function needs to be run</param>
template <typename Func,
	typename std::enable_if< std::is_same<decltype((std::declval<Func>())()), void>::value, bool>::type = true
>
void Benchmark(Func&& function, const char* name, uint32_t repeat, std::ofstream& out, 
	int rank, int root) {
	uint32_t rep_temp = repeat;
	auto start = std::chrono::high_resolution_clock::now();

	while (repeat--) {
		function();
	}

	auto end = std::chrono::high_resolution_clock::now();

	auto diff = (end - start).count() / 1e6;
	
	if (rank == root) {
		std::cout << name << " took " << diff / rep_temp << " ms" << std::endl;
		out << diff / rep_temp << std::endl;
	}
}

/// <summary>
/// Runs the given function a certain amount
/// of times while registering the time
/// it takes to execute it, then performs
/// the average of the wall clock time
/// and prints it to the console.
/// Also returns the value from the function
/// (which means that it would be a good idea
/// to use a routine that always returns the same
/// thing)
/// </summary>
/// <typeparam name="Func">Function type</typeparam>
/// <typeparam name="type">Hidden</typeparam>
/// <param name="function">The function to benchmark</param>
/// <param name="name">Benchmark name</param>
/// <param name="repeat">Amount of times that the function needs to be run</param>
/// <returns>The return value of the function</returns>
template <typename Func,
	typename = typename std::enable_if< !std::is_same<decltype((std::declval<Func>())()), void>::value, bool>::type
>
decltype((std::declval<Func>())()) Benchmark(Func&& function, const char* name, uint32_t repeat, 
	std::ofstream& out, int rank, int root) {
	uint32_t rep_temp = repeat;
	using RetType = decltype(function()); //Deduce return type
	RetType ret{};

	auto start = std::chrono::high_resolution_clock::now();

	while (repeat--) {
		ret = function();
	}

	auto end = std::chrono::high_resolution_clock::now();

	auto diff = (end - start).count() / 1e6;

	if (rank == root) {
		std::cout << name << " took " << diff / rep_temp << " ms" << std::endl;
		out << diff / rep_temp << std::endl;
	}

	return ret;
}

////////////////////////////////////////////

#endif // !PARCO_BENCH
