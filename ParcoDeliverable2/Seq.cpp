#include "Seq.hpp"

bool checkSym(const MatType* M, u32 N) {
	bool is_symm = true;

	for (u32 row = 0; row < N; row++) {
		for (u32 col = row + 1; col < N; col++) {
			
			if (M[row * N + col] != M[col * N + row]) {
				is_symm = false;
			}

		}
	}

	return is_symm;
}

void matTranspose(MatType const* M, MatType* T, uint32_t N) {
	for (uint32_t row_idx = 0; row_idx < N; row_idx++) {
		for (uint32_t col_idx = 0; col_idx < N; col_idx++) {
			T[col_idx * N + row_idx] = M[row_idx * N + col_idx];
		}
	}
}