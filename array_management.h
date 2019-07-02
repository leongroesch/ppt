#ifndef _ARRAY_MANAGEMENT_H_
#define _ARRAY_MANAGEMENT_H_

#include <iostream>
#include <cstdlib>

void initMat(uint32_t dim, double* mat) {
	for (uint32_t i = 0; i < dim; i++) {
		for (uint32_t j = 0; j < dim; j++) {
			mat[i*dim+j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
		}
	}
}

void convertToBinary(uint32_t dim, double* mat, uint32_t size_bin, unsigned char* mat_bin) {
	uint32_t pos_bit = 0;
	uint32_t pos_byte = 0;
	unsigned char sum = 0;

	for (uint32_t i = 0; i < dim; i++) {
		for (uint32_t j = 0; j < dim; j++) {
			if (mat[i*dim+j] >= 0) {
				sum += pow(2, 7-pos_bit);
			}

			if (pos_bit == 7) {
				mat_bin[pos_byte] = sum;

				pos_byte++;
				sum = 0;
				pos_bit = 0;
			} else {
				pos_bit++;
			}
		}
	}

	if (dim*dim % 8 != 0) {
		mat_bin[pos_byte] = sum;
	}
}

void printMatrix(uint32_t dim, double* mat) {
	std::cout << "dim: " << dim << "x" << dim << "\n{\n";
	for (uint32_t i = 0; i < dim; i++) {
		for (uint32_t j = 0; j < dim; j++) {
			std::cout << mat[i*dim+j] << ", ";
		}
		std::cout << '\n';
	}
	std::cout << "}\n";
}

void printMatrix(uint32_t dim, uint32_t* mat) {
	std::cout << "dim: " << dim << "x" << dim << "\n{\n";
	for (uint32_t i = 0; i < dim; i++) {
		for (uint32_t j = 0; j < dim; j++) {
			std::cout << mat[i*dim+j] << ", ";
		}
		std::cout << '\n';
	}
	std::cout << "}\n";
}

void printCharMatrix(uint32_t dim, unsigned char* mat){
	std::cout << "dim: " << dim << "x" << dim << "\n{\n";
	for (uint32_t i = 0; i < dim; i++) {
		for (uint32_t j = 0; j < dim; j++) {
			std::cout << (uint32_t) mat[i*dim+j] << ", ";
		}
		std::cout << '\n';
	}
	std::cout << "}\n";
}

void printBinary(uint32_t dim, uint32_t size_bin, unsigned char* mat) {
	unsigned char rest;

	for (uint32_t i = 0; i < size_bin; i++) {
		rest = mat[i];
		for (uint32_t j = 0; j < 8; j++) {
			if (i * 8 + j == dim*dim) {
				std::cout << "\n";
				break;
			}

			if(rest - pow(2,7-j) >= 0) {
				rest = rest - pow(2,7-j);
				std::cout << "1 ";
			} else {
				std::cout << "0 ";
			}

			if((i * 8 + j + 1) % dim == 0) {
				std::cout << "\n";
			}

		}
	}
}

#endif
