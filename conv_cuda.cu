#include <cuda.h>
#include <math.h>
#include <iostream>

using namespace std;

__global__
void convStandard(uint32_t* d_MATDIM, uint32_t* d_KERDIM, double* mat, double* ker, double* res) {
	uint32_t MATDIM = d_MATDIM[0];
	uint32_t KERDIM = d_KERDIM[0];
	uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)) {
		uint32_t index = KERDIM/2 + (KERDIM/2) * MATDIM + (threadID / (MATDIM-KERDIM+1)) * MATDIM + threadID % (MATDIM-KERDIM+1);
		double sum = 0.0;

		for(int i = -((int32_t) KERDIM)/2; i < ((int32_t) KERDIM)/2+1; i++){
			for(int j = -((int32_t) KERDIM)/2; j < ((int32_t) KERDIM)/2+1; j++){
				sum += mat[index + i*MATDIM + j] * ker[(i+KERDIM/2) * KERDIM + (j+KERDIM/2)];
			}
		}

		res[threadID] = sum;
	}
}

__global__
void convBinW(uint32_t* d_MATDIM, uint32_t* d_KERDIM, double* mat, unsigned char* ker, double* res) {
	uint32_t MATDIM = d_MATDIM[0];
	uint32_t KERDIM = d_KERDIM[0];
	uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)) {
		uint32_t index_mat = KERDIM/2 + (KERDIM/2) * MATDIM + (threadID / (MATDIM-KERDIM+1)) * MATDIM + threadID % (MATDIM-KERDIM+1);
		double sum = 0.0;
		for(int i = -((int32_t) KERDIM)/2; i < ((int32_t) KERDIM)/2+1; i++){
			for(int j = -((int32_t) KERDIM)/2; j < ((int32_t) KERDIM)/2+1; j++){
				uint32_t index_ker = (i+KERDIM/2) * KERDIM + (j+KERDIM/2);

				if ((unsigned char)((unsigned char)(ker[index_ker/8] << (index_ker % 8)) >> 7) == 1)
					sum += mat[index_mat + i*MATDIM + j];
				else
					sum -= mat[index_mat + i*MATDIM + j];
			}
		}
		res[threadID] = sum;
	}
}

__global__
void convBinWBinI(uint32_t* d_MATDIM, uint32_t* d_KERDIM, unsigned char* mat, unsigned char* ker, unsigned char* res) {
	uint32_t MATDIM = d_MATDIM[0];
	uint32_t KERDIM = d_KERDIM[0];
	uint32_t xnor_number = 0;
	unsigned char bit_counter = 0;
	unsigned char pop_count = 0;

	uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)) {
		uint32_t index_mat = KERDIM/2 + (KERDIM/2) * MATDIM + (threadID / (MATDIM-KERDIM+1)) * MATDIM + threadID % (MATDIM-KERDIM+1);

		for(int i = -((int32_t) KERDIM)/2; i < ((int32_t) KERDIM)/2+1; i++){
			for(int j = -((int32_t) KERDIM)/2; j < ((int32_t) KERDIM)/2+1; j++){
				uint32_t index_ker = (i+(KERDIM >> 1)) * KERDIM + (j+(KERDIM >> 1));
				uint32_t index_mat_bin = index_mat + i*MATDIM + j;

				if (bit_counter == 32) {
					pop_count += (unsigned char) __popc((unsigned int) xnor_number);
					bit_counter = 0;
					xnor_number = 0;
				} else {
					if (((ker[index_ker >> 3] << (index_ker % 8)) >> 7) == ((mat[index_mat_bin >> 3] << (index_mat_bin % 8)) >> 7))
            			xnor_number |= 1 << bit_counter;
					bit_counter++;
				}
			}
		}

		pop_count += (unsigned char) __popc((unsigned int) xnor_number);
		res[threadID] = pop_count;
	}
}

__device__
void get_byte(uint32_t* d_MATDIM, unsigned char* matrix, unsigned char* result, uint32_t lh_idx, uint32_t rh_idx)
{
  unsigned char lh_byte = matrix[lh_idx/8];
  unsigned char rh_byte = matrix[rh_idx/8];
  if(rh_idx-lh_idx == 7)
  {
    lh_byte = lh_byte<<lh_idx%8;
    rh_byte = rh_byte>>(8-lh_idx%8);
  }
  else
  {
    lh_byte = lh_byte<<lh_idx%8;
    lh_byte &= 0xFF<<(7-(rh_idx-lh_idx));
    rh_byte = (rh_byte>>(7-rh_idx%8))<<((lh_idx%8)-(rh_idx%8+1));
  }
  *result = lh_byte | rh_byte;
}

//Leong: First approach of a byte wise binary/binary convolution
__global__
void newConvBinWBinI(uint32_t* d_MATDIM, uint32_t* d_KERDIM, unsigned char* matrix, unsigned char* kernel, unsigned char* result){
	uint32_t KERDIM = d_KERDIM[0];
	uint32_t MATDIM = d_MATDIM[0];
	uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
	result[threadID] = 0;
	uint32_t midpoint_index = KERDIM/2 + (KERDIM/2) * MATDIM + (threadID / (MATDIM-KERDIM+1)) * MATDIM + threadID % (MATDIM-KERDIM+1);
	unsigned char kernel_byte = 0;
	unsigned char matrix_byte = 0;
	unsigned char result_byte = 0;
	int lh_matrix_idx = 0;
	int lh_kernel_idx = 0;
	unsigned int offset = 0;

	//Iterate ofter the maks columns
	for(int row = -(KERDIM/2); row <= (int)KERDIM/2  ; row++)
	{
		lh_matrix_idx = midpoint_index+(row*MATDIM) - KERDIM/2;
		lh_kernel_idx = (row+KERDIM/2)*KERDIM;
		//Iterate over the bytes in one collumn
		for(unsigned int byte = 0; byte <= KERDIM/8; byte++)
		{
			if(byte == KERDIM/8)
			 offset =	(KERDIM%8)-1;
			else
				offset = 7;
		  get_byte(d_KERDIM, kernel, &kernel_byte, lh_kernel_idx+8*byte, lh_kernel_idx+8*byte+offset);
			get_byte(d_MATDIM, matrix, &matrix_byte, lh_matrix_idx+8*byte, lh_matrix_idx+8*byte+offset);
			//XNOR
			result_byte = ~(kernel_byte^matrix_byte);
			//Only use the x left most bits from the last byte
			if(byte == KERDIM/8)
				result_byte = ((0XFF<<(7-offset)) & result_byte);
			result[threadID] += __popc(result_byte);
		}
	}

}

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
	cout << "dim: " << dim << "x" << dim << "\n{\n";
	for (uint32_t i = 0; i < dim; i++) {
		for (uint32_t j = 0; j < dim; j++) {
			cout << mat[i*dim+j] << ", ";
		}
		cout << '\n';
	}
	cout << "}\n";
}

void printBinary(uint32_t dim, uint32_t size_bin, unsigned char* mat) {
	unsigned char rest;

	for (uint32_t i = 0; i < size_bin; i++) {
		rest = mat[i];
		for (uint32_t j = 0; j < 8; j++) {
			if (i * 8 + j == dim*dim) {
				cout << "\n";
				break;
			}

			if(rest - pow(2,7-j) >= 0) {
				rest = rest - pow(2,7-j);
				cout << "1 ";
			} else {
				cout << "0 ";
			}

			if((i * 8 + j + 1) % dim == 0) {
				cout << "\n";
			}

		}
	}
}

int main(int argc, char* argv[]) {
	if (argc != 4) {
		cout << "Usage: srun out <int: dimension of input matrix> <int: dimension of kernel> <blocksize>\n";
		return 0;
	}

	uint32_t MATDIM = strtol(argv[1], NULL, 10);
	uint32_t KERDIM = strtol(argv[2], NULL, 10);
	uint32_t N = strtol(argv[3], NULL, 10);

	uint32_t h_MATDIM[1];
	h_MATDIM[0] = MATDIM;
	uint32_t h_KERDIM[1];
	h_KERDIM[0] = KERDIM;

	struct timespec tstart={0,0}, tend={0,0};
	double elapsed;

	// Matrix (double)
	double 	h_mat[MATDIM*MATDIM];
	// Kernel (double)
	double 	h_ker[KERDIM*KERDIM];

    // Matrix (bits)
	unsigned char 	h_mat_bin[(uint32_t) ceil(MATDIM*MATDIM/8.0)];
    // Kernel (bits)
	unsigned char 	h_ker_bin[(uint32_t) ceil(KERDIM*KERDIM/8.0)];

	// Result of standard convolution
	double 	h_res_standard[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];
	// Result of convolution with binary weights
	double 	h_res_binW[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];
	// Result of convolution with binary weights and binary inputs
	unsigned char 	h_res_binWbinI[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];

	unsigned char 	new_h_res_binWbinI[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];

	uint32_t mat_size = 			MATDIM*MATDIM * sizeof(double);
	uint32_t ker_size = 			KERDIM*KERDIM * sizeof(double);
	uint32_t mat_bin_size = 		(uint32_t) ceil(MATDIM*MATDIM/8.0) * sizeof(unsigned char);
	uint32_t ker_bin_size = 		(uint32_t) ceil(KERDIM*KERDIM/8.0) * sizeof(unsigned char);
	uint32_t res_standard_size =	(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) * sizeof(double);
	uint32_t res_binW_size =		(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) * sizeof(double);
	uint32_t res_binWbinI_size =	(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) * sizeof(unsigned char);

	// Pointers for allocation on device
	uint32_t *d_MATDIM, *d_KERDIM;
	double *d_mat, *d_ker, *d_res_standard, *d_res_binW;
	unsigned char *d_mat_bin, *d_ker_bin, *d_res_binWbinI, *new_d_res_binWbinI;

	// Allocate all matrices on device (cudaFree later!)
	cudaMalloc((void**) &d_mat, mat_size);
	cudaMalloc((void**) &d_ker, ker_size);
	cudaMalloc((void**) &d_mat_bin, mat_bin_size);
	cudaMalloc((void**) &d_ker_bin, ker_bin_size);
	cudaMalloc((void**) &d_res_standard, res_standard_size);
	cudaMalloc((void**) &d_res_binW, res_binW_size);
	cudaMalloc((void**) &d_res_binWbinI, res_binWbinI_size);

	cudaMalloc((void**) &new_d_res_binWbinI, res_binWbinI_size);

	cudaMalloc((void**) &d_MATDIM, sizeof(uint32_t));
	cudaMalloc((void**) &d_KERDIM, sizeof(uint32_t));

	// Seed for random number generation
	srand(time(NULL));

	// Randomize the values of the double matrix with values -1.0 ... 1.0
	initMat(MATDIM, h_mat);
    // Convert the double matrix into binary (0 = -1, 1 = 1)
	convertToBinary(MATDIM, h_mat, (uint32_t) ceil(MATDIM*MATDIM/8.0), h_mat_bin);
	// TODO DEBUG: Print the binary matrix.
	printBinary(MATDIM, (uint32_t) ceil(MATDIM*MATDIM/8.0), h_mat_bin);

    // TODO DEBUG: Print the double matrix.
   // printMatrix(MATDIM, h_mat);

	initMat(KERDIM, h_ker);
	// Convert the double matrix into binary
	convertToBinary(KERDIM, h_ker, (uint32_t) ceil(KERDIM*KERDIM/8.0), h_ker_bin);
	// TODO DEBUG: Print the double matrix.
	// printMatrix(KERDIM, h_ker);
	// TODO DEBUG: Print the binary matrix.
	printBinary(KERDIM, (uint32_t) ceil(KERDIM*KERDIM/8.0), h_ker_bin);

	// Copy all the matrices to the device (except the result matrices)
	cudaMemcpy(d_mat, h_mat, mat_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ker, h_ker, ker_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_bin, h_mat_bin, mat_bin_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ker_bin, h_ker_bin, ker_bin_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_MATDIM, h_MATDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_KERDIM, h_KERDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);

	uint32_t grid_size = ceil((MATDIM-KERDIM+1) * (MATDIM-KERDIM+1) / ((double) N));

	// // Compute the different modes of convolution
	// clock_gettime(CLOCK_MONOTONIC, &tstart);
	// convStandard<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res_standard);
	// clock_gettime(CLOCK_MONOTONIC, &tend);
	// elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	// cout << "Standard convolution took " << elapsed << " seconds.\n";
	//
	// clock_gettime(CLOCK_MONOTONIC, &tstart);
	// convBinW<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker_bin, d_res_binW);
	// clock_gettime(CLOCK_MONOTONIC, &tend);
	// elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	// cout << "Binary weights took " << elapsed << " nanoseconds.\n";

	clock_gettime(CLOCK_MONOTONIC, &tstart);
	convBinWBinI<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat_bin, d_ker_bin, d_res_binWbinI);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Binary inputs and binary weights took " << elapsed << " nanoseconds.\n";
	cout << elapsed << "\n";

	//Leon
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	newConvBinWBinI<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat_bin, d_ker_bin, new_d_res_binWbinI);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Byte wise Binary inputs and binary weights took " << elapsed << " nanoseconds.\n";
	cout << elapsed << "\n";

	// Fetch the results from device
	// cudaMemcpy(h_res_standard, d_res_standard, res_standard_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_res_binW, d_res_binW, res_binW_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_binWbinI, d_res_binWbinI, res_binWbinI_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_h_res_binWbinI, new_d_res_binWbinI, res_binWbinI_size, cudaMemcpyDeviceToHost);

	// TODO DEBUG: Print the results
	// cout << "Standard convolution DOUBLExDOUBLE\n";
	// printMatrix(MATDIM-KERDIM+1, h_res_standard);
	// cout << "Binary weight convolution DOUBLExBITS\n";
	// printMatrix(MATDIM-KERDIM+1, h_res_binW);
	cout << "Binary weights and binary inputs BITSxBITS\n";
	cout << "dim: " << MATDIM-KERDIM+1 << "x" << MATDIM-KERDIM+1 << "\n{\n";
	for (uint32_t i = 0; i < MATDIM-KERDIM+1; i++) {
		for (uint32_t j = 0; j < MATDIM-KERDIM+1; j++) {
			cout << (uint32_t) h_res_binWbinI[i*(MATDIM-KERDIM+1)+j] << ", ";
		}
		cout << '\n';
	}
	cout << "}\n";

	cout << "NEW Binary weights and binary inputs BITSxBITS\n";
	cout << "dim: " << MATDIM-KERDIM+1 << "x" << MATDIM-KERDIM+1 << "\n{\n";
	for (uint32_t i = 0; i < MATDIM-KERDIM+1; i++) {
		for (uint32_t j = 0; j < MATDIM-KERDIM+1; j++) {
			cout << (uint32_t) new_h_res_binWbinI[i*(MATDIM-KERDIM+1)+j] << ", ";
		}
		cout << '\n';
	}
	cout << "}\n";

	cudaFree(d_mat);
	cudaFree(d_ker);
	cudaFree(d_mat_bin);
	cudaFree(d_ker_bin);
	cudaFree(d_res_standard);
	cudaFree(d_res_binW);
	cudaFree(d_res_binWbinI);
	cudaFree(new_d_res_binWbinI);
	cudaFree(d_MATDIM);
	cudaFree(d_KERDIM);

	return 0;
}
