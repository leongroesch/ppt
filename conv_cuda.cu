#include <cuda.h>
#include <math.h>
#include <iostream>
#include <stdio.h>

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
void newConvBinW(uint32_t* d_MATDIM, uint32_t* d_KERDIM, double* mat, unsigned char* ker, double* res) {
	uint32_t MATDIM = d_MATDIM[0];
	uint32_t KERDIM = d_KERDIM[0];
	uint32_t KERSIZE = KERDIM*KERDIM;
	uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t RESDIM = MATDIM-KERDIM+1;
	uint32_t POSMAT = threadID + (threadID % RESDIM) * (MATDIM - RESDIM);
	if (threadID < (RESDIM * RESDIM)) {
		double sum = 0.0;
		for (uint32_t i = 0; i < KERSIZE; i++) {
			uint32_t currentPosMat = POSMAT + ((int)(i / KERDIM) * MATDIM + i % KERDIM);
			uint32_t kerPosByte = (int)(i / 7);
			uint32_t kerPosBit = 7 - i % 7;
			if((unsigned char)((ker[kerPosByte] >> kerPosBit) & 0x1) == 1) {
				sum += mat[currentPosMat];
			} else {
				sum -= mat[currentPosMat];
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

bool cmpf(double A, double B, double epsilon = 0.00005f)
{
    return (fabs(A - B) < epsilon);
}

void testBinW(uint32_t dimM, uint32_t dimK, uint32_t N, int testNo, double* mat, double* res, unsigned char* ker) {
    uint32_t DIMMAT = dimM, DIMKER = dimK, DIMRES = dimM - dimK + 1, MATSIZE = DIMMAT * DIMMAT, KERSIZE = DIMKER * DIMKER, RESSIZE = DIMRES * DIMRES;
    uint32_t h_MATDIM[1];
    h_MATDIM[0] = DIMMAT;
    uint32_t h_KERDIM[1];
    h_KERDIM[0] = DIMKER;
    double resToTest[RESSIZE];

		uint32_t mat_size = MATSIZE * sizeof(double);
		uint32_t ker_size = (uint32_t) ceil(KERSIZE/8.0) * sizeof(unsigned char);
		uint32_t res_size = RESSIZE * sizeof(double);

		uint32_t *d_MATDIM, *d_KERDIM;
    double *d_mat, *d_res_binW;
    unsigned char *d_ker_bin;

		cudaMalloc((void**) &d_mat, mat_size);
		cudaMalloc((void**) &d_ker_bin, ker_size);
		cudaMalloc((void**) &d_res_binW, res_size);
		cudaMalloc((void**) &d_MATDIM, sizeof(uint32_t));
    cudaMalloc((void**) &d_KERDIM, sizeof(uint32_t));

    cudaMemcpy(d_mat, mat, mat_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_ker_bin, ker, ker_size, cudaMemcpyHostToDevice);

		cudaMemcpy(d_MATDIM, h_MATDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_KERDIM, h_KERDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);

		uint32_t grid_size = ceil(RESSIZE / ((double) N));

		newConvBinW<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker_bin, d_res_binW);

    cudaMemcpy(resToTest, d_res_binW, res_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < RESSIZE; i++) {
        if(! cmpf(res[i], resToTest[i])) {
            cout << "\033[1;31mTest №: " << testNo << " failed!\n\033[0m";
						cout << res[i] << " " << resToTest[i] << "\n";
						/*cudaFree(d_ker_bin);
						cudaFree(d_mat);
						cudaFree(d_res_binW);
						cudaFree(d_MATDIM);
						cudaFree(d_KERDIM);
						return;*/
        }
    }

    cout << "\033[1;32mTest №: " << testNo << " successed!\n\033[0m";
    cudaFree(d_ker_bin);
		cudaFree(d_mat);
		cudaFree(d_res_binW);
    cudaFree(d_MATDIM);
    cudaFree(d_KERDIM);

}

void runTests(uint32_t N) {
	double mat0[16], res0[9];
	unsigned char ker0[(uint32_t) ceil(4/8.0)];

	mat0[0] = 0.178184;
	mat0[1] = -0.368889;
	mat0[2] = -0.0857587;
	mat0[3] = -0.256823;
	mat0[4] = -0.974276;
	mat0[5] = -0.575986;
	mat0[6] = -0.416872;
	mat0[7] = 0.450409;
	mat0[8] = -0.984626;
	mat0[9] = -0.933976;
	mat0[10] = -0.726241;
	mat0[11] = -0.354601;
	mat0[12] = 0.644028;
	mat0[13] = -0.586327;
	mat0[14] = 0.37183;
	mat0[15] = 0.765458;
	ker0[0] = 0b11100000;
	res0[0] = -0.588995;
	res0[1] = -1.209863;
	res0[2] = -1.600912;
	res0[3] = 0.203936;
	res0[4] = -1.200593;
	res0[5] = -1.532846;
	res0[6] = -0.338103;
	res0[7] = -0.688247;
	res0[8] = -1.474470;

	testBinW(4, 2, N, 0, mat0, res0, ker0);

	double mat1[16], res1[16];
	unsigned char ker1[(uint32_t) ceil(4/8.0)];

	mat1[0] = -0.510971;
	mat1[1] = -0.247672;
	mat1[2] = -0.859489;
	mat1[3] = -0.390785;
	mat1[4] = -0.75706;
	mat1[5] = -0.718692;
	mat1[6] = 0.292791;
	mat1[7] = 0.0486137;
	mat1[8] = -0.855771;
	mat1[9] = 0.384334;
	mat1[10] = 0.623243;
	mat1[11] = 0.445734;
	mat1[12] = -0.741026;
	mat1[13] = -0.262363;
	mat1[14] = 0.357182;
	mat1[15] = 0.869584;
	ker1[0] = 0b00000000;

	for (int i = 0; i < 16; i++) {
		res1[i] = -mat1[i];
	}

	testBinW(4, 1, N, 1, mat1, res1, ker1);

	ker1[0] = 0b10000000;

	for (int i = 0; i < 16; i++) {
		res1[i] = mat1[i];
	}

	testBinW(4, 1, N, 2, mat1, res1, ker1);

	double mat2[16], res2[4];
	unsigned char ker2[(uint32_t) ceil(9/8.0)];

	mat2[0] = 0.0803077;
	mat2[1] = 0.0543088;
	mat2[2] = -0.874869;
	mat2[3] = -0.654904;
	mat2[4] = -0.954831;
	mat2[5] = -0.186203;
	mat2[6] = -0.744362;
	mat2[7] = 0.749519;
	mat2[8] = -0.921765;
	mat2[9] = 0.78594;
	mat2[10] = -0.946144;
	mat2[11] = -0.519783;
	mat2[12] = -0.275407;
	mat2[13] = 0.301698;
	mat2[14] = -0.0470979;
	mat2[15] = -0.799294;
	ker2[0] = 0b00100100;
	ker2[1] = 0b00000000;
	res2[0] = 0.469155;
	res2[1] = 2.87521;
	res2[2] = 1.38936;
	res2[3] = 1.8652;

	testBinW(4, 3, N, 3, mat2, res2, ker2);
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		cout << "Usage: srun out <int: dimension of input matrix> <int: dimension of kernel> <blocksize>\n";
		cout << "(Optional) 1, if you want to run the Test";
		return 0;
	}

	uint32_t MATDIM = strtol(argv[1], NULL, 10);
	uint32_t KERDIM = strtol(argv[2], NULL, 10);
	uint32_t N = strtol(argv[3], NULL, 10);
	if (argv[4] != NULL && strtol(argv[4], NULL, 10) == 1) {
		runTests(N);
	}

	uint32_t h_MATDIM[1];
	h_MATDIM[0] = MATDIM;
	uint32_t h_KERDIM[1];
	h_KERDIM[0] = KERDIM;

	struct timespec tstart={0,0}, tend={0,0};
	double elapsed;

	// Matrix (double)
	double* 	h_mat = new double[MATDIM*MATDIM];
	// Kernel (double)
	double* 	h_ker = new double[KERDIM*KERDIM];

    // Matrix (bits)
	unsigned char* 	h_mat_bin = new unsigned char[(uint32_t) ceil(MATDIM*MATDIM/8.0)];
    // Kernel (bits)
	unsigned char* 	h_ker_bin = new unsigned char[(uint32_t) ceil(KERDIM*KERDIM/8.0)];

	// Result of standard convolution
	double* 	h_res_standard = new double[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];
	// Result of convolution with binary weights
	double* 	h_res_binW = new double[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];
	// Result of convolution with binary weights and binary inputs
	unsigned char* 	h_res_binWbinI = new unsigned char[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];

	unsigned char* 	new_h_res_binWbinI = new unsigned char[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];

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
	if(argc == 4)
		printBinary(MATDIM, (uint32_t) ceil(MATDIM*MATDIM/8.0), h_mat_bin);

    // TODO DEBUG: Print the double matrix.
  printMatrix(MATDIM, h_mat);

	initMat(KERDIM, h_ker);
	// Convert the double matrix into binary
	convertToBinary(KERDIM, h_ker, (uint32_t) ceil(KERDIM*KERDIM/8.0), h_ker_bin);
	// TODO DEBUG: Print the double matrix.
	printMatrix(KERDIM, h_ker);
	// TODO DEBUG: Print the binary matrix.
	if(argc == 4)
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
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	convStandard<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res_standard);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Standard convolution took " << elapsed << " seconds.\n";

	clock_gettime(CLOCK_MONOTONIC, &tstart);
	newConvBinW<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker_bin, d_res_binW);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Binary weights took " << elapsed << " nanoseconds.\n";

	clock_gettime(CLOCK_MONOTONIC, &tstart);
	convBinWBinI<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat_bin, d_ker_bin, d_res_binWbinI);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Binary inputs and binary weights took " << elapsed << " nanoseconds.\n";
	cout << elapsed << "\n";

	//Leon
	/*clock_gettime(CLOCK_MONOTONIC, &tstart);
	newConvBinWBinI<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat_bin, d_ker_bin, new_d_res_binWbinI);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Byte wise Binary inputs and binary weights took " << elapsed << " nanoseconds.\n";
	cout << elapsed << "\n";*/

	// Fetch the results from device
	cudaMemcpy(h_res_standard, d_res_standard, res_standard_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_binW, d_res_binW, res_binW_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_binWbinI, d_res_binWbinI, res_binWbinI_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_h_res_binWbinI, new_d_res_binWbinI, res_binWbinI_size, cudaMemcpyDeviceToHost);

	// TODO DEBUG: Print the results
	cout << "Standard convolution DOUBLExDOUBLE\n";
	printMatrix(MATDIM-KERDIM+1, h_res_standard);
	cout << "Binary weight convolution DOUBLExBITS\n";
	printMatrix(MATDIM-KERDIM+1, h_res_binW);
	if(argc == 4)
	{
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
	}

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
