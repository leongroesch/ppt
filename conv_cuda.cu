#include <iostream>

#include "convolution.h"
#include "array_management.h"
#include "test_cases.h"

using namespace std;

int main(int argc, char* argv[]) {

	if(argc == 1)
	{
		testing::InitGoogleTest(&argc, argv);
	  return RUN_ALL_TESTS();
	}
	else if (argc < 4) {
		cout << "Usage: srun out <int: dimension of input matrix> <int: dimension of kernel> <blocksize>\n";
		return 0;
	}

	uint32_t MATDIM = strtol(argv[1], NULL, 10);
	uint32_t KERDIM = strtol(argv[2], NULL, 10);
	uint32_t N = strtol(argv[3], NULL, 10);
	uint32_t grid_size = ceil((MATDIM-KERDIM+1) * (MATDIM-KERDIM+1) / ((double) N));

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

	double* 	new_h_res_binW = new double[(MATDIM-KERDIM+1)*(MATDIM-KERDIM+1)];
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
	double *d_mat, *d_ker, *d_res_standard, *d_res_binW, *new_d_res_binW;
	unsigned char *d_mat_bin, *d_ker_bin, *d_res_binWbinI, *new_d_res_binWbinI;

	// Allocate all matrices on device (cudaFree later!)
	cudaMalloc((void**) &d_mat, mat_size);
	cudaMalloc((void**) &d_ker, ker_size);
	cudaMalloc((void**) &d_mat_bin, mat_bin_size);
	cudaMalloc((void**) &d_ker_bin, ker_bin_size);
	cudaMalloc((void**) &d_res_standard, res_standard_size);
	cudaMalloc((void**) &d_res_binW, res_binW_size);
	cudaMalloc((void**) &d_res_binWbinI, res_binWbinI_size);

	cudaMalloc((void**) &new_d_res_binW, res_binW_size);
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

  //TODO DEBUG: Print the double matrix.
   printMatrix(MATDIM, h_mat);

	initMat(KERDIM, h_ker);
	// Convert the double matrix into binary
	convertToBinary(KERDIM, h_ker, (uint32_t) ceil(KERDIM*KERDIM/8.0), h_ker_bin);
	// TODO DEBUG: Print the double matrix.
	//printMatrix(KERDIM, h_ker);
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

	// // Compute the different modes of convolution
	// clock_gettime(CLOCK_MONOTONIC, &tstart);
	// convStandard<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res_standard);
	// clock_gettime(CLOCK_MONOTONIC, &tend);
	// elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	// cout << "Standard convolution took " << elapsed << " seconds.\n";

	cout << "\n----------Binary weights----------\n";
	//Run and measure time for newConvBinW
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	newConvBinW<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker_bin, new_d_res_binW);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Binary weights took " << elapsed << " nanoseconds.\n";

	//Run and measure time for old convBinW
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	convBinW<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker_bin, d_res_binW);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Binary weights took " << elapsed << " nanoseconds.\n";

	cout << "\n----------Binary weights Binary Inputs----------\n";
	//Run and measure time for newConvBinWBinI
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	newConvBinWBinI<unsigned char><<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat_bin, d_ker_bin, new_d_res_binWbinI);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Byte wise Binary inputs and binary weights took " << elapsed << " nanoseconds.\n";
	cout << elapsed << "\n";

	//Run and measure time for old convBinWBinI
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	convBinWBinI<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat_bin, d_ker_bin, d_res_binWbinI);
	clock_gettime(CLOCK_MONOTONIC, &tend);
	elapsed = ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec);
	cout << "Binary inputs and binary weights took " << elapsed << " nanoseconds.\n";
	cout << elapsed << "\n";

	// Fetch the results from device
	// cudaMemcpy(h_res_standard, d_res_standard, res_standard_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_binW, d_res_binW, res_binW_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_h_res_binW, new_d_res_binW, res_binW_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_res_binWbinI, d_res_binWbinI, res_binWbinI_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_h_res_binWbinI, new_d_res_binWbinI, res_binWbinI_size, cudaMemcpyDeviceToHost);

	// TODO DEBUG: Print the results
	// cout << "Standard convolution DOUBLExDOUBLE\n";
	// printMatrix(MATDIM-KERDIM+1, h_res_standard);
	// cout << "Binary weight convolution DOUBLExBITS\n";
	cout << "\n----------Reuslt for old binary Weights----------\n";
	printMatrix(MATDIM-KERDIM+1, h_res_binW);
	cout << "\n----------Reuslt for new binary Wieghts----------\n";
	printMatrix(MATDIM-KERDIM+1, new_h_res_binW);
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
	cudaFree(new_d_res_binW);
	cudaFree(d_res_binWbinI);
	cudaFree(new_d_res_binWbinI);
	cudaFree(d_MATDIM);
	cudaFree(d_KERDIM);

	return 0;
}
