#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "convolution.h"
#include "array_management.h"

bool cmpf(double A, double B, double epsilon = 0.00005f)
{
    return (fabs(A - B) < epsilon);
}

void convBinWWrapper(uint32_t dimM, uint32_t dimK, double* mat, double* res, unsigned char* ker) {
    uint32_t DIMMAT = dimM, DIMKER = dimK, DIMRES = dimM - dimK + 1, MATSIZE = DIMMAT * DIMMAT, KERSIZE = DIMKER * DIMKER, RESSIZE = DIMRES * DIMRES;
    uint32_t h_MATDIM[1];
    h_MATDIM[0] = DIMMAT;
    uint32_t h_KERDIM[1];
    h_KERDIM[0] = DIMKER;

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

    double N = 10;
		uint32_t grid_size = ceil(RESSIZE / ( N));

		newConvBinW<<<grid_size, N>>>(d_MATDIM, d_KERDIM, d_mat, d_ker_bin, d_res_binW);

    cudaMemcpy(res, d_res_binW, res_size, cudaMemcpyDeviceToHost);

    cudaFree(d_ker_bin);
		cudaFree(d_mat);
		cudaFree(d_res_binW);
    cudaFree(d_MATDIM);
    cudaFree(d_KERDIM);
}

template <typename Word>
void convBinWBinIWrapper(uint32_t blocks, uint32_t threads, uint32_t MATDIM, uint32_t KERDIM, Word* matrix, Word* kernel, uint32_t* result)
{
  double word_length = sizeof(Word) * 8.0;
  //Calculate Array size
  uint32_t res_size = (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) * sizeof(uint32_t);
  uint32_t mat_size = (uint32_t) ceil(MATDIM*MATDIM/word_length) * sizeof(Word);
	uint32_t ker_size = (uint32_t) ceil(KERDIM*KERDIM/word_length) * sizeof(Word);

  //Mallocate device data
  uint32_t *d_MATDIM, *d_KERDIM;
  cudaMalloc((void**) &d_MATDIM, sizeof(uint32_t));
  cudaMalloc((void**) &d_KERDIM, sizeof(uint32_t));

  Word *d_mat, *d_ker;
  uint32_t *d_res;
  cudaMalloc((void**) &d_mat, mat_size);
	cudaMalloc((void**) &d_ker, ker_size);
  cudaMalloc((void**) &d_res, res_size);

  //Copy input to device
  cudaMemcpy(d_MATDIM, &MATDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_KERDIM, &KERDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaMemcpy(d_ker, kernel, ker_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat, matrix, mat_size, cudaMemcpyHostToDevice);

  newConvBinWBinI<Word><<<blocks, threads>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res);

  //copyResult back to host
  cudaMemcpy(result, d_res, res_size, cudaMemcpyDeviceToHost);

  cudaFree(d_MATDIM);
  cudaFree(d_KERDIM);
  cudaFree(d_mat);
  cudaFree(d_ker);
  cudaFree(d_res);
}


void convBinWBinIWrapper_Old(uint32_t blocks, uint32_t threads, uint32_t MATDIM, uint32_t KERDIM, unsigned char* matrix, unsigned char* kernel, unsigned char* result)
{
  //Calculate Array size
  uint32_t res_size = (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) * sizeof(unsigned char);
  uint32_t mat_size = (uint32_t) ceil(MATDIM*MATDIM/8.0) * sizeof(unsigned char);
	uint32_t ker_size = (uint32_t) ceil(KERDIM*KERDIM/8.0) * sizeof(unsigned char);

  //Mallocate device data
  uint32_t *d_MATDIM, *d_KERDIM;
  cudaMalloc((void**) &d_MATDIM, sizeof(uint32_t));
  cudaMalloc((void**) &d_KERDIM, sizeof(uint32_t));

  unsigned char *d_mat, *d_ker, *d_res;
  cudaMalloc((void**) &d_mat, mat_size);
	cudaMalloc((void**) &d_ker, ker_size);
  cudaMalloc((void**) &d_res, res_size);

  //Copy input to device
  cudaMemcpy(d_MATDIM, &MATDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_KERDIM, &KERDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaMemcpy(d_ker, kernel, ker_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat, matrix, mat_size, cudaMemcpyHostToDevice);

  convBinWBinI<<<blocks, threads>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res);

  //copyResult back to host
  cudaMemcpy(result, d_res, res_size, cudaMemcpyDeviceToHost);

  cudaFree(d_MATDIM);
  cudaFree(d_KERDIM);
  cudaFree(d_mat);
  cudaFree(d_ker);
  cudaFree(d_res);
}

TEST(binIbinW, M1K1)
{
  //Input Data
  uint32_t MATDIM = 1;
  uint32_t KERDIM = 1;
  unsigned char matrix[1] = {0b10000000};
  unsigned char kernel[1] = {0b10000000};
  uint32_t result[1];

  convBinWBinIWrapper<unsigned char>(1, 1, MATDIM, KERDIM, matrix, kernel, result);

  ASSERT_EQ(1, result[0]);
}

TEST(binIbinW, M5K3)
{
  uint32_t MATDIM = 5;
  uint32_t KERDIM = 3;
  /*  Matrix        Kernel
      11010        101
      10001        011
      00110        001
      11111
      00000
    */
  unsigned char matrix[4] = { 0b11010100, 0b01001101, 0b11110000, 0b00000000 };
  unsigned char kernel[2] = {0b10101100, 0b10000000};
  uint32_t result[9];

  convBinWBinIWrapper<unsigned char>(1, 9, MATDIM, KERDIM, matrix, kernel, result);

  uint32_t suspected_result[9] = {4, 6, 2, 5, 5,  4, 6, 5, 5};

  for(int i = 0; i < 9; i++)
      ASSERT_EQ(suspected_result[i], result[i]);
}

TEST(binIbinW, M11K9)
{
  uint32_t MATDIM = 11;
  uint32_t KERDIM = 9;
  unsigned char matrix[16] = {0b10011010, 0b10101011, 0b00010110, 0b11010110, 0b11110100, 0b10111010, 0b11101010, 0b00101010,
                              0b11100110, 0b10110001, 0b10101100, 0b10001011, 0b10001011, 0b10100011, 0b01010110, 0b11111111};
  unsigned char kernel[11] = {0b00110101, 0b01101100, 0b11100100, 0b11100010, 0b10001101, 0b00110110, 0b10110111, 0b01110010, 0b01011000, 0b10101010, 0b11111111};
  uint32_t result[9];

  convBinWBinIWrapper<unsigned char>(1, 9, MATDIM, KERDIM, matrix, kernel, result);

  uint32_t suspected_result[9] = {41, 40, 36, 33, 43, 39, 45, 36, 44};
  for(int i = 0; i < 9; i++)
    ASSERT_EQ(suspected_result[i], result[i]);
}

TEST(binIbinW_WordLength32, M3K3)
{
  uint32_t MATDIM = 3;
  uint32_t KERDIM = 3;
  uint32_t matrix[1] = {0b10111101011111111111111111111111};
  uint32_t kernel[1] = {0b10111101011111111111111111111111};
  uint32_t result[1];

  convBinWBinIWrapper<uint32_t>(1, 4, MATDIM, KERDIM, matrix, kernel, result);

  ASSERT_EQ(9, result[0]);
}

TEST(binIbinW_WordLength32, M6K5)
{
  uint32_t MATDIM = 6;
  uint32_t KERDIM = 5;
  uint32_t matrix[2] = {0b10111101000100111010110101001011, 0b10000000000000000000000000000000}; //;
  uint32_t kernel[1] = {0b01010111100110000111101100000000};
  uint32_t result[4];

  convBinWBinIWrapper<uint32_t>(1, 4, MATDIM, KERDIM, matrix, kernel, result);

  uint32_t suspected_result[4] = {8, 15, 13, 13};
  for(int i = 0; i < 4; i++)
    ASSERT_EQ(suspected_result[i], result[i]);
}

TEST(binIbinW, OldAsOracle)
{
  srand(time(0));
  for(int j = 0; j < 10; j++)
  {
    uint32_t MATDIM = rand() % 7 + 1; //Oracle only works for small matrices
    uint32_t KERDIM = (MATDIM == 1) ? 1 : 0;
    while( (KERDIM%2) == 0 || KERDIM == 7)
      KERDIM = rand() %  MATDIM + 1;

    std::cerr << "[          ] " << MATDIM << " " << KERDIM << "\n";

    double* double_matrix = new double[ MATDIM * MATDIM * sizeof(double)];
    double* double_kernel = new double[ KERDIM * KERDIM * sizeof(double)];

    initMat(MATDIM, double_matrix);
    initMat(KERDIM, double_kernel);

    unsigned char* bin_matrix = new unsigned char[ (uint32_t)ceil(MATDIM*MATDIM/8.0) ];
    unsigned char* bin_kernel = new unsigned char[ (uint32_t)ceil(KERDIM*KERDIM/8.0) ];
    uint32_t* result = new uint32_t[ (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) ];
    unsigned char* suspected_result = new unsigned char[ (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) ];

    convertToBinary(MATDIM, double_matrix, (uint32_t)ceil(MATDIM*MATDIM/8.0), bin_matrix);
    convertToBinary(KERDIM, double_kernel, (uint32_t)ceil(KERDIM*KERDIM/8.0), bin_kernel);

    uint32_t N = 10;
    uint32_t grid_size = ceil((MATDIM-KERDIM+1) * (MATDIM-KERDIM+1) / ((double) N));

    convBinWBinIWrapper_Old(grid_size, N, MATDIM, KERDIM, bin_matrix, bin_kernel, suspected_result);
    convBinWBinIWrapper<unsigned char>(grid_size, N, MATDIM, KERDIM, bin_matrix, bin_kernel, result);

    for(int i=0; i < (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1); i++)
      ASSERT_EQ(suspected_result[i], (unsigned char)result[i]);
  }
}


TEST(binW, M16K4){

	double mat[16] = {0.178184, -0.368889, -0.0857587, -0.256823, -0.974276, -0.575986, -0.416872, 0.450409, -0.984626, -0.933976, -0.726241, -0.354601, 0.644028, -0.586327, 0.37183, 0.765458};
	unsigned char ker = 0b11100000;
  double result[9];

	convBinWWrapper(4, 2, mat, result, &ker);

  double suspected_result[9] = {-0.588995, -1.209863, -1.600912, 0.203936, -1.200593, -1.532846, -0.338103, -0.688247, -1.474470};

  for(int i=0; i < 9; i++)
    ASSERT_NEAR(suspected_result[i], result[i], 0.00005);

}

TEST(binW, M16K1)
{
  double mat[16] = {-0.510971, -0.247672, -0.859489, -0.390785, -0.75706, -0.718692, 0.292791, 0.0486137, -0.855771, 0.384334, 0.623243, 0.445734, -0.741026, -0.262363, 0.357182, 0.869584};
  unsigned char ker = 0b00000000;
	double result[16];

  convBinWWrapper(4, 1, mat, result, &ker);

  double suspected_result[16];

	for (int i = 0; i < 16; i++) {
		suspected_result[i] = -mat[i];
	}

  for(int i=0; i < 9; i++)
    ASSERT_DOUBLE_EQ(suspected_result[i], result[i]);

}

TEST(binW, M16K1_2)
{
  double mat[16] = {-0.510971, -0.247672, -0.859489, -0.390785, -0.75706, -0.718692, 0.292791, 0.0486137, -0.855771, 0.384334, 0.623243, 0.445734, -0.741026, -0.262363, 0.357182, 0.869584};
  unsigned char ker = 0b10000000;
  double result[16];

  convBinWWrapper(4, 1, mat, result, &ker);

  double suspected_result[16];

	for (int i = 0; i < 16; i++) {
		suspected_result[i] = mat[i];
	}

  for(int i=0; i < 9; i++)
    ASSERT_DOUBLE_EQ(suspected_result[i], result[i]);

}

TEST(binW, M16K9)
{
  double mat[16] = {0.0803077, 0.0543088, -0.874869, -0.654904, -0.954831, -0.186203, -0.744362, 0.749519, -0.921765, 0.78594, -0.946144, -0.519783, -0.275407, 0.301698, -0.0470979, -0.799294};
	unsigned char ker[2] = {0b00100100, 0b00000000};
  double result[4];

  convBinWWrapper(4, 3, mat, result, ker);

  double suspected_result [4] = {0.469155, 2.87521, 1.38936, 1.8652};

  for(int i=0; i < 4; i++)
    ASSERT_NEAR (suspected_result[i], result[i], 0.00005);

}
