#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "convolution.h"
#include "array_management.h"

template <typename Word>
void convBinWBinIWrapper(uint32_t blocks, uint32_t threads, uint32_t MATDIM, uint32_t KERDIM, Word* matrix, Word* kernel, unsigned char* result)
{
  double word_length = sizeof(Word) * 8.0;
  //Calculate Array size
  uint32_t res_size = (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) * sizeof(unsigned char);
  uint32_t mat_size = (uint32_t) ceil(MATDIM*MATDIM/word_length) * sizeof(Word);
	uint32_t ker_size = (uint32_t) ceil(KERDIM*KERDIM/word_length) * sizeof(Word);

  //Mallocate device data
  uint32_t *d_MATDIM, *d_KERDIM;
  cudaMalloc((void**) &d_MATDIM, sizeof(uint32_t));
  cudaMalloc((void**) &d_KERDIM, sizeof(uint32_t));

  Word *d_mat, *d_ker;
  unsigned char *d_res;
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
  unsigned char result[1];

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
  unsigned char result[9];

  convBinWBinIWrapper<unsigned char>(1, 9, MATDIM, KERDIM, matrix, kernel, result);

  unsigned char suspected_result[9] = {4, 6, 2, 5, 5,  4, 6, 5, 5};

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
  unsigned char result[9];

  convBinWBinIWrapper<unsigned char>(1, 9, MATDIM, KERDIM, matrix, kernel, result);

  unsigned char suspected_result[9] = {41, 40, 36, 33, 43, 39, 45, 36, 44};
  for(int i = 0; i < 9; i++)
    ASSERT_EQ(suspected_result[i], result[i]);
}

TEST(binIbinW_WordLength32, M3K3)
{
  uint32_t MATDIM = 3;
  uint32_t KERDIM = 3;
  uint32_t matrix[1] = {0b10111101011111111111111111111111};
  uint32_t kernel[1] = {0b10111101011111111111111111111111};
  unsigned char result[1];

  convBinWBinIWrapper<uint32_t>(1, 4, MATDIM, KERDIM, matrix, kernel, result);

  ASSERT_EQ(9, result[0]);
}

TEST(binIbinW_WordLength32, M6K5)
{
  uint32_t MATDIM = 6;
  uint32_t KERDIM = 5;
  uint32_t matrix[2] = {0b10111101000100111010110101001011, 0b10000000000000000000000000000000}; //;
  uint32_t kernel[1] = {0b01010111100110000111101100000000};
  unsigned char result[4];

  convBinWBinIWrapper<uint32_t>(1, 4, MATDIM, KERDIM, matrix, kernel, result);

  unsigned char suspected_result[4] = {8, 15, 13, 13};
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
    unsigned char* result = new unsigned char[ (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) ];
    unsigned char* suspected_result = new unsigned char[ (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1) ];

    convertToBinary(MATDIM, double_matrix, (uint32_t)ceil(MATDIM*MATDIM/8.0), bin_matrix);
    convertToBinary(KERDIM, double_kernel, (uint32_t)ceil(KERDIM*KERDIM/8.0), bin_kernel);

    uint32_t N = 10;
    uint32_t grid_size = ceil((MATDIM-KERDIM+1) * (MATDIM-KERDIM+1) / ((double) N));

    convBinWBinIWrapper_Old(grid_size, N, MATDIM, KERDIM, bin_matrix, bin_kernel, suspected_result);
    convBinWBinIWrapper<unsigned char>(grid_size, N, MATDIM, KERDIM, bin_matrix, bin_kernel, result);

    for(int i=0; i < (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1); i++)
      ASSERT_EQ(suspected_result[i], result[i]);
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
						cudaFree(d_ker_bin);
						cudaFree(d_mat);
						cudaFree(d_res_binW);
						cudaFree(d_MATDIM);
						cudaFree(d_KERDIM);
						return;
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
