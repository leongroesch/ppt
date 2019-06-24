#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include "convolution.h"
#include "array_management.h"

void convBinWBinIWrapper(uint32_t blocks, uint32_t threads, uint32_t MATDIM, uint32_t KERDIM, unsigned char* matrix, unsigned char* kernel, unsigned char* result, bool old)
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

  if(old)
    convBinWBinI<<<blocks, threads>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res);
  else
    newConvBinWBinI<<<blocks, threads>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res);

  //copyResult back to host
  cudaMemcpy(result, d_res, res_size, cudaMemcpyDeviceToHost);

  cudaFree(d_MATDIM);
  cudaFree(d_KERDIM);
  cudaFree(d_mat);
  cudaFree(d_ker);
  cudaFree(d_res);
}



TEST(binIbinW_Test, OneElement)
{
  //Input Data
  uint32_t MATDIM = 1;
  uint32_t KERDIM = 1;
  unsigned char matrix[1] = {0b10000000};
  unsigned char kernel[1] = {0b10000000};
  unsigned char result[1];

  convBinWBinIWrapper(1, 1, MATDIM, KERDIM, matrix, kernel, result, false);

  ASSERT_EQ(result[0], 1);
}

TEST(binIbinW_Test, 5Matrix3Kernel)
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

  convBinWBinIWrapper(1, 9, MATDIM, KERDIM, matrix, kernel, result, false);

  unsigned char suspacted_result[9] = {4, 6, 2, 5, 5,  4, 6, 5, 5};

  for(int i = 0; i < 9; i++)
      ASSERT_EQ(result[i], suspacted_result[i]);
}

TEST(binIBinW_Test, OldAsOracle)
{
  uint32_t MATDIM = ceil( (rand()/RAND_MAX) * 1024);
  uint32_t KERDIM = 1;
  while(KERDIM%2)
    KERDIM = ceil( (rand()/RAND_MAX) * MATDIM);

  std::cerr << "[          ]" << MATDIM << " " << KERDIM << "\n";

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

  convBinWBinIWrapper(grid_size, N, MATDIM, KERDIM, bin_matrix, bin_kernel, suspected_result, true);
  convBinWBinIWrapper(grid_size, N, MATDIM, KERDIM, bin_matrix, bin_kernel, result, false);

  for(int i=0; i < (MATDIM-KERDIM+1)*(MATDIM-KERDIM+1); i++)
    ASSERT_EQ(result[i], suspected_result[i]);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
