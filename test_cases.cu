#include <gtest/gtest.h>
#include "convolution.h"

TEST(binIbinW_Test, OneElement)
{
  //Mallocate Host data
  uint32_t h_KERDIM = 1;
  uint32_t h_MATDIM = 1;
  unsigned char h_mat[1] = {0b10000000};
  unsigned char h_ker[1] = {0b10000000};
  unsigned char h_res[1];

  //Calculate Array size
  uint32_t res_size = (h_MATDIM-h_KERDIM+1)*(h_MATDIM-h_KERDIM+1) * sizeof(unsigned char);
  uint32_t mat_size = (uint32_t) ceil(h_MATDIM*h_MATDIM/8.0) * sizeof(unsigned char);
	uint32_t ker_size = (uint32_t) ceil(h_KERDIM*h_KERDIM/8.0) * sizeof(unsigned char);

  //Mallocate device data
  uint32_t *d_MATDIM, *d_KERDIM;
  cudaMalloc((void**) &d_MATDIM, sizeof(uint32_t));
  cudaMalloc((void**) &d_KERDIM, sizeof(uint32_t));

  unsigned char *d_mat, *d_ker, *d_res;
  cudaMalloc((void**) &d_mat, mat_size);
	cudaMalloc((void**) &d_ker, ker_size);
  cudaMalloc((void**) &d_res, res_size);

  //Copy input to device
  cudaMemcpy(d_MATDIM, &h_MATDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_KERDIM, &h_KERDIM, sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaMemcpy(d_ker, h_ker, ker_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat, h_mat, mat_size, cudaMemcpyHostToDevice);

  newConvBinWBinI<<<1, 1>>>(d_MATDIM, d_KERDIM, d_mat, d_ker, d_res);

  //copyResult back to host
  cudaMemcpy(h_res, d_res, res_size, cudaMemcpyDeviceToHost);

  cudaFree(d_MATDIM);
  cudaFree(d_KERDIM);
  cudaFree(d_mat);
  cudaFree(d_ker);
  cudaFree(d_res);

  ASSERT_EQ(h_res[0], 1);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
