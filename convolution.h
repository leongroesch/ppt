#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <cuda.h>
#include <math.h>

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
					if ((((ker[index_ker >> 3] << (index_ker % 8)) >> 7)&0x01) == (((mat[index_mat_bin >> 3] << (index_mat_bin % 8)) >> 7)&0x01))
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

//Leong: Byte wise binary/binary convolution
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

#endif
