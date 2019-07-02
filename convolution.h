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
			uint32_t kerPosByte = (int)(i / 8);
			uint32_t kerPosBit = 7 - i % 8;
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

template<typename Word>
__device__
void get_word(uint32_t* d_MATDIM, Word* matrix, Word* result, uint32_t lh_idx, uint32_t rh_idx)
{
	uint32_t word_length = sizeof(Word) * 8;
  Word lh_word = matrix[lh_idx/word_length];
  Word rh_word = matrix[rh_idx/word_length];
  if(rh_idx-lh_idx == (word_length-1) )
  {
    lh_word = lh_word<<lh_idx%word_length;
    rh_word = rh_word>>(word_length-lh_idx%word_length);
  }
  else
  {
    lh_word = lh_word<<lh_idx%word_length;
    lh_word &= 0xFF<<((word_length-1)-(rh_idx-lh_idx));
    rh_word = (rh_word>>((word_length-1)-rh_idx%word_length))<<((lh_idx%word_length)-(rh_idx%word_length+1));
  }
  *result = lh_word | rh_word;
}

//Leong: Byte wise binary/binary convolution
template<typename Word>
__global__
void newConvBinWBinI(uint32_t* d_MATDIM, uint32_t* d_KERDIM, Word* matrix, Word* kernel, uint32_t* result){
	uint32_t KERDIM = d_KERDIM[0];
	uint32_t MATDIM = d_MATDIM[0];
	uint32_t word_length = sizeof(Word) * 8;
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	result[thread_id] = 0;
	uint32_t midpoint_index = KERDIM/2 + (KERDIM/2) * MATDIM + (thread_id / (MATDIM-KERDIM+1)) * MATDIM + thread_id % (MATDIM-KERDIM+1);
	Word kernel_word = 0;
	Word matrix_word = 0;
	Word result_word = 0;
	int lh_matrix_idx = 0;
	int lh_kernel_idx = 0;
	uint32_t offset = 0;

	//Iterate ofter the maks columns
	for(int row = -(KERDIM/2); row <= (int)KERDIM/2  ; row++)
	{
		lh_matrix_idx = midpoint_index+(row*MATDIM) - KERDIM/2;
		lh_kernel_idx = (row+KERDIM/2)*KERDIM;
		//Iterate over the bytes in one collumn
		for(unsigned int byte = 0; byte <= KERDIM/word_length; byte++)
		{
			if(byte == KERDIM/word_length)
			 offset =	(KERDIM%word_length)-1;
			else
				offset = word_length-1;
		  get_word<Word>(d_KERDIM, kernel, &kernel_word, lh_kernel_idx+word_length*byte, lh_kernel_idx+word_length*byte+offset);
			get_word<Word>(d_MATDIM, matrix, &matrix_word, lh_matrix_idx+word_length*byte, lh_matrix_idx+word_length*byte+offset);
			//XNOR
			result_word = ~(kernel_word^matrix_word);
			//Only use the x left most bits from the last byte
			if(byte == KERDIM/word_length)
				result_word = ((0XFF<<(word_length-1-offset)) & result_word);
			result[thread_id] += __popc(result_word);
		}
	}

}

#endif
