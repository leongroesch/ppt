#include "binary_array.h"

using namespace std;

binary_array::binary_array(int _size, unsigned int _row_count)
: size(_size), byte_size(ceil(size/8.0)), array(byte_size), row_count(_row_count)
{

}
binary_array::binary_array(unsigned char* _array, unsigned int _size, unsigned int _row_count)
: size(_size), byte_size(ceil(size/8.0)), array(byte_size), row_count(_row_count)
{
  for(int i = byte_size-1; i >= 0; i++)
  {
    array[i] = _array[i];
  }

}

binary_array::binary_array(vector<int> _array, int _size, unsigned int _row_count)
: size(_size), byte_size(ceil(size/8.0)), array(byte_size), row_count(_row_count)
{
  for(int i = _array.size()-1; i >=0 ; i--)
  {
    array[i] = ((char)_array[i]);
  }
}

int binary_array::popc() const
{
  int count = 0;
  int byte = 0;
  int bit = 0;
  for(int i = size-1; i >= 0; i--)
  {
    byte = floor(i/8);
    bit = i%8;
    if(((array[byte]>>bit)&0x01)) count++;
  }
  return count;
}

void binary_array::print() const
{
  int byte;
  int bit;
  for(int i = size-1; i >= 0; i--)
  {
    byte = floor(i/8);
    bit = i%8;
    cout<<((array[byte]>>bit)&0x01)<<" ";
  }
  cout<<"\n\n";
}

void binary_array::print(char x) const
{
  for(int i = 8; i >= 0; i--)
    cout<<(x>>i&0x01)<<" ";
  cout<<"\n";
}

friend binary_array binary_array::xnor_mask(binary_array &mask, int midpoint_index) const
{
  if(size < tl_index + (mask.row_size * mask_col_count))
    throw out_of_range("The mask does not fit over the matrix");

  //Calculate frequently used data
  int kernel_offset = 0;
  //Iterate ofter the maks columns
  char arry_operand = 0;
  char kernel_operand = 0;
  for(int row = -(maks.row_count/2); row < mask.row_count/2  ; i++)
  {
    int lh_matrix_idx = midpoint_index+(row*row_size) - mask.row_size/2;
    //<<<To Do>>>
    int lh_kernel_idx = row
    //Iterate over the bytes in one collumn
    for(int byte = 0; byte < to_byte(mask.row_size)-1; byte++)
    {
      if((lh_matrix_idx%8) != 0)
        array_operand = array[lh_matrix_idx/8+byte]<<(lh_matrix_idx%8) | array[lh_matrix_idx/8+byte+1] >> 8-(lh_matrix_idx%8);
      else
        array_operand = array[lh_matrix_idx/8+byte];
      kernel_operand = mask.array[byte]


    }
    if(mask.row_size%8 == 0)
      array_operand = array[lh_matrix_idx/8+to_byte(mask.row_size)];
    else
      array_operand = (array[lh_matrix_idx/8 + mask.row_size/8]<<8-mask.row_size%8  | array[lh_matrix_idx/8+to_byte(mask.row_size)]);
  }

  return result;
}







//end
