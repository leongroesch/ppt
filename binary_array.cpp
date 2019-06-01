#include "binary_array.h"

using namespace std;

binary_array::binary_array(int _size, unsigned int _row_count)
: size(_size), byte_count(ceil(size/8.0)), array(byte_count), row_count(_row_count)
{
  row_size = size/row_count;
}
binary_array::binary_array(unsigned char* _array, unsigned int _size, unsigned int _row_count)
: size(_size), byte_count(ceil(size/8.0)), array(byte_count), row_count(_row_count)
{
  row_size = size/row_count;
  for(int i = 0; i < byte_count; i++)
  {
    array[i] = _array[i];
  }

}

binary_array::binary_array(vector<int> _array, int _size, unsigned int _row_count)
: size(_size), byte_count(ceil(size/8.0)), array(byte_count), row_count(_row_count)
{
  row_size = size/row_count;
  for(unsigned int i = 0; i < _array.size() ; i++)
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
  for(int i = 0; i < size; i++)
  {
    if(i%row_size == 0)
      cout<<"\n";
    byte = i/8;
    bit = 7-(i%8);
    cout<<((array[byte]>>bit)&0x01)<<" ";
  }
  cout<<"\n\n";
}

void binary_array::print(char x) const
{
  for(int i = 7; i >= 0; i--)
    cout<<(x>>i&0x01)<<" ";
  cout<<"\n";
}

binary_array binary_array::xnor_mask(binary_array &mask, int midpoint_index)
{
  // if(size < midpoint_index + (mask.row_size * mask_col_count))
  //   throw out_of_range("The mask does not fit over the matrix");

  //Calculate frequently used data
  binary_array result(mask.size, mask.row_count);
  //Iterate ofter the maks columns
  unsigned char array_operand = 0;
  unsigned char kernel_operand = 0;
  for(int row = -(mask.row_count/2); row <= (int)mask.row_count/2  ; row++)
  {
    int lh_matrix_idx = midpoint_index+(row*row_size) - mask.row_size/2;
    int lh_kernel_idx = (row+mask.row_count/2)*mask.row_size;
    //Iterate over the bytes in one collumn
    for(unsigned int byte = 0; byte < to_byte(mask.row_size)-1; byte++)
    {
      if((lh_matrix_idx%8) != 0)
        array_operand = array[lh_matrix_idx/8+byte]<<(lh_matrix_idx%8) | array[lh_matrix_idx/8+byte+1] >> (8-lh_matrix_idx%8);
      else
        array_operand = array[lh_matrix_idx/8+byte];
      kernel_operand = mask.array[to_byte(lh_kernel_idx)+byte];
      cout<<"kernel_operand in for: ";
      print(kernel_operand);
      cout<<"array operand in for: ";
      print(array_operand);
      result[to_byte(lh_kernel_idx)+byte] = ~(array_operand^kernel_operand);
    }
    if(mask.row_size%8 == 0)
      array_operand = array[lh_matrix_idx/8+to_byte(mask.row_size)];
    else if(mask.row_size > 8)
      array_operand = (array[lh_matrix_idx/8 + mask.row_size/8]<<(8-mask.row_size%8)  | array[lh_matrix_idx/8+to_byte(mask.row_size)]);
    else
    {
      array_operand = array[lh_matrix_idx/8 + mask.row_size/8];
      array_operand = array_operand<<(lh_matrix_idx%8);
      array_operand = array_operand>>(8-(mask.row_size%8));
    }
    kernel_operand = (mask.array[lh_kernel_idx/8+mask.row_size/8]<<lh_kernel_idx%8)>>(8-(mask.row_size%8));
    cout<<"kernel_operand : ";
    print(kernel_operand);
    cout<<"array operand : ";
    print(array_operand);
    result[to_byte(lh_kernel_idx)+to_byte(mask.row_size)] = ~(array_operand^kernel_operand);
  }

  return result;
}

unsigned char binary_array::byte(int lh_idx, int rh_idx)
{
  if(lh_idx > rh_idx || rh_idx-lh_idx > 7|| rh_idx >= size)
    throw out_of_range("Invalid argumnts");
  unsigned char lh_byte = array[lh_idx/8];
  unsigned char rh_byte = array[rh_idx/8];
  if(rh_idx-lh_idx == 7)
  {
    lh_byte = lh_byte<<lh_idx%8;
    rh_byte = rh_byte>>(8-lh_idx%8);
  }
  else
  {
    lh_byte = lh_byte<<lh_idx%8;
    lh_byte &= 0xFF<<(rh_idx-lh_idx+1);
    rh_byte = (rh_byte>>(7-rh_idx%8))<<((lh_idx%8)-(rh_idx%8+1));
  }
  return lh_byte | rh_byte;
}






//end
