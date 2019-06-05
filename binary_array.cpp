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

unsigned char binary_array::popc() const
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

void binary_array::print(char x)
{
  for(int i = 7; i >= 0; i--)
    cout<<(x>>i&0x01)<<" ";
  cout<<"\n";
}

void binary_array::print(char x, const char* message)
{
  for(int i = 7; i >= 0; i--)
    cout<<(x>>i&0x01)<<" ";
  cout<<" "<<message<<"\n";
}

binary_array binary_array::xnor_mask(binary_array &mask, int midpoint_index)
{
  // if(size < midpoint_index + (mask.row_size * mask_col_count))
  //   throw out_of_range("The mask does not fit over the matrix");

  //Calculate frequently used data
  binary_array result(mask.size, mask.row_count);
  unsigned char kernel_byte = 0;
  unsigned char matrix_byte = 0;
  //Iterate ofter the maks columns
  for(int row = -(mask.row_count/2); row <= (int)mask.row_count/2  ; row++)
  {
    int lh_matrix_idx = midpoint_index+(row*row_size) - mask.row_size/2;
    int lh_kernel_idx = (row+mask.row_count/2)*mask.row_size;
    //Iterate over the bytes in one collumn
    for(unsigned int byte = 0; byte < mask.row_size/8; byte++)
    {
      kernel_byte = mask.get_byte(lh_kernel_idx+8*byte, lh_kernel_idx+8*byte+8);
      matrix_byte = get_byte(lh_matrix_idx+8*byte, lh_matrix_idx+8*byte+8);
      result.set_byte(lh_kernel_idx+8*byte, lh_kernel_idx+8*byte+8, ~(kernel_byte^ matrix_byte));
    }
    unsigned int byte = mask.row_size/8;
    unsigned int offset = (mask.row_size%8)-1;
    kernel_byte = mask.get_byte(lh_kernel_idx+8*byte, lh_kernel_idx+8*byte+offset);
    matrix_byte = get_byte(lh_matrix_idx+8*byte, lh_matrix_idx+8*byte+offset);
    print(~(kernel_byte^ matrix_byte));
    result.set_byte(lh_kernel_idx+8*byte, lh_kernel_idx+8*byte+offset, ~(kernel_byte^ matrix_byte));
  }

  return result;
}

unsigned char binary_array::get_byte(int lh_idx, int rh_idx)
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
    lh_byte &= 0xFF<<(7-(rh_idx-lh_idx));
    rh_byte = (rh_byte>>(7-rh_idx%8))<<((lh_idx%8)-(rh_idx%8+1));
  }
  return lh_byte | rh_byte;
}

void binary_array::set_byte(int lh_idx, int rh_idx, unsigned char value)
{
  if(lh_idx > rh_idx || rh_idx-lh_idx > 7|| rh_idx >= size)
    throw out_of_range("Invalid argumnts");
  unsigned char lh_byte = array[lh_idx/8];
  unsigned char rh_byte = array[rh_idx/8];

  if(rh_idx-lh_idx < 7)
    value &= 0xFF<<(7-(rh_idx-lh_idx));

  if(lh_idx/8 == rh_idx/8)
  {
    lh_byte &= (0xFF<<(8-lh_idx%8)) | (0xFF>>(rh_idx%8+1));
    lh_byte |= value>>(lh_idx%8);
  }
  else
  {
    lh_byte &= 0xFF<<(8-lh_idx%8);
    lh_byte |= value>>(lh_idx%8);
    rh_byte &= 0xFF>>(rh_idx%8+1);
    rh_byte |= value<<(8-lh_idx%8);
    array[rh_idx/8] = rh_byte;
  }

  array[lh_idx/8] = lh_byte;
}





//end
