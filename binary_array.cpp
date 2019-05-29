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

friend binary_array binary_array::xnor_mask(binary_array &mask, int rh_index) const
{
  if(size < rh_index + mask.size)
    throw out_of_range("Bal");

  int rh_byte = rh_index / 8;
  int rh_bit = rh_index % 8;
  int pop_count = 0;
  binary_array result(mask.size);

  cout<<mask.number_of_bytes<<" "<<mask.size<<"\n";
  char operation_byte = 0;
  for(int byte = 0; byte < mask.number_of_bytes; byte++)
  {
    operation_byte = (array[byte+rh_byte]>>rh_bit) | (array[byte+rh_byte+1]<<8-rh_bit);
    operation_byte = ~(operation_byte^mask.array[byte]);
    result[byte] = operation_byte;
  }

  return result;
}



//end
