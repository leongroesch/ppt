#ifndef _BINARY_ARRAY_
#define _BINARY_ARRAY_

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

class binary_array
{
public:
  int size;
  int byte_count;
  std::vector<unsigned char> array;
  unsigned int row_count;
  unsigned int row_size;


  inline unsigned int to_byte(int bit){ return ceil(bit/8.0);}

  binary_array(int _size, unsigned int _row_count);
  binary_array(unsigned char* _array, unsigned int _size, unsigned int _row_count);
  binary_array(std::vector<int>  _array, int _size, unsigned int _row_count);

  // int get_size() const {return size;};
  // unsigned get_row_count() const {return row_count;};
  int popc() const;
  void print() const;
  void print(char x) const;
  void push_back(char value) {array.push_back(value);} ;
  binary_array xnor_mask(binary_array &mask, int midpoint_index);
  unsigned char byte(int lh_idx, int rh_idx);
  unsigned char& operator[](int index){return array[index];}
};

#endif
