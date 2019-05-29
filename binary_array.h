#ifndef _BINARY_ARRAY_
#define _BINARY_ARRAY_

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

class binary_array
{
private:
  std::vector<char> array;
  int size;
  int number_of_bytes;
  unsigned int row_count;
  unsigned int row_size;
  unsigned int col_size;

public:
  binary_array(int _size, unsigned int _row_count);
  binary_array(unsigned char* _array, unsigned int _size, unsigned int _row_count);
  binary_array(std::vector<int>  _array, int _size, unsigned int _row_count);

  // int get_size() const {return size;};
  // unsigned get_row_count() const {return row_count;};
  int popc() const;
  void print() const;
  void print(char x) const;
  void push_back(char value) {array.push_back(value);} ;
  friend binary_array xnor_mask(binary_array &mask, int rh_index) const;
  char& operator[](int index){return array[index];}
};

#endif
