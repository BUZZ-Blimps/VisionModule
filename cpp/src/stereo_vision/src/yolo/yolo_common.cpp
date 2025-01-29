#include "yolo_common.hpp"

#include <iostream>
#include <fstream>
#include <cstdlib>                  // for malloc and free

/*-------------------------------------------
                  Functions
-------------------------------------------*/

void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("\tindex=%d, name=%s, \n\t\tn_dims=%d, dims=[%d, %d, %d, %d], \n\t\tn_elems=%d, size=%d, fmt=%s, \n\t\ttype=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// Function to read binary file into a buffer allocated in memory
unsigned char* load_model(const char* filename, int& fileSize)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate); // Open file in binary mode and seek to the end

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    fileSize = (int) file.tellg(); // Get the file size
    file.seekg(0, std::ios::beg); // Seek back to the beginning

    char* buffer = (char*)malloc(fileSize); // Allocate memory for the buffer

    if (!buffer) {
        std::cerr << "Memory allocation failed." << std::endl;
        return nullptr;
    }

    file.read(buffer, fileSize); // Read the entire file into the buffer
    file.close(); // Close the file

    return (unsigned char*) buffer;
}

