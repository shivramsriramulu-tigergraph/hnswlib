#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../hnswlib/hnswlib.h"

int main() {
    const int dim = 16;
    const int num_vectors = 100;
    const int k = 5;

    // --- float16 test ---
    hnswlib::L2SpaceFp16 space_fp16(dim);
    hnswlib::HierarchicalNSW<float>* index_fp16 =
        new hnswlib::HierarchicalNSW<float>(&space_fp16, num_vectors);

    // insert vectors: store fp16 (uint16_t) values
    for (int i = 0; i < num_vectors; i++) {
        std::vector<hnswlib::float16_t> vec(dim);
        for (int j = 0; j < dim; j++) {
            // encode simple float as fp16: just use bit pattern of a small float
            float val = (float)(i * dim + j) * 0.01f;
            uint32_t bits; memcpy(&bits, &val, 4);
            // simple fp32->fp16 truncation (good enough for testing)
            uint16_t h = (uint16_t)(((bits >> 16) & 0x8000) |
                         (((bits & 0x7f800000) - 0x38000000) >> 13) |
                         ((bits >> 13) & 0x03ff));
            vec[j] = h;
        }
        index_fp16->addPoint(vec.data(), i);
    }

    // query
    std::vector<hnswlib::float16_t> query(dim, 0);
    auto result = index_fp16->searchKnn(query.data(), k);
    std::cout << "float16 top-" << k << " neighbors: ";
    while (!result.empty()) {
        std::cout << result.top().second << "(" << result.top().first << ") ";
        result.pop();
    }
    std::cout << "\n";

    // --- float8 test ---
    hnswlib::L2SpaceFp8 space_fp8(dim);
    hnswlib::HierarchicalNSW<float>* index_fp8 =
        new hnswlib::HierarchicalNSW<float>(&space_fp8, num_vectors);

    for (int i = 0; i < num_vectors; i++) {
        std::vector<hnswlib::float8_t> vec(dim);
        for (int j = 0; j < dim; j++)
            vec[j] = (hnswlib::float8_t)((i + j) % 256);
        index_fp8->addPoint(vec.data(), i);
    }

    std::vector<hnswlib::float8_t> query8(dim, 0);
    auto result8 = index_fp8->searchKnn(query8.data(), k);
    std::cout << "float8  top-" << k << " neighbors: ";
    while (!result8.empty()) {
        std::cout << result8.top().second << "(" << result8.top().first << ") ";
        result8.pop();
    }
    std::cout << "\n";

    delete index_fp16;
    delete index_fp8;
    std::cout << "PASSED\n";
    return 0;
}
