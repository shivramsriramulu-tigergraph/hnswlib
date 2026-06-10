#ifndef HNSWLIB_HNSWLIB_INTERFACE_H_
#define HNSWLIB_HNSWLIB_INTERFACE_H_

#include <iostream>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <cstring>
#include <assert.h>
#include <functional>

namespace hnswlib {

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                for (size_t id = start; id < end; id++) {
                    fn(id, threadId);
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}

template<typename idtype>
class CustomFilterFunctor: public hnswlib::BaseFilterFunctor<idtype> {
    std::function<bool(idtype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(idtype)>& f) {
        filter = f;
    }

    bool operator()(idtype id) {
        return filter(id);
    }
};

template<typename idtype, typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    idtype cur_l;
    hnswlib::HierarchicalNSW<idtype, data_t>* appr_alg;
    hnswlib::SpaceInterface<data_t>* l2space;


    Index(const std::string &space_name, const int dim, const std::string& data_type = "float32") : space_name(space_name), dim(dim) {
        normalize = false;
        if (data_type != "float32" && data_type != "float16" && data_type != "float8") {
            throw std::runtime_error("data_type must be one of: float32, float16, float8");
        }
        if (space_name == "l2") {
            if (data_type == "float16")
                l2space = new hnswlib::L2SpaceFp16(dim);
            else if (data_type == "float8")
                l2space = new hnswlib::L2SpaceFp8(dim);
            else
                l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            if (data_type == "float16")
                l2space = new hnswlib::InnerProductSpaceFp16(dim);
            else if (data_type == "float8")
                l2space = new hnswlib::InnerProductSpaceFp8(dim);
            else
                l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            if (data_type == "float16")
                l2space = new hnswlib::InnerProductSpaceFp16(dim);
            else if (data_type == "float8")
                l2space = new hnswlib::InnerProductSpaceFp8(dim);
            else
                l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed = 100,
        bool allow_replace_deleted = true) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<idtype, data_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, uint32_t max_elements) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      
      appr_alg = new hnswlib::HierarchicalNSW<idtype, data_t>(l2space, path_to_index, false, max_elements, true);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    double normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        double factor_result = 1.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        factor_result = sqrtf(norm)+ 1e-30f;
        norm = 1.0f / (factor_result);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
        return factor_result;
    }


    void updateItems(const char* element_array, size_t rows, size_t features, int num_threads = -1, bool replace_deleted = false) {

        if (!index_inited)
            throw std::runtime_error("Index not inited");

        if (num_threads <= 0)
            num_threads = num_threads_default;
        
        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        size_t flag_size = sizeof(uint8_t);
        size_t id_size = sizeof(idtype); // The size of idtype
        size_t embedding_size = l2space->get_data_size();

        {
            int start = 0;
            if (!ep_added) {
                uint8_t* ptr;
                while (start < rows) {
                    ptr = (uint8_t*)(element_array + start * (flag_size + id_size + embedding_size));
                    // skip all delete records, until the first upsert record
                    if (*ptr == 1)
                        break;
                    start++;
                }
                if (start == rows) return;  // all delete records
                assert(*ptr == 1);
                ptr += flag_size;

                idtype id = *(idtype*)ptr;
                ptr += id_size;

                data_t* vector_data = (data_t*)ptr;
                std::vector<data_t> norm_array(dim);
                double normalize_factor = 1.0;
                if (normalize) {
                    normalize_factor = normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, id, replace_deleted, normalize_factor);
                start++;
                ep_added = true;
            }

            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    uint8_t* element_ptr = (uint8_t*)(element_array + row * (flag_size + id_size + embedding_size));
                    idtype id = *(idtype*)(element_ptr + flag_size);
                    if (id % num_threads == threadId) {
                        uint8_t flag = *element_ptr;
                        element_ptr += flag_size + id_size;

                        if (flag == 1) {
                            appr_alg->addPoint((void*)element_ptr, id, replace_deleted);
                        }
                        else if (flag == 0) appr_alg->markDelete(id);
                        else throw std::runtime_error("Wrong flag");
                    }
                    });
            } else {
                std::vector<data_t> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    uint8_t* element_ptr = (uint8_t*)(element_array + row * (flag_size + id_size + embedding_size));
                    idtype id = *(idtype*)(element_ptr + flag_size);
                    if (id % num_threads == threadId) {
                        uint8_t flag = *element_ptr;
                        element_ptr += flag_size + id_size;

                        if (flag == 1) {
                            double normalize_factor = 1.0f;
                            normalize_factor = normalize_vector((data_t*)element_ptr, (norm_array.data() + start_idx));
                            appr_alg->addPoint((void*)(norm_array.data() + start_idx), id, replace_deleted, normalize_factor);
                        }
                        else if (flag == 0) appr_alg->markDelete(id);
                        else throw std::runtime_error("Wrong flag");
                    }
                    });
            }
            cur_l += rows;
        }
    }

    std::priority_queue<std::pair<data_t, idtype >>
    searchKnn(const void *query_data, size_t k, size_t efSearch, BaseFilterFunctor<idtype>* isIdAllowed = nullptr) {
        if (!index_inited)
            throw std::runtime_error("Index not inited");
        if (normalize == false)
            return appr_alg->searchKnn(query_data, k, efSearch, isIdAllowed);

        std::vector<data_t> norm_array(dim);
        normalize_vector((data_t*)query_data, norm_array.data());
        return appr_alg->searchKnn((void*)norm_array.data(), k, efSearch, isIdAllowed);
    }

    std::priority_queue<std::pair<data_t, idtype >>
    searchKnnBruteForce(const void *query_data, size_t k, std::vector<idtype>& id_list) {
        if (!index_inited)
            throw std::runtime_error("Index not inited");
        if (normalize == false)
            return appr_alg->searchKnnBruteForce(query_data, k, id_list);

        std::vector<data_t> norm_array(dim);
        normalize_vector((data_t*)query_data, norm_array.data());
        return appr_alg->searchKnnBruteForce((void*)norm_array.data(), k, id_list);
    }

    std::priority_queue<std::pair<data_t, idtype >>
    searchRange(const void *query_data, float threshold, size_t efSearch, size_t max_efSearch, BaseFilterFunctor<idtype>* isIdAllowed = nullptr) {
        if (!index_inited)
            throw std::runtime_error("Index not inited");
        
        bool stop_flag = false;
        size_t l_search = efSearch; // starting size of the candidate list
        size_t max_l_search = max_efSearch;
        std::priority_queue<std::pair<data_t, idtype >> final_result;

        std::vector<data_t> norm_array(dim);
        if (normalize == true)
            normalize_vector((data_t*)query_data, norm_array.data());

        while (!stop_flag) {
            size_t i = 0;
            auto knn_result = normalize ? 
                            appr_alg->searchKnn((void*)norm_array.data(), l_search, l_search, isIdAllowed) : 
                            appr_alg->searchKnn(query_data, l_search, l_search, isIdAllowed);
            while (!knn_result.empty()) {
                std::pair<float, uint64_t> neighbor = knn_result.top(); // In default the top has the largest distance
                if (neighbor.first < threshold && i < l_search / 2) { // Find the first element that is in the range
                    if (l_search >= max_l_search) { //Reach the max l_search, directly return remaining elements in results
                        final_result = knn_result;
                        stop_flag = true;
                        break;
                    }   
                    else 
                        break;
                }
                else if (neighbor.first < threshold) {
                    final_result = knn_result;
                    stop_flag = true;
                    break;
                }
                //this element has distance larger than threshold, discard it
                knn_result.pop();
                i++;   
            }
            l_search *= 2;
            if (l_search > max_l_search)
                stop_flag = true;
        }
        return final_result;
    }

    std::priority_queue<std::pair<data_t, idtype >>
    searchRangeBruteForce(const void *query_data, float threshold, std::vector<idtype>& id_list) {
        if (!index_inited)
            throw std::runtime_error("Index not inited");
        if (normalize == false)
            return appr_alg->searchRangeBruteForce(query_data, threshold, id_list);

        std::vector<data_t> norm_array(dim);
        normalize_vector((data_t*)query_data, norm_array.data());
        return appr_alg->searchRangeBruteForce((void*)norm_array.data(), threshold, id_list);
    }

    std::vector<idtype> getIdsList() {
        std::vector<idtype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    bool getEmbedding(idtype label, data_t* container) {
        std::vector<data_t> data_vector = appr_alg->getDataByLabel(label);
        
        // Check if the retrieved data is empty
        if (data_vector.empty()) {
            return false;
        }
        
        assert(data_vector.size() == dim);
        size_t data_size = dim * sizeof(data_t);
        
        if (normalize) {
            std::cout << "normalized getting" << std::endl;
            double factor = appr_alg->getNormalizeFactorByLabel(label);
            for (size_t i = 0; i < data_vector.size(); ++i) {
                container[i] = data_vector[i] * factor;
            }
        }
        else
            std::memcpy(container, data_vector.data(), data_size);

        return true;
        
    }

    void markDeleted(idtype label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(idtype label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }

    size_t getActiveCount() const {
        return appr_alg->cur_element_count - appr_alg->deleted_elements.size();
    }

    size_t getEstimatedMemorySize() const {
        size_t total_size = sizeof(*this);  // Start with the size of the class itself
        if (l2space)
            total_size += sizeof(*l2space);
        if (appr_alg)
            total_size += appr_alg->estimateMemoryConsumption();

        return total_size;
    }
};

/*
template<typename data_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;
    int num_threads_default;

    idtype cur_l;
    hnswlib::BruteforceSearch<data_t>* alg;
    hnswlib::SpaceInterface<float>* space;


    BFIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        alg = NULL;
        index_inited = false;

        num_threads_default = std::thread::hardware_concurrency();
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    size_t getMaxElements() const {
        return alg->maxelements_;
    }


    size_t getCurrentCount() const {
        return alg->cur_element_count;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void init_new_index(const size_t maxElements) {
        if (alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        alg = new hnswlib::BruteforceSearch<data_t>(space, maxElements);
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(std::string, py::object ids_ = py::none()) {
        py::array_t < data_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t rows, features;
        uint32_t num_points, dimension;

        float * vector_array;
        vector_array = read_float_from_fbin(input_file, num_points, dimension);
        rows = static_cast<size_t>(num_points);
        features = static_cast<size_t>(dimension);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            for (size_t row = 0; row < rows; row++) {
                size_t id = ids.size() ? ids.at(row) : cur_l + row;
                if (!normalize) {
                    alg->addPoint((void *) items.data(row), (size_t) id);
                } else {
                    std::vector<float> normalized_vector(dim);
                    normalize_vector((float *)items.data(row), normalized_vector.data());
                    alg->addPoint((void *) normalized_vector.data(), (size_t) id);
                }
            }
            cur_l+=rows;
        }
    }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<data_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(idtype)>& filter = nullptr) {
        py::array_t < data_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        idtype *data_numpy_l;
        data_t *data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new idtype[rows * k];
            data_numpy_d = new data_t[rows * k];

            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<data_t, idtype >> result = alg->searchKnn(
                    (void*)items.data(row), k, p_idFilter);
                for (int i = k - 1; i >= 0; i--) {
                    auto& result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
            });
        }

        py::capsule free_when_done_l(data_numpy_l, [](void *f) {
            delete[] f;
        });
        py::capsule free_when_done_d(data_numpy_d, [](void *f) {
            delete[] f;
        });


        return py::make_tuple(
                py::array_t<idtype>(
                        { rows, k },  // shape
                        { k * sizeof(idtype),
                          sizeof(idtype)},  // C-style contiguous strides for each index
                        data_numpy_l,  // the data pointer
                        free_when_done_l),
                py::array_t<data_t>(
                        { rows, k },  // shape
                        { k * sizeof(data_t), sizeof(data_t) },  // C-style contiguous strides for each index
                        data_numpy_d,  // the data pointer
                        free_when_done_d));
    }
};
*/
}

#endif  /* ! HNSWLIB_HNSWLIB_INTERFACE_H_ */