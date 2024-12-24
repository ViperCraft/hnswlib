#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <stdlib.h>
#include <assert.h>
#include <memory>
#include <stdint.h>
#include "hnswalg.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>

namespace hnswlib {

struct DataMapping
{
    DataMapping( void const *d, size_t s )
        : data(d)
        , size(s)
        {}

    DataMapping() = default;

    void const *data = nullptr;
    size_t size = 0;
};

class MMapDataMapper {
public:
  using Mapping = DataMapping;
public:
  static Mapping map(const char* filename, bool need_mlock) {
    int fd = open(filename, O_RDONLY, (int)0400);
    if (fd == -1) {
      return Mapping();
    }

    struct stat fd_stat;
    if (fstat(fd, &fd_stat)) {
        close(fd);
        return Mapping();
    }

    void *mmaped = mmap(0, fd_stat.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mmaped == MAP_FAILED) {
      return Mapping();
    }
#if defined(MADV_DONTDUMP)
    // Exclude from a core dump those pages
    madvise(mmaped, fd_stat.st_size, MADV_DONTDUMP);
#endif

    if (need_mlock) {
      ::mlock(mmaped, fd_stat.st_size);
    }

    return Mapping(mmaped, (size_t)fd_stat.st_size);
  }

  static void unmap(const Mapping & mapping) {
    if (mapping.data) {
      ::munmap(const_cast<void*>(mapping.data), mapping.size);
    }
  }

  template<typename T>
  static void readPOD( char const *&top, T &out )
  {
    out = *(T const*)top;
    top += sizeof(T);
  }

  static void mlock(const Mapping & mapping) {
    if (mapping.data) {
      ::mlock(const_cast<void*>(mapping.data), mapping.size);
    }
  }
};

constexpr size_t align_size( size_t sz, size_t block_siz )
{
    return (sz + block_siz - 1) & ~(block_siz - 1);
}

inline void* aligned_malloc( size_t sz, size_t alignment = 16 )
{
    void *out_bytes;
    if( posix_memalign(&out_bytes, alignment, sz) )
        throw std::bad_alloc();

    return out_bytes;
}

class VisitedBitmap
{
public:
    VisitedBitmap( size_t max_capacity )
        : capacity_( align_size(max_capacity, 512) / 8 )
        , bitmap_( static_cast<uint64_t*>(aligned_malloc(capacity_, 64)) )
        {}

    void clear()
    {
        memset(bitmap_, 0, capacity_);
    }
    
    bool mark( uint32_t pos )
    {
        if( is_marked(pos) )
        {
            return false;
        }
        uint32_t addr = pos / 64;
        bitmap_[addr] |= 1UL << (pos % 64);
        return true;
    }

    void set( uint32_t pos )
    {
        uint32_t addr = pos / 64;
        bitmap_[addr] |= 1UL << (pos % 64);
    }

    bool is_marked( uint32_t pos ) const
    {
        uint64_t const w = bitmap_[ pos / 64 ];
        return ((w >> (pos % 64)) & 1UL) == 1UL;
    }
private:
    size_t              capacity_;
    uint64_t            *bitmap_;
};

// reads filecontent directly from disk-cache w/o moving to the anon memory and allocation
// reading and search only, so temporary structures for quick expand also omitted 
// to get maximum speed, can be used with HierarchicalNSW class
// NOTE: this code is completely based on code from original HierarchicalNSW class!
template<typename dist_t, typename MMapImpl = MMapDataMapper>
class HierarchicalNSWFastReader
{
    using pq_result_t = std::priority_queue<std::pair<dist_t, labeltype >>;
    using pq_top_candidates_t = std::priority_queue<std::pair<dist_t, tableint>, 
        std::vector<std::pair<dist_t, tableint>>, 
        typename HierarchicalNSW<dist_t>::CompareByFirst>;
public:

    HierarchicalNSWFastReader(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        size_t max_elements = 0) {
        loadIndex(location, s, max_elements);
    }


    ~HierarchicalNSWFastReader() {
        free(linkLists_);
        linkLists_ = nullptr;

        MMapImpl::unmap(mapping_);
    }


    pq_result_t
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        pq_result_t result;
        if (max_elements_ == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    //if (cand < 0 || cand > max_elements_)
                    //    throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        pq_top_candidates_t top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }
        
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

private:

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    pq_top_candidates_t
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        //VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        //vl_type *visited_array = vl->mass;
        //vl_type visited_array_tag = vl->curV;
        visited_bitmap_->clear();

        pq_top_candidates_t top_candidates;
        pq_top_candidates_t candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char const* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        //visited_array[ep_id] = visited_array_tag;
        visited_bitmap_->set(ep_id);

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);

#ifdef USE_SSE
            //_mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            //_mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                //_mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                //if (!(visited_array[candidate_id] == visited_array_tag)) {
                //    visited_array[candidate_id] = visited_array_tag;
                if( visited_bitmap_->mark(candidate_id) ) {

                    char const *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        //visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {

        mapping_ = MMapImpl::map(location.c_str(), false);

        size_t data_size_{0};
        size_t M_{0};
        size_t maxM0_{0};
        size_t maxM_{0};
        size_t ef_construction_{0};
        size_t cur_element_count{0};  // current number of elements
        double mult_{0.0}, revSize_{0.0};

        size_t total_filesize = mapping_.size;

        char const *input = static_cast<char const*>(mapping_.data);


        MMapImpl::readPOD(input, offsetLevel0_);
        MMapImpl::readPOD(input, max_elements_);
        MMapImpl::readPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        MMapImpl::readPOD(input, size_data_per_element_);
        MMapImpl::readPOD(input, label_offset_);
        MMapImpl::readPOD(input, offsetData_);
        MMapImpl::readPOD(input, maxlevel_);
        MMapImpl::readPOD(input, enterpoint_node_);

        MMapImpl::readPOD(input, maxM_);
        MMapImpl::readPOD(input, maxM0_);
        MMapImpl::readPOD(input, M_);
        MMapImpl::readPOD(input, mult_);
        MMapImpl::readPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        data_level0_memory_ = input;

        input += max_elements * size_data_per_element_;

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_t size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        //visited_list_pool_.reset(new VisitedListPool(1, max_elements));
        visited_bitmap_.reset( new VisitedBitmap(max_elements) );

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize;
            MMapImpl::readPOD(input, linkListSize);
            if (linkListSize == 0) {
                linkLists_[i] = nullptr;
            } else {
                linkLists_[i] = const_cast<char*>(input);
                input += linkListSize;
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
            }
        }

        return;
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }

    inline labeltype const* getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char const* getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    inline linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    inline linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    inline linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    inline linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

private:
    static const unsigned char DELETE_MARK = 0x01;

    using mutex_t = typename NoMTSupport::mutex_type;
    using VisitedListPool = VisitedListPool<mutex_t>;
    size_t max_elements_{0};
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    size_t num_deleted_{0};  // number of deleted elements
    size_t ef_{ 0 };

    tableint enterpoint_node_{0};
    int maxlevel_{0};

    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char const *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    //std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    std::unique_ptr<VisitedBitmap> visited_bitmap_;

    DataMapping mapping_;
};

}
