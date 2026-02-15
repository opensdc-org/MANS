#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace mans {
namespace cpu {

class BufferCache {
public:
    static BufferCache& instance() {
        static BufferCache* inst = []() {
            auto* ptr = new BufferCache();
            std::atexit(&BufferCache::atexit_release);
            return ptr;
        }();
        return *inst;
    }

    void* get(const char* tag, std::size_t bytes) {
        return get_aligned(tag, alignof(std::max_align_t), bytes);
    }

    template <typename T>
    T* get_t(const char* tag, std::size_t count) {
        return static_cast<T*>(get(tag, count * sizeof(T)));
    }

    void* get_aligned(const char* tag, std::size_t alignment, std::size_t bytes) {
        if (!tag || bytes == 0) {
            return nullptr;
        }
        if (alignment < alignof(std::max_align_t)) {
            alignment = alignof(std::max_align_t);
        }

        auto& entry = buffers_[tag];
        if (entry.ptr && entry.capacity >= bytes && entry.alignment >= alignment) {
            return entry.ptr;
        }

        void* new_ptr = nullptr;
        if (alignment <= alignof(std::max_align_t)) {
            new_ptr = std::realloc(entry.ptr, bytes);
            if (!new_ptr) {
                new_ptr = std::malloc(bytes);
                if (!new_ptr) {
                    return nullptr;
                }
                std::free(entry.ptr);
            }
        } else {
            if (posix_memalign(&new_ptr, alignment, bytes) != 0) {
                return nullptr;
            }
            std::free(entry.ptr);
        }

        entry.ptr = new_ptr;
        entry.capacity = bytes;
        entry.alignment = alignment;
        return entry.ptr;
    }

    template <typename T>
    T* get_aligned_t(const char* tag, std::size_t alignment, std::size_t count) {
        return static_cast<T*>(get_aligned(tag, alignment, count * sizeof(T)));
    }

    void release_all() {
        for (auto& kv : buffers_) {
            std::free(kv.second.ptr);
            kv.second.ptr = nullptr;
            kv.second.capacity = 0;
            kv.second.alignment = 0;
        }
        buffers_.clear();
    }

private:
    struct Entry {
        void* ptr = nullptr;
        std::size_t capacity = 0;
        std::size_t alignment = 0;
    };

    BufferCache() = default;

    static void atexit_release() {
        BufferCache::instance().release_all();
    }

    std::unordered_map<std::string, Entry> buffers_;
};

} // namespace cpu
} // namespace mans
