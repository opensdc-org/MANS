#ifndef MANS_CONTAINER_H
#define MANS_CONTAINER_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mans {
namespace container {

constexpr char kContainerMagic[8] = {'M', 'A', 'N', 'S', 'C', 'H', 'N', 'K'};
constexpr char kChunkMagic[4] = {'M', 'C', 'H', 'K'};
constexpr std::uint32_t kContainerVersion = 1;
constexpr std::uint16_t kChunkVersion = 1;
constexpr std::uint64_t kFlagRawPayload = 1ull << 0;

#pragma pack(push, 1)
struct ContainerHeader {
    char magic[8];
    std::uint32_t version;
    std::uint8_t dtype;
    std::uint8_t reserved0;
    std::uint16_t header_bytes;
    std::uint64_t chunk_count;
    std::uint64_t index_offset;
    std::uint64_t data_offset;
    std::uint64_t chunk_header_bytes;
    std::uint64_t flags;
};

struct IndexEntry {
    std::uint64_t offset;
    std::uint64_t comp_len;
    std::uint64_t raw_len;
};

struct ChunkHeader {
    char magic[4];
    std::uint16_t version;
    std::uint16_t header_bytes;
    std::uint64_t comp_len;
    std::uint64_t raw_len;
    std::uint64_t chunk_index;
};
#pragma pack(pop)

inline bool magic_matches(const char* value, const char* magic, std::size_t size) {
    return std::memcmp(value, magic, size) == 0;
}

} // namespace container
} // namespace mans

#endif
