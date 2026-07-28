#include "model.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace dav2 {
namespace {

constexpr char magic[8] = {'D', 'A', 'V', '2', 'M', 'O', 'D', '\0'};
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t endian_tag = 0x01020304;
constexpr std::uint32_t dtype_float32 = 1;
constexpr std::uint64_t alignment = 64;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian;
    std::uint32_t encoder;
    std::uint32_t tensor_count;
    std::uint64_t directory_offset;
    std::uint64_t directory_bytes;
    std::uint64_t data_offset;
    std::uint64_t file_bytes;
    std::uint64_t reserved;
};

struct TensorRecord {
    char name[112];
    std::uint32_t dtype;
    std::uint32_t rank;
    std::uint64_t dimensions[4];
    std::uint64_t data_offset;
    std::uint64_t data_bytes;
    std::uint64_t element_count;
    std::uint32_t crc32;
    std::uint32_t flags;
    std::uint64_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 64);
static_assert(sizeof(TensorRecord) == 192);

bool range_valid(std::uint64_t offset, std::uint64_t bytes, std::uint64_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

std::wstring utf8_to_wide(const std::string& text) {
#if defined(_WIN32)
    if (text.empty()) {
        throw std::invalid_argument("model path is empty");
    }
    const int length = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), nullptr, 0);
    if (length <= 0) {
        throw std::invalid_argument("model path is not valid UTF-8");
    }
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
            static_cast<int>(text.size()), result.data(), length) != length) {
        throw std::invalid_argument("failed to decode model path");
    }
    return result;
#else
    (void)text;
    return {};
#endif
}

}  // namespace

ModelFile::ModelFile(
    const std::string& path_utf8,
    dav2_encoder expected_encoder) {
#if !defined(_WIN32)
    (void)path_utf8;
    (void)expected_encoder;
    throw std::runtime_error("model mapping is not implemented on this platform");
#else
    try {
        const std::wstring path = utf8_to_wide(path_utf8);
        file_ = CreateFileW(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("failed to open model file");
        }
        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file_, &file_size) || file_size.QuadPart < 0) {
            throw std::runtime_error("failed to query model file size");
        }
        size_ = static_cast<std::uint64_t>(file_size.QuadPart);
        if (size_ < sizeof(FileHeader)) {
            throw std::runtime_error("model file is truncated");
        }
        mapping_ = CreateFileMappingW(
            file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            throw std::runtime_error("failed to map model file");
        }
        view_ = static_cast<const std::byte*>(
            MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!view_) {
            throw std::runtime_error("failed to view model file");
        }

        const auto& header =
            *reinterpret_cast<const FileHeader*>(view_);
        if (std::memcmp(header.magic, magic, sizeof(magic)) != 0 ||
            header.version != format_version ||
            header.endian != endian_tag) {
            throw std::runtime_error("invalid DAV2 model header");
        }
        if (header.encoder != static_cast<std::uint32_t>(expected_encoder)) {
            throw std::runtime_error("model encoder does not match create options");
        }
        if (header.tensor_count == 0 || header.tensor_count > 2048 ||
            header.file_bytes != size_) {
            throw std::runtime_error("invalid DAV2 model bounds");
        }
        const std::uint64_t expected_directory =
            static_cast<std::uint64_t>(header.tensor_count) *
            sizeof(TensorRecord);
        if (header.directory_bytes != expected_directory ||
            !range_valid(
                header.directory_offset, header.directory_bytes, size_) ||
            header.data_offset % alignment != 0 ||
            header.data_offset <
                header.directory_offset + header.directory_bytes ||
            header.data_offset > size_) {
            throw std::runtime_error("invalid DAV2 tensor directory");
        }

        tensors_.reserve(header.tensor_count);
        const auto* records = reinterpret_cast<const TensorRecord*>(
            view_ + header.directory_offset);
        for (std::uint32_t index = 0; index < header.tensor_count; ++index) {
            const TensorRecord& record = records[index];
            const void* terminator =
                std::memchr(record.name, '\0', sizeof(record.name));
            if (!terminator || record.name[0] == '\0' ||
                record.dtype != dtype_float32 ||
                record.rank == 0 || record.rank > 4 ||
                record.flags != 0 || record.reserved != 0 ||
                record.data_offset < header.data_offset ||
                record.data_offset % alignment != 0 ||
                !range_valid(record.data_offset, record.data_bytes, size_)) {
                throw std::runtime_error("invalid DAV2 tensor record");
            }
            std::uint64_t elements = 1;
            for (std::uint32_t dimension = 0; dimension < 4; ++dimension) {
                const std::uint64_t value = record.dimensions[dimension];
                if (dimension < record.rank) {
                    if (value == 0 ||
                        elements >
                            std::numeric_limits<std::uint64_t>::max() / value) {
                        throw std::runtime_error("invalid DAV2 tensor dimensions");
                    }
                    elements *= value;
                } else if (value != 0) {
                    throw std::runtime_error("invalid DAV2 unused dimension");
                }
            }
            if (elements != record.element_count ||
                elements >
                    std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
                elements * sizeof(float) != record.data_bytes) {
                throw std::runtime_error("invalid DAV2 tensor byte count");
            }
            const auto* name_end = static_cast<const char*>(terminator);
            std::string_view name(
                record.name,
                static_cast<std::size_t>(name_end - record.name));
            TensorView tensor{
                reinterpret_cast<const float*>(
                    view_ + record.data_offset),
                {
                    record.dimensions[0],
                    record.dimensions[1],
                    record.dimensions[2],
                    record.dimensions[3],
                },
                record.rank,
                record.element_count,
                record.crc32,
            };
            if (!tensors_.emplace(name, tensor).second) {
                throw std::runtime_error("duplicate DAV2 tensor name");
            }
        }
    } catch (...) {
        close();
        throw;
    }
#endif
}

ModelFile::~ModelFile() {
    close();
}

void ModelFile::close() noexcept {
#if defined(_WIN32)
    tensors_.clear();
    if (view_) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (mapping_) {
        CloseHandle(mapping_);
        mapping_ = nullptr;
    }
    if (file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
    }
    size_ = 0;
#endif
}

const TensorView& ModelFile::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "model is missing tensor: " + std::string(name));
    }
    return found->second;
}

bool ModelFile::contains(std::string_view name) const {
    return tensors_.find(name) != tensors_.end();
}

}  // namespace dav2
