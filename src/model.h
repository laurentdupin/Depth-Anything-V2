#pragma once

#include "depth_anything_v2.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace dav2 {

struct TensorView {
    const float* data = nullptr;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
    std::uint32_t crc32 = 0;
};

class ModelFile {
public:
    ModelFile(const std::string& path_utf8, dav2_encoder expected_encoder);
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ~ModelFile();

    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }

private:
    void close() noexcept;

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#endif
    const std::byte* view_ = nullptr;
    std::uint64_t size_ = 0;
    std::unordered_map<std::string_view, TensorView> tensors_;
};

}  // namespace dav2
