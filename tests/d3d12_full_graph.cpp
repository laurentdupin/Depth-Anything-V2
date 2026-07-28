#include "depth_anything_v2.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}

void check(dav2_status status, const char* operation) {
    if (status != DAV2_STATUS_OK) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            dav2_status_string(status) + ": " +
            dav2_last_error());
    }
}

ComPtr<ID3D12Device> matching_device(std::uint64_t luid) {
    ComPtr<IDXGIFactory6> factory;
    check(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT enumerated =
            factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        check(enumerated, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0;
        static_assert(sizeof(candidate) == sizeof(description.AdapterLuid));
        std::memcpy(
            &candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check(
            D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    throw std::runtime_error(
        "no D3D12 adapter matches the Vulkan device LUID");
}

ComPtr<ID3D12CommandQueue> make_queue(ID3D12Device* device) {
    const D3D12_COMMAND_QUEUE_DESC description{
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        0,
        D3D12_COMMAND_QUEUE_FLAG_NONE,
        0,
    };
    ComPtr<ID3D12CommandQueue> result;
    check(
        device->CreateCommandQueue(
            &description, IID_PPV_ARGS(&result)),
        "CreateCommandQueue");
    return result;
}

void cpu_wait(ID3D12Fence* fence, std::uint64_t value) {
    if (fence->GetCompletedValue() >= value) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }
    const HRESULT scheduled =
        fence->SetEventOnCompletion(value, event);
    if (FAILED(scheduled)) {
        CloseHandle(event);
        check(scheduled, "SetEventOnCompletion");
    }
    if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(event);
        throw std::runtime_error("fence wait failed");
    }
    CloseHandle(event);
}

struct SharedInput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Fence> ready;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t fence_value = 1;

    SharedInput() = default;
    SharedInput(const SharedInput&) = delete;
    SharedInput& operator=(const SharedInput&) = delete;
    SharedInput(SharedInput&& other) noexcept
        : resource(std::move(other.resource)),
          upload(std::move(other.upload)),
          ready(std::move(other.ready)),
          allocator(std::move(other.allocator)),
          commands(std::move(other.commands)),
          resource_handle(
              std::exchange(other.resource_handle, nullptr)),
          fence_handle(std::exchange(other.fence_handle, nullptr)),
          fence_value(other.fence_value) {}

    ~SharedInput() {
        if (resource_handle != nullptr) CloseHandle(resource_handle);
        if (fence_handle != nullptr) CloseHandle(fence_handle);
    }
};

SharedInput upload_capture(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    const std::vector<std::uint8_t>& pixels,
    bool signal_ready) {
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        pixels.size(),
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    SharedInput result;
    check(
        device->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_SHARED,
            &description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&result.resource)),
        "CreateCommittedResource(capture)");
    check(
        device->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&result.upload)),
        "CreateCommittedResource(upload)");
    void* mapped = nullptr;
    check(
        result.upload->Map(0, nullptr, &mapped),
        "Map(upload)");
    std::memcpy(mapped, pixels.data(), pixels.size());
    result.upload->Unmap(0, nullptr);

    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&result.allocator)),
        "CreateCommandAllocator(upload)");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            result.allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&result.commands)),
        "CreateCommandList(upload)");
    result.commands->CopyBufferRegion(
        result.resource.Get(),
        0,
        result.upload.Get(),
        0,
        pixels.size());
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                result.resource.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            },
        },
    };
    result.commands->ResourceBarrier(1, &barrier);
    check(result.commands->Close(), "Close(upload)");
    ID3D12CommandList* submitted[] = {result.commands.Get()};
    queue->ExecuteCommandLists(1, submitted);

    check(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&result.ready)),
        "CreateFence(capture)");
    check(
        device->CreateSharedHandle(
            result.resource.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.resource_handle),
        "CreateSharedHandle(capture)");
    check(
        device->CreateSharedHandle(
            result.ready.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &result.fence_handle),
        "CreateSharedHandle(capture fence)");
    if (signal_ready) {
        check(
            queue->Signal(result.ready.Get(), result.fence_value),
            "Signal(capture)");
    }
    return result;
}

std::vector<std::uint8_t> make_pixels(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t frame) {
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(width) * height * 4);
    for (std::size_t index = 0; index < result.size() / 4; ++index) {
        result[index * 4] =
            static_cast<std::uint8_t>((index * 13 + frame * 31) % 251);
        result[index * 4 + 1] =
            static_cast<std::uint8_t>((index * 29 + frame * 7) % 253);
        result[index * 4 + 2] =
            static_cast<std::uint8_t>((index * 43 + frame * 17) % 255);
        result[index * 4 + 3] = 255;
    }
    return result;
}

std::vector<float> read_depth(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    const dav2_d3d12_output_descriptor& descriptor) {
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12Fence> ready;
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.shared_resource_handle)),
            IID_PPV_ARGS(&output)),
        "OpenSharedHandle(depth)");
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.ready_fence_handle)),
            IID_PPV_ARGS(&ready)),
        "OpenSharedHandle(depth fence)");
    check(
        queue->Wait(ready.Get(), descriptor.ready_fence_value),
        "Wait(depth fence)");

    const D3D12_HEAP_PROPERTIES readback_heap{
        D3D12_HEAP_TYPE_READBACK,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        1,
        1,
    };
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        descriptor.byte_size,
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE,
    };
    ComPtr<ID3D12Resource> readback;
    check(
        device->CreateCommittedResource(
            &readback_heap,
            D3D12_HEAP_FLAG_NONE,
            &description,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&readback)),
        "CreateCommittedResource(readback)");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    check(
        device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator(readback)");
    check(
        device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&commands)),
        "CreateCommandList(readback)");
    const D3D12_RESOURCE_BARRIER barrier{
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        D3D12_RESOURCE_BARRIER_FLAG_NONE,
        {
            {
                output.Get(),
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            },
        },
    };
    commands->ResourceBarrier(1, &barrier);
    commands->CopyBufferRegion(
        readback.Get(), 0, output.Get(), 0, descriptor.byte_size);
    check(commands->Close(), "Close(readback)");
    ID3D12CommandList* submitted[] = {commands.Get()};
    queue->ExecuteCommandLists(1, submitted);
    ComPtr<ID3D12Fence> complete;
    check(
        device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&complete)),
        "CreateFence(readback)");
    check(queue->Signal(complete.Get(), 1), "Signal(readback)");
    cpu_wait(complete.Get(), 1);

    std::vector<float> result(
        static_cast<std::size_t>(descriptor.width) *
        descriptor.height);
    void* mapped = nullptr;
    check(readback->Map(0, nullptr, &mapped), "Map(readback)");
    std::memcpy(
        result.data(), mapped, result.size() * sizeof(float));
    readback->Unmap(0, nullptr);
    return result;
}

void wait_depth_ready(
    ID3D12Device* device,
    const dav2_d3d12_output_descriptor& descriptor) {
    ComPtr<ID3D12Fence> ready;
    check(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(
                static_cast<std::uintptr_t>(
                    descriptor.ready_fence_handle)),
            IID_PPV_ARGS(&ready)),
        "OpenSharedHandle(depth fence)");
    cpu_wait(ready.Get(), descriptor.ready_fence_value);
}

void validate_depth(const std::vector<float>& depth) {
    const auto bounds =
        std::minmax_element(depth.begin(), depth.end());
    if (bounds.first == depth.end() ||
        !std::isfinite(*bounds.first) ||
        !std::isfinite(*bounds.second) ||
        *bounds.second <= *bounds.first) {
        throw std::runtime_error(
            "GPU depth output is not a finite non-constant map");
    }
}

void compare_reference(
    dav2_context* context,
    const std::vector<std::uint8_t>& bgra,
    std::uint32_t width,
    std::uint32_t height,
    const std::vector<float>& gpu) {
    std::vector<std::uint8_t> bgr(
        static_cast<std::size_t>(width) * height * 3);
    for (std::size_t index = 0; index < bgr.size() / 3; ++index) {
        bgr[index * 3] = bgra[index * 4];
        bgr[index * 3 + 1] = bgra[index * 4 + 1];
        bgr[index * 3 + 2] = bgra[index * 4 + 2];
    }
    std::vector<float> reference(
        static_cast<std::size_t>(width) * height);
    check(
        dav2_infer_bgr8(
            context,
            bgr.data(),
            static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(height),
            static_cast<std::ptrdiff_t>(width * 3),
            140,
            reference.data(),
            reference.size()),
        "dav2_infer_bgr8(reference)");
    float maximum_difference = 0.0f;
    float reference_range =
        *std::max_element(reference.begin(), reference.end()) -
        *std::min_element(reference.begin(), reference.end());
    for (std::size_t index = 0; index < gpu.size(); ++index) {
        maximum_difference = std::max(
            maximum_difference,
            std::abs(gpu[index] - reference[index]));
    }
    const float relative =
        maximum_difference /
        std::max(reference_range, std::numeric_limits<float>::epsilon());
    std::cout << "CPU correlation max/range=" << relative << '\n';
    if (relative >= 0.01f) {
        throw std::runtime_error(
            "GPU-resident path exceeds the 1% CPU-output deviation gate");
    }
}

std::filesystem::path model_path() {
    if (const char* configured = std::getenv("DAV2_VITS_MODEL")) {
        return configured;
    }
#if defined(DAV2_DEFAULT_VITS_MODEL)
    return DAV2_DEFAULT_VITS_MODEL;
#else
    return {};
#endif
}

}  // namespace

int main() try {
    const std::filesystem::path model = model_path();
    if (model.empty() || !std::filesystem::exists(model)) {
        std::cout << "DAV2 vits model unavailable\n";
        return 77;
    }
    dav2_gpu_capabilities probed{};
    probed.struct_size = sizeof(probed);
    check(
        dav2_probe_gpu_capabilities(0, &probed),
        "dav2_probe_gpu_capabilities");
    dav2_create_options options{
        sizeof(options), DAV2_ABI_VERSION, DAV2_ENCODER_VITS, 0, 0};
    dav2_context* context = nullptr;
    check(
        dav2_create(model.string().c_str(), &options, &context),
        "dav2_create");

    dav2_gpu_capabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    check(
        dav2_get_gpu_capabilities(context, &capabilities),
        "dav2_get_gpu_capabilities");
    if (probed.flags != capabilities.flags ||
        probed.adapter_luid != capabilities.adapter_luid) {
        throw std::runtime_error(
            "pre-model and loaded-model GPU capabilities disagree");
    }
    constexpr std::uint64_t required_capabilities =
        DAV2_GPU_CAP_D3D12_SHARED_INPUT |
        DAV2_GPU_CAP_D3D12_FENCE_WAIT |
        DAV2_GPU_CAP_D3D12_SHARED_OUTPUT |
        DAV2_GPU_CAP_D3D12_FENCE_SIGNAL |
        DAV2_GPU_CAP_ASYNC_SUBMIT |
        DAV2_GPU_CAP_CANCELLATION |
        DAV2_GPU_CAP_NO_HOST_PIXEL_STAGING |
        DAV2_GPU_CAP_NO_HOST_DEPTH_STAGING;
    if ((capabilities.flags & required_capabilities) !=
            required_capabilities ||
        capabilities.adapter_luid == 0) {
        std::cout << "GPU capability flags=0x" << std::hex
                  << capabilities.flags << std::dec
                  << " adapter_luid=" << capabilities.adapter_luid
                  << '\n';
        dav2_destroy(context);
        std::cout << "complete D3D12/Vulkan interop unavailable\n";
        return 77;
    }
    ComPtr<ID3D12Device> device =
        matching_device(capabilities.adapter_luid);
    ComPtr<ID3D12CommandQueue> queue =
        make_queue(device.Get());

    constexpr std::uint32_t width = 73;
    constexpr std::uint32_t height = 51;
    for (std::uint32_t frame = 0; frame < 3; ++frame) {
        const std::vector<std::uint8_t> pixels =
            make_pixels(width, height, frame);
        SharedInput input =
            upload_capture(device.Get(), queue.Get(), pixels, true);
        dav2_transfer_counters before{};
        before.struct_size = sizeof(before);
        check(
            dav2_get_transfer_counters(context, &before),
            "dav2_get_transfer_counters(before)");
        dav2_d3d12_submit_request request{};
        request.struct_size = sizeof(request);
        request.abi_version = DAV2_ABI_VERSION;
        request.shared_resource_handle =
            reinterpret_cast<std::uintptr_t>(input.resource_handle);
        request.resource_byte_size = pixels.size();
        request.width = width;
        request.height = height;
        request.row_stride_bytes = width * 4;
        request.pixel_format = DAV2_GPU_PIXEL_BGRA8;
        request.input_size = 140;
        request.wait_fence_handle =
            reinterpret_cast<std::uintptr_t>(input.fence_handle);
        request.wait_fence_value = input.fence_value;
        request.source_frame_id = 9000 + frame;
        request.timestamp_ns = 123456789 + frame;
        dav2_gpu_job* job = nullptr;
        check(
            dav2_submit_d3d12(context, &request, &job),
            "dav2_submit_d3d12");

        // Submit imported duplicate handles, not caller-owned handles.
        CloseHandle(input.resource_handle);
        input.resource_handle = nullptr;
        CloseHandle(input.fence_handle);
        input.fence_handle = nullptr;
        input.resource.Reset();
        input.ready.Reset();

        dav2_d3d12_output_descriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        dav2_gpu_output_lease* lease = nullptr;
        check(
            dav2_gpu_output_acquire(
                job, 0, &descriptor, &lease),
            "dav2_gpu_output_acquire");
        if (descriptor.source_frame_id != request.source_frame_id ||
            descriptor.timestamp_ns != request.timestamp_ns ||
            descriptor.width != width ||
            descriptor.height != height ||
            descriptor.pixel_format !=
                DAV2_GPU_PIXEL_DEPTH_FLOAT32) {
            throw std::runtime_error(
                "GPU output metadata correlation failed");
        }

        wait_depth_ready(device.Get(), descriptor);
        dav2_gpu_job_status status{};
        for (std::uint32_t attempt = 0; attempt < 1000; ++attempt) {
            status = {};
            status.struct_size = sizeof(status);
            check(
                dav2_gpu_job_poll(job, &status),
                "dav2_gpu_job_poll");
            if (status.state != DAV2_GPU_JOB_RUNNING) break;
            Sleep(1);
        }
        if (status.state != DAV2_GPU_JOB_COMPLETE ||
            status.source_frame_id != request.source_frame_id) {
            throw std::runtime_error(
                "GPU completion state correlation failed");
        }
        // The lease, not the job handle, owns descriptor handle lifetime.
        dav2_gpu_job_release(job);
        const std::vector<float> depth =
            read_depth(device.Get(), queue.Get(), descriptor);
        validate_depth(depth);
        dav2_gpu_output_release(lease);

        dav2_transfer_counters after{};
        after.struct_size = sizeof(after);
        check(
            dav2_get_transfer_counters(context, &after),
            "dav2_get_transfer_counters(after)");
        if (after.tensor_upload_bytes !=
                before.tensor_upload_bytes ||
            after.tensor_download_bytes !=
                before.tensor_download_bytes) {
            throw std::runtime_error(
                "DAV2 performed CPU tensor staging on the GPU path");
        }
        if (frame == 0) {
            compare_reference(
                context, pixels, width, height, depth);
        }
        std::cout << "frame " << request.source_frame_id
                  << " GPU-resident graph passed\n";
    }

    // Hold the imported wait unsignaled so cancellation is deterministic.
    const std::vector<std::uint8_t> pixels =
        make_pixels(width, height, 99);
    SharedInput blocked =
        upload_capture(device.Get(), queue.Get(), pixels, false);
    dav2_d3d12_submit_request request{};
    request.struct_size = sizeof(request);
    request.abi_version = DAV2_ABI_VERSION;
    request.shared_resource_handle =
        reinterpret_cast<std::uintptr_t>(blocked.resource_handle);
    request.resource_byte_size = pixels.size();
    request.width = width;
    request.height = height;
    request.row_stride_bytes = width * 4;
    request.pixel_format = DAV2_GPU_PIXEL_BGRA8;
    request.input_size = 140;
    request.wait_fence_handle =
        reinterpret_cast<std::uintptr_t>(blocked.fence_handle);
    request.wait_fence_value = blocked.fence_value;
    request.source_frame_id = 9999;
    dav2_gpu_job* cancelled = nullptr;
    check(
        dav2_submit_d3d12(context, &request, &cancelled),
        "dav2_submit_d3d12(cancel)");
    check(
        dav2_gpu_job_cancel(cancelled),
        "dav2_gpu_job_cancel");
    dav2_gpu_job_status cancelled_status{};
    cancelled_status.struct_size = sizeof(cancelled_status);
    check(
        dav2_gpu_job_poll(cancelled, &cancelled_status),
        "dav2_gpu_job_poll(cancel)");
    if (cancelled_status.state != DAV2_GPU_JOB_CANCELLED ||
        cancelled_status.source_frame_id != request.source_frame_id) {
        throw std::runtime_error(
            "cancelled GPU job state correlation failed");
    }
    dav2_d3d12_output_descriptor cancelled_output{};
    cancelled_output.struct_size = sizeof(cancelled_output);
    dav2_gpu_output_lease* cancelled_lease = nullptr;
    if (dav2_gpu_output_acquire(
            cancelled,
            0,
            &cancelled_output,
            &cancelled_lease) == DAV2_STATUS_OK) {
        throw std::runtime_error(
            "cancelled GPU job exposed an output");
    }

    // Destroying the public context is safe while the job retains the model.
    dav2_destroy(context);
    context = nullptr;
    check(
        queue->Signal(blocked.ready.Get(), blocked.fence_value),
        "Signal(cancelled capture)");
    dav2_gpu_job_release(cancelled);
    std::cout
        << "cancellation, retained fence lifetime, and shutdown passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
