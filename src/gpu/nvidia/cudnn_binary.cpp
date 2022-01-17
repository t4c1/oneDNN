/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/nvidia/cudnn_binary.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"

#include <optional>

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_binary_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->src_md(0)).has_zero_dim())
        return status::success;

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto *mem_src_0 = static_cast<sycl::sycl_memory_storage_base_t *>(&CTX_IN_STORAGE(DNNL_ARG_SRC_0));
        auto *mem_src_1 = static_cast<sycl::sycl_memory_storage_base_t *>(&CTX_IN_STORAGE(DNNL_ARG_SRC_1));
        auto *mem_dst = static_cast<sycl::sycl_memory_storage_base_t *>(&CTX_OUT_STORAGE(DNNL_ARG_DST));
        std::optional<decltype(CTX_IN_ACCESSOR(DNNL_ARG_SRC_0))> src_0_acc;
        std::optional<decltype(CTX_IN_ACCESSOR(DNNL_ARG_SRC_1))> src_1_acc;
        std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_DST))> dst_acc;
        if(mem_src_0->memory_kind() == sycl::memory_kind::buffer){
            src_0_acc.emplace(utils::downcast<sycl::sycl_buffer_memory_storage_t *>(mem_src_0)->buffer(),cgh);
        }
        if(mem_src_1->memory_kind() == sycl::memory_kind::buffer){
            src_1_acc.emplace(utils::downcast<sycl::sycl_buffer_memory_storage_t *>(mem_src_1)->buffer(), cgh);
        }
        if(mem_dst->memory_kind() == sycl::memory_kind::buffer){
            dst_acc.emplace(utils::downcast<sycl::sycl_buffer_memory_storage_t *>(mem_dst)->buffer(), cgh);
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                    cuda_stream->engine());
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();

            void* a;
            switch (mem_src_0->memory_kind()) {
            case sycl::memory_kind::buffer:
                a = sc.memory<void *>(ih, src_0_acc.value());
                break;
            case sycl::memory_kind::usm:
                a = utils::downcast<const sycl::sycl_usm_memory_storage_t *>(
                                    mem_src_0)->usm_ptr();
                break;
            default:
                assert(!"not expected");
            }
            void* b;
            switch (mem_src_1->memory_kind()) {
            case sycl::memory_kind::buffer:
                b = sc.memory<void *>(ih, src_1_acc.value());
                break;
            case sycl::memory_kind::usm:
                b = utils::downcast<const sycl::sycl_usm_memory_storage_t *>(
                                    mem_src_1)->usm_ptr();
                break;
            default:
                assert(!"not expected");
            }
            void* c;
            switch (mem_dst->memory_kind()) {
            case sycl::memory_kind::buffer:
                c = sc.memory<void *>(ih, dst_acc.value());
                break;
            case sycl::memory_kind::usm:
                c = utils::downcast<const sycl::sycl_usm_memory_storage_t *>(
                                    mem_dst)->usm_ptr();
                break;
            default:
                assert(!"not expected");
            }

            pd()->binary_impl_->execute(handle, a, b, c);
        });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
