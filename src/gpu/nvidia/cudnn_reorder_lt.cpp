/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020-2022 Codeplay Software Limited
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

#include "gpu/nvidia/cudnn_reorder_lt.hpp"
#include "gpu/nvidia/stream.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream_utils.hpp"

#include "xpu/sycl/memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_reorder_lt_t::execute_internal_reorder(const exec_ctx_t &ctx,
        const memory_arg_t &src,
	const memory_arg_t &dst,
        const memory_arg_t *src_scales,
	const memory_arg_t *dst_scales) const{

    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = src;
    r_args[DNNL_ARG_DST] = dst;
    if (src_scales) r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = *src_scales;
    if (dst_scales) r_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = *dst_scales;

    exec_ctx_t r_ctx(ctx, std::move(r_args));
    nested_scratchpad_t ns(
            ctx, memory_tracking::names::key_nested, generic_reorder_);
    r_ctx.set_scratchpad_grantor(ns.grantor());

    return generic_reorder_->execute(r_ctx);
}
status_t cudnn_reorder_lt_t::execute(const exec_ctx_t &ctx) const {
    memory_desc_wrapper wrap(pd()->src_md());
    if (wrap.size() == 0) { return status::success; }

    nvidia::stream_t *cuda_stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    return cuda_stream->interop_task([&](::sycl::handler &cgh) {
        auto arg_src = ctx.args().at(DNNL_ARG_SRC); // CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = ctx.args().at(DNNL_ARG_DST);
        auto arg_src_scale
                = ctx.args().find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
        auto arg_dst_scale
                = ctx.args().find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
	const memory_arg_t *src_scales = nullptr;
        if (arg_src_scale != ctx.args().end()) {
            src_scales = &arg_src_scale->second;
        }
	const memory_arg_t *dst_scales = nullptr;
        if (arg_dst_scale != ctx.args().end()) {
            dst_scales = &arg_dst_scale->second;
        }
        if (pd()->src_float_) {
            execute_internal_reorder(
                    ctx, arg_src, arg_dst, src_scales, dst_scales);
        }
        if (pd()->dst_float_) {
        //    execute_internal_reorder(
        //            ctx, arg_dst, arg_src, arg_dst_scale, arg_src_scale);
        }
        // compat::host_task(cgh, [=, this](const compat::interop_handle &ih) {
        //     auto &sycl_engine = *utils::downcast<nvidia::engine_t *>(
        //             cuda_stream->engine());
        //     auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
        //     auto handle = cuda_stream->get_cudnn_handle();

        //     void *src_ = arg_src.get_native_pointer(ih);
        //     void *dst_ = arg_dst.get_native_pointer(ih);

        //     auto a = static_cast<uint8_t *>(src_)
        //             + pd()->reorder_->src_offset_in_bytes();
        //     auto b = static_cast<uint8_t *>(dst_)
        //             + pd()->reorder_->dst_offset_in_bytes();

        //     void *src_sc = arg_src_scale.get_native_pointer(ih);
        //     void *dst_sc = arg_dst_scale.get_native_pointer(ih);

        //     pd()->reorder_->execute(handle, a, b, src_sc, dst_sc);
        // });
    });
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
