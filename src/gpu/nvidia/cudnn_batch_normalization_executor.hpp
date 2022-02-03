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

#ifndef GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_EXECUTOR_HPP
#define GPU_NVIDIA_CUDNN_BATCH_NORMALIZATION_EXECUTOR_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_batch_normalization_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

#include "sycl_cuda_helper.hpp"

#include <unistd.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct bnorm_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cudnn_batch_normalization_impl_base_t>
                    bnorm_impl) const = 0;
    virtual ~bnorm_exec_base_t() = default;

protected:
    template <typename T, ::sycl::access::mode md, typename sc_t>
    void *mean_var_ptr(::sycl::accessor<T, 1, md> acc, sc_t &sc,
            const compat::interop_handle &ih) const {
        return sc.template memory<void *>(ih, acc);
    }

    template <typename sc_t>
    std::nullptr_t mean_var_ptr(std::nullptr_t acc, sc_t &,
            const compat::interop_handle &ih) const {
        return acc;
    }

    template <typename read_acc_t, typename write_acc_t, typename float_acc_t>
    void interop_task_fwd(
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, ::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream,
            std::optional<read_acc_t> src_acc,
            sycl::sycl_memory_storage_base_t *src_mem,
            std::optional<write_acc_t> dst_acc,
            sycl::sycl_memory_storage_base_t *dst_mem,
            std::optional<write_acc_t> mean_acc,
            sycl::sycl_memory_storage_base_t *mean_mem,
            std::optional<write_acc_t> var_acc,
            sycl::sycl_memory_storage_base_t *var_mem,
            float_acc_t scale_bias_acc,
            sycl::sycl_memory_storage_base_t *scale_bias_mem,
            std::optional<write_acc_t> wkspace_acc,
            sycl::sycl_memory_storage_base_t *wkspace_mem, bool init_ss,
            bool init_mean_var) const {

        maybe_init_mean_var(cuda_stream, mean_acc, var_acc, init_mean_var);
        /*
        maybe_init_ss(cuda_stream, scale_bias_acc, scale_bias_mem,
                bnorm_impl->C(),
                init_ss); // TODO this needs to be rewritten
*/
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();
            
            maybe_init_ss(sc, ih, cuda_stream, scale_bias_acc, scale_bias_mem,
                bnorm_impl->C(), init_ss); // TODO this needs to be rewritten

            auto x = get_cudnn_ptr(sc, ih, src_acc, src_mem);
            auto y = get_cudnn_ptr(sc, ih, dst_acc, dst_mem);

            auto mean = (mean_mem || mean_acc.has_value())
                    ? get_cudnn_ptr(sc, ih, mean_acc, mean_mem)
                    : nullptr;
            auto var = (var_mem || var_acc.has_value())
                    ? get_cudnn_ptr(sc, ih, var_acc, var_mem)
                    : nullptr;

            auto scale = static_cast<float *>(
                    get_cudnn_ptr(sc, ih, scale_bias_acc, scale_bias_mem));
            auto bias = scale + bnorm_impl->C();

            uint8_t *y_prime = nullptr, *save_mean = nullptr,
                    *save_var = nullptr;
            if (!wkspace_mem->is_null()) {
                save_mean = static_cast<uint8_t *>(
                        get_cudnn_ptr(sc, ih, wkspace_acc, wkspace_mem));
                save_var = save_mean + bnorm_impl->mean_var_size_bytes();
                y_prime = save_var + bnorm_impl->mean_var_size_bytes();
            }

            void * p1 = static_cast<void *>(scale);
            void * p2 = static_cast<void *>(bias);

            printf("Using pointers: x %p, y %p, mean %p, var %p, scale %p, "
                   "bias %p, y_prime %p, save_mean %p, save_var %p\n\n",
                    x, y, mean, var, static_cast<void *>(scale), static_cast<void *>(bias), y_prime, save_mean, save_var);
            std::shared_ptr<bnorm_args_t> args(new bnorm_fwd_args_t(x, y, mean,
                    var, p1, p2, y_prime, save_mean, save_var));
            auto fwd_args = static_cast<bnorm_fwd_args_t *>(args.get());
        printf( "FIRST TIME! "
                "About to execute cudnnBatchNormalization!\nPointers: "
                " x %p, y %p, fwd_args->scale_ %p, "
                "fwd_args->bias_ %p, "
                "fwd_args->mean_ %p, fwd_args->var_ %p\n\n",
                fwd_args->x_, fwd_args->y_, fwd_args->scale_, fwd_args->bias_,
                fwd_args->mean_, fwd_args->var_);

            bnorm_impl->execute(handle, args);
            cudaDeviceSynchronize();
        });
    }

    template <typename read_acc_t, typename write_acc_t, typename ss_acc_t,
            typename d_ss_acc_t>
    void interop_task_bwd(
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, ::sycl::handler &cgh,
            nvidia::sycl_cuda_stream_t *cuda_stream,
            std::optional<read_acc_t> src_acc,
            sycl::sycl_memory_storage_base_t *src_mem,
            std::optional<read_acc_t> diff_dst_acc,
            sycl::sycl_memory_storage_base_t *diff_dst_mem,
            std::optional<write_acc_t> diff_src_acc,
            sycl::sycl_memory_storage_base_t *diff_src_mem,
            ss_acc_t scale_bias_acc,
            sycl::sycl_memory_storage_base_t *scale_bias_mem,
            d_ss_acc_t diff_scale_bias_acc,
            sycl::sycl_memory_storage_base_t *diff_scale_bias_mem,
            std::optional<read_acc_t> wkspace_acc,
            sycl::sycl_memory_storage_base_t *wkspace_mem,
            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output,
            bool init_ss, bool init_mean_var) const {
/*
        maybe_init_ss(cuda_stream, scale_bias_acc, scale_bias_mem,
                bnorm_impl->C(),
                init_ss); // TODO this needs to be rewritten
                */
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = cuda_stream->get_cudnn_handle();
        
            maybe_init_ss(sc, ih, cuda_stream, scale_bias_acc, scale_bias_mem,
                bnorm_impl->C(),
                init_ss); // TODO this needs to be rewritten

            auto x = get_cudnn_ptr(sc, ih, src_acc, src_mem);
            auto dy = get_cudnn_ptr(sc, ih, diff_dst_acc, diff_dst_mem);
            auto dx = get_cudnn_ptr(sc, ih, diff_src_acc, diff_src_mem);

            auto scale = static_cast<uint8_t *>(
                    get_cudnn_ptr(sc, ih, scale_bias_acc, scale_bias_mem));
            auto bias = scale + bnorm_impl->C() * sizeof(float);

            auto diff_scale = static_cast<uint8_t *>(get_cudnn_ptr(
                    sc, ih, diff_scale_bias_acc, diff_scale_bias_mem));
            auto diff_bias = diff_scale + (bnorm_impl->C() * sizeof(float));

            auto save_mean = static_cast<uint8_t *>(
                    get_cudnn_ptr(sc, ih, wkspace_acc, wkspace_mem));
            auto save_var = save_mean + bnorm_impl->mean_var_size_bytes();
            auto wkspace = save_var + bnorm_impl->mean_var_size_bytes();
            auto relu_dy = bnorm_impl->fuse_norm_relu()
                    ? sc.memory<void *>(ih, *temp_relu_output)
                    : nullptr;
            std::shared_ptr<bnorm_args_t> args(
                    new bnorm_bwd_args_t(x, dx, dy, save_mean, save_var, scale,
                            bias, diff_scale, diff_bias, wkspace, relu_dy));

            bnorm_impl->execute(handle, args);
        });
    }

    template <typename T>
    void maybe_init_ss(cuda_sycl_scoped_context_handler_t &sc, const compat::interop_handle &ih, nvidia::sycl_cuda_stream_t *cuda_stream, T,
            sycl::sycl_memory_storage_base_t *, size_t, bool) const {
        printf("NULL MAYBE INIT SS!!\n\n");
    }

    template <typename T>
    void maybe_init_ss(cuda_sycl_scoped_context_handler_t &sc, const compat::interop_handle &ih, nvidia::sycl_cuda_stream_t *cuda_stream,
            std::optional<::sycl::accessor<T, 1, ::sycl::access::mode::write>>
                    scale_bias_acc,
            sycl::sycl_memory_storage_base_t *scale_bias_mem, size_t n,
            bool init_ss) const {
        if (init_ss) {
            T *scale_ptr, *bias_ptr;
            constexpr T scale_val = 1, bias_val = 0;
            if (!scale_bias_mem
                    || scale_bias_mem->memory_kind()
                            == sycl::memory_kind::buffer) {

                cuda_stream->interop_task([&](::sycl::handler &cgh) {
                    scale_ptr = sc.memory<T *>(ih, scale_bias_acc.value());
                    printf("Scale ptr %p\n\n", scale_ptr);
                    cudaMemset(static_cast<void *>(scale_ptr), scale_val, n*sizeof(T));
                });
                cuda_stream->interop_task([&](::sycl::handler &cgh) {
                    bias_ptr = sc.memory<T *>(ih, scale_bias_acc.value()) + n;
                    printf("Bias ptr %p\n\n", scale_ptr);
                    cudaMemset(static_cast<void *>(bias_ptr), bias_val, n*sizeof(T));
                });

            } else if (scale_bias_mem->memory_kind()
                    == sycl::memory_kind::usm) {
                    scale_ptr = static_cast<T *>(
                            utils::downcast<
                                    const sycl::sycl_usm_memory_storage_t *>(
                                    scale_bias_mem)
                                    ->usm_ptr());
                    bias_ptr
                            = static_cast<
                                      T *>(utils::downcast<
                                           const sycl::sycl_usm_memory_storage_t
                                                   *>(scale_bias_mem)
                                                   ->usm_ptr())
                            + n;
                    cuda_stream->queue().memset(scale_ptr, scale_val, n);
                    cuda_stream->queue().memset(bias_ptr, bias_val, n);
            }
        }
    }

    // Handle the cases when mean and var are read-only accessors or nullptr
    template <typename T>
    void maybe_init_mean_var(
            nvidia::sycl_cuda_stream_t *cuda_stream, T, T, bool) const {}

    template <typename T, typename write_acc_type>
    void maybe_init_mean_var(nvidia::sycl_cuda_stream_t *cuda_stream,
            write_acc_type mean_acc, write_acc_type var_acc,
            bool init_mean_var) const {
        if (init_mean_var) {
            constexpr T mean_var_val = 0;
            cuda_stream->interop_task([&](::sycl::handler &cgh) {
                cgh.fill(mean_acc, mean_var_val);
            });

            cuda_stream->interop_task([&](::sycl::handler &cgh) {
                cgh.fill(var_acc, mean_var_val);
            });
        }
    }
};

struct bnorm_exec_fwd_inf_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> scale_bias_buff(n_channels * 2);
        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *dst_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_DST));
            auto dst_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(dst_mem, cgh);
            auto *wkspace_mem = bnorm_impl->is_training()
                    ? static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE))
                    : static_cast<sycl::sycl_memory_storage_base_t *>(
                            &memory_storage_t::empty_storage());
            std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>
                    wkspace_acc;
            if (!wkspace_mem->is_null()) {
                wkspace_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                        DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            }

            sycl::sycl_memory_storage_base_t *scale_bias_mem = nullptr;
            auto scale_bias_acc = std::optional(
                    scale_bias_buff.get_access<::sycl::access::mode::write>(
                            cgh));

            bool init_ss = true, init_mean_var = false;

            auto nullptr_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(nullptr, cgh);

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, dst_acc, dst_mem, nullptr_acc, nullptr,
                    nullptr_acc, nullptr, scale_bias_acc, scale_bias_mem,
                    wkspace_acc, wkspace_mem, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *dst_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_DST));
            auto dst_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(dst_mem, cgh);
            auto *wkspace_mem = bnorm_impl->is_training()
                    ? static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE))
                    : static_cast<sycl::sycl_memory_storage_base_t *>(
                            &memory_storage_t::empty_storage());
            std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>
                    wkspace_acc;
            if (!wkspace_mem->is_null()) {
                wkspace_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                        DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            }
            auto *scale_bias_mem
                    = static_cast<sycl::sycl_buffer_memory_storage_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT));
            auto scale_bias_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SCALE_SHIFT))>(scale_bias_mem, cgh);

            bool init_ss = false, init_mean_var = false;

            auto nullptr_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(nullptr, cgh);

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, dst_acc, dst_mem, nullptr_acc, nullptr,
                    nullptr_acc, nullptr, scale_bias_acc, scale_bias_mem,
                    wkspace_acc, wkspace_mem, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> scale_bias_buff(n_channels * 2);
        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *dst_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_DST));
            auto dst_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(dst_mem, cgh);
            auto *wkspace_mem = bnorm_impl->is_training()
                    ? static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE))
                    : static_cast<sycl::sycl_memory_storage_base_t *>(
                            &memory_storage_t::empty_storage());
            std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>
                    wkspace_acc;
            if (!wkspace_mem->is_null()) {
                wkspace_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                        DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            }
            auto *mean_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_MEAN));
            auto mean_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_MEAN))>(mean_mem, cgh);
            auto *var_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_VARIANCE));
            auto var_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_VARIANCE))>(var_mem, cgh);

            sycl::sycl_memory_storage_base_t *scale_bias_mem = nullptr;
            auto scale_bias_acc = std::optional(
                    scale_bias_buff.get_access<::sycl::access::mode::write>(
                            cgh));

            bool init_ss = true, init_mean_var = false;

            auto nullptr_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(nullptr, cgh);

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, dst_acc, dst_mem, nullptr_acc, nullptr,
                    nullptr_acc, nullptr, scale_bias_acc, scale_bias_mem,
                    wkspace_acc, wkspace_mem, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());
        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *dst_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_DST));
            auto dst_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(dst_mem, cgh);
            auto *wkspace_mem = bnorm_impl->is_training()
                    ? static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE))
                    : static_cast<sycl::sycl_memory_storage_base_t *>(
                            &memory_storage_t::empty_storage());
            std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>
                    wkspace_acc;
            if (!wkspace_mem->is_null()) {
                wkspace_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                        DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            }
            auto *mean_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_MEAN));
            auto mean_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_MEAN))>(mean_mem, cgh);
            auto *var_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_VARIANCE));
            auto var_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_VARIANCE))>(var_mem, cgh);
            auto *scale_bias_mem
                    = static_cast<sycl::sycl_buffer_memory_storage_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT));
            auto scale_bias_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SCALE_SHIFT))>(scale_bias_mem, cgh);

            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, dst_acc, dst_mem, mean_acc, mean_mem, var_acc,
                    var_mem, scale_bias_acc, scale_bias_mem, wkspace_acc,
                    wkspace_mem, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();

        ::sycl::buffer<float> scale_bias_buff(n_channels * 2);
        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *dst_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_DST));
            auto dst_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(dst_mem, cgh);
            auto *wkspace_mem = bnorm_impl->is_training()
                    ? static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE))
                    : static_cast<sycl::sycl_memory_storage_base_t *>(
                            &memory_storage_t::empty_storage());
            std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>
                    wkspace_acc;
            if (!wkspace_mem->is_null()) {
                wkspace_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                        DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            }
            auto *mean_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_MEAN));
            auto mean_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_MEAN))>(mean_mem, cgh);
            auto *var_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_VARIANCE));
            auto var_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_VARIANCE))>(var_mem, cgh);

            sycl::sycl_memory_storage_base_t *scale_bias_mem = nullptr;
            auto scale_bias_acc = std::optional(
                    scale_bias_buff.get_access<::sycl::access::mode::write>(
                            cgh));
            bool init_ss = true, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, dst_acc, dst_mem, mean_acc, mean_mem, var_acc,
                    var_mem, scale_bias_acc, scale_bias_mem, wkspace_acc,
                    wkspace_mem, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *dst_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_DST));
            auto dst_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DST))>(dst_mem, cgh);
            auto *wkspace_mem = bnorm_impl->is_training()
                    ? static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE))
                    : static_cast<sycl::sycl_memory_storage_base_t *>(
                            &memory_storage_t::empty_storage());
            std::optional<decltype(CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE))>
                    wkspace_acc;
            if (!wkspace_mem->is_null()) {
                wkspace_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                        DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            }
            auto *mean_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_MEAN));
            auto mean_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_MEAN))>(mean_mem, cgh);
            auto *var_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_OUT_STORAGE(DNNL_ARG_VARIANCE));
            auto var_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_VARIANCE))>(var_mem, cgh);
            auto *scale_bias_mem
                    = static_cast<sycl::sycl_buffer_memory_storage_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT));
            auto scale_bias_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SCALE_SHIFT))>(scale_bias_mem, cgh);

            bool init_ss = false, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, dst_acc, dst_mem, mean_acc, mean_mem, var_acc,
                    var_mem, scale_bias_acc, scale_bias_mem, wkspace_acc,
                    wkspace_mem, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> scale_bias_buff(n_channels * 2);
        ::sycl::buffer<float> diff_scale_buff(n_channels * 2);

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *diff_dst_mem
                    = static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST));
            auto diff_dst_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_DIFF_DST))>(diff_dst_mem, cgh);
            auto *diff_src_mem
                    = static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC));
            auto diff_src_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DIFF_SRC))>(diff_src_mem, cgh);
            auto *wkspace_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_WORKSPACE));
            auto wkspace_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);

            sycl::sycl_memory_storage_base_t *scale_bias_mem = nullptr;
            auto scale_bias_acc = std::optional(
                    scale_bias_buff.get_access<::sycl::access::mode::write>(
                            cgh));
            sycl::sycl_memory_storage_base_t *diff_scale_bias_mem = nullptr;
            auto diff_scale_bias_acc = std::optional(
                    scale_bias_buff.get_access<::sycl::access::mode::write>(
                            cgh));

            bool init_ss = true, init_mean_var = false;

            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<::sycl::accessor<uint8_t, 1,
                        ::sycl::access::mode::read_write,
                        sycl::compat::target_device>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, diff_dst_acc, diff_dst_mem, diff_src_acc,
                    diff_src_mem, scale_bias_acc, scale_bias_mem,
                    diff_scale_bias_acc, diff_scale_bias_mem, wkspace_acc,
                    wkspace_mem, temp_relu_output, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_dw_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *diff_dst_mem
                    = static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST));
            auto diff_dst_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_DIFF_DST))>(diff_dst_mem, cgh);
            auto *diff_src_mem
                    = static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC));
            auto diff_src_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DIFF_SRC))>(diff_src_mem, cgh);
            auto *scale_bias_mem
                    = static_cast<sycl::sycl_buffer_memory_storage_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT));
            auto scale_bias_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SCALE_SHIFT))>(scale_bias_mem, cgh);
            auto *diff_scale_bias_mem
                    = static_cast<sycl::sycl_buffer_memory_storage_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE_SHIFT));
            auto diff_scale_bias_acc
                    = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                            DNNL_ARG_DIFF_SCALE_SHIFT))>(
                            diff_scale_bias_mem, cgh);
            auto *wkspace_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_WORKSPACE));
            auto wkspace_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            bool init_ss = false, init_mean_var = false;

            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<::sycl::accessor<uint8_t, 1,
                        ::sycl::access::mode::read_write,
                        sycl::compat::target_device>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }
            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, diff_dst_acc, diff_dst_mem, diff_src_acc,
                    diff_src_mem, scale_bias_acc, scale_bias_mem,
                    diff_scale_bias_acc, diff_scale_bias_mem, wkspace_acc,
                    wkspace_mem, temp_relu_output, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_d_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cudnn_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        nvidia::sycl_cuda_stream_t *cuda_stream
                = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> diff_scale_bias_buff(n_channels * 2);

        return cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto *src_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_SRC));
            auto src_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SRC))>(src_mem, cgh);
            auto *diff_dst_mem
                    = static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST));
            auto diff_dst_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_DIFF_DST))>(diff_dst_mem, cgh);
            auto *diff_src_mem
                    = static_cast<sycl::sycl_memory_storage_base_t *>(
                            &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC));
            auto diff_src_acc = get_cudnn_accessor<decltype(CTX_OUT_ACCESSOR(
                    DNNL_ARG_DIFF_SRC))>(diff_src_mem, cgh);
            auto *scale_bias_mem
                    = static_cast<sycl::sycl_buffer_memory_storage_t *>(
                            &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT));
            auto scale_bias_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_SCALE_SHIFT))>(scale_bias_mem, cgh);
            sycl::sycl_memory_storage_base_t *diff_scale_bias_mem = nullptr;
            auto diff_scale_bias_acc = std::optional(
                    diff_scale_bias_buff
                            .get_access<::sycl::access::mode::write>(cgh));
            auto *wkspace_mem = static_cast<sycl::sycl_memory_storage_base_t *>(
                    &CTX_IN_STORAGE(DNNL_ARG_WORKSPACE));
            auto wkspace_acc = get_cudnn_accessor<decltype(CTX_IN_ACCESSOR(
                    DNNL_ARG_WORKSPACE))>(wkspace_mem, cgh);
            bool init_ss = false, init_mean_var = false;

            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<::sycl::accessor<uint8_t, 1,
                        ::sycl::access::mode::read_write,
                        sycl::compat::target_device>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, cuda_stream, src_acc,
                    src_mem, diff_dst_acc, diff_dst_mem, diff_src_acc,
                    diff_src_mem, scale_bias_acc, scale_bias_mem,
                    diff_scale_bias_acc, diff_scale_bias_mem, wkspace_acc,
                    wkspace_mem, temp_relu_output, init_ss, init_mean_var);
        });
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
