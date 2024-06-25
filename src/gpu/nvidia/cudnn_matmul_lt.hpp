/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_LT_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_LT_HPP

#include "gpu/nvidia/cudnn_matmul_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_lt_t : cudnn_matmul_base_t {
    using cudnn_matmul_base_t::cudnn_matmul_base_t;

    struct pd_t : public pd_base_t {
        using pd_base_t::pd_base_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_matmul_lt_t);

        status_t init(impl::engine_t *engine) override {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            data_type_t src_dt = src_md()->data_type;
            data_type_t dst_dt = dst_md()->data_type;
            data_type_t wei_dt = weights_md(0)->data_type;
            data_type_t bia_dt
                    = with_bias() ? weights_md(1)->data_type : data_type::f32;

            bool f32_case = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
            bool f16_case = utils::everyone_is(f16, src_dt, wei_dt, dst_dt);
            bool bf16_case = utils::everyone_is(bf16, src_dt, wei_dt, dst_dt);
#ifdef DNNL_NO_IMMA_INT8_DST
            bool s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                    && utils::one_of(dst_dt, s8, s32);
#else
            bool s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                    && utils::one_of(dst_dt, s32);
#endif
            auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());

            bool ok = is_dense_format_kind() && (blocking_ok() || imma_blocks())
                    && attr()->has_default_values(
                            smask_t::scales_runtime | smask_t::post_ops)
                    && scales_ok() && attr_post_ops_ok(attr())
                    && IMPLICATION(bf16_case,
                            has_bf16_support(sycl_engine_impl->device()))
                    && set_default_formats()
                    && (f32_case || f16_case || bf16_case || s8_case)
                    && IMPLICATION(s8_case && imma_blocks(), !with_bias())
                    && IMPLICATION(with_bias(),
                            (IMPLICATION(f32_case, utils::one_of(bia_dt, f32))
                                    && IMPLICATION(f16_case,
                                            utils::one_of(bia_dt, f16, f32))
                                    && IMPLICATION(bf16_case,
                                            utils::one_of(bia_dt, bf16, f32))
                                    && IMPLICATION(s8_case,
                                            utils::one_of(bia_dt, s8, f32))))
                    && !(with_bias() && s8_case)
                    && IMPLICATION(with_bias(), !has_runtime_dims_or_strides());

            memory_desc_wrapper weight_wrap(weights_md());
            memory_desc_wrapper dst_wrap(dst_md());

            ok = ok
                    && IMPLICATION(
                            is_md_col32(weight_wrap) || is_md_col32(dst_wrap),
                            s8_case);
            ok = ok && (imma_blocks() || dst_ok()) && bias_ok() && eltwise_ok();
            if (!ok) return status::unimplemented;

            if (!with_bias() && !with_eltwise() && !s8_case) {
                return status::unimplemented;
            }
            if (s8_case
                    && ((with_bias() && bias_ok())
                            || (with_eltwise() && eltwise_ok()))) {
                return status::unimplemented;
            }

            if (src_md()->ndims > 3) return status::unimplemented;

            return status::success;
        }

    private:
        bool dst_ok() {
            bool ok = false;

            memory_desc_wrapper dst_wrap(dst_md());
            //check if dst is col_major
            bool isbatched = batched() && dst_wrap.dims()[0];
            const auto &md_strides
                    = &dst_wrap.blocking_desc().strides[isbatched];
            ok = (md_strides[1] == 1 && dst_wrap.dims()[isbatched + 0] > 1);
            // dst not supported for ndims = 1
            ok = ok
                    && (dst_wrap.dims()[isbatched + 1] != 1
                            && dst_wrap.dims()[isbatched + 0] != 1);

            return ok;
        }

        bool bias_ok() {

            memory_desc_wrapper dst_wrap(dst_md());
            memory_desc_wrapper bia_wrap(weights_md(1));

            bool isbatched = batched() && dst_wrap.dims()[0];
            if (!with_bias()) { return true; }
            bool ok = !(bia_wrap.data_type() != dst_wrap.data_type());
            if (bia_wrap.dims()[0 + isbatched] != 1) { ok = false; }

            if (!has_runtime_dims_or_strides()) {
                auto M = dst_wrap.dims()[isbatched + 1];
                auto N = dst_wrap.dims()[isbatched + 0];
                if ((bia_wrap.dims()[1 + isbatched] != M
                            || bia_wrap.dims()[0 + isbatched] != 1)
                        || M == 1 || N == 1) {
                    ok = false;
                }
            }
            return ok;
        }

        bool with_eltwise() {
            return attr()->post_ops_.contain(primitive_kind::eltwise, 0)
                    || attr()->post_ops_.contain(primitive_kind::eltwise, 1);
        }

        bool eltwise_ok() {
            bool with_elt = with_eltwise();
            if (!with_elt) { return true; }

            int eltwise_idx_ = attr()->post_ops_.find(primitive_kind::eltwise);
            auto eltwise_algo
                    = attr()->post_ops_.entry_[eltwise_idx_].eltwise.alg;
            if (eltwise_algo == alg_kind::eltwise_relu) { return true; }
            return false;
        }

        bool imma_blocks() {
            // weights should be blocked in Ab32a, ab or ba
            bool weights_supported = false;
            memory_desc_wrapper weight_wrap(weights_md());
            if (is_md_col32(weight_wrap) || weight_wrap.is_plain()) {
                weights_supported = true;
            }
            // src not blocked
            bool src_supported = false;
            memory_desc_wrapper src_wrap(src_md());
            if (src_wrap.is_plain()) { src_supported = true; }
            // dst blocked in Ab32a, ab or ba
            bool dst_supported = false;
            memory_desc_wrapper dst_wrap(dst_md());
            if (is_md_col32(dst_wrap) || dst_wrap.is_plain()) {
                dst_supported = true;
            }
            return (weights_supported && src_supported && dst_supported);
        }
    };

    status_t init(impl::engine_t *engine) override {
        // LT matmul
        matmul_impl_.reset(new cudnn_matmul_lt_impl_t());
        auto status = matmul_impl_->init((matmul_pd_t *)pd(), engine);

        bool has_runtime_args = matmul_impl_->has_runtime_params();
        if (has_runtime_args) {
            executor_.reset(new cudnn_matmul_lt_runtime_args_exec_t);
        } else if (!has_runtime_args) {
            executor_.reset(new cudnn_matmul_lt_exec_t);
        }

        return status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_;
    std::shared_ptr<cudnn_matmul_lt_base_exec_t> executor_;

private:
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
