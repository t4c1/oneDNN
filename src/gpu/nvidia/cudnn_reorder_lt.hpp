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

#ifndef GPU_NVIDIA_CUDNN_REORDER_LT_HPP
#define GPU_NVIDIA_CUDNN_REORDER_LT_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"
// #include "gpu/nvidia/cudnn_reorder_impl.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_reorder_lt_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("cuda:cublaslt:any", cudnn_reorder_lt_t);
	pd_t(const pd_t &rhs) = default;
        ~pd_t() = default;

        // Function to verify data and memory format
        bool valid_data_n_mem_format(impl::engine_t *engine) {
            auto src_dt_ = src_md()->data_type;
            auto dst_dt_ = dst_md()->data_type;
            bool ok = utils::one_of(
                    src_dt_, data_type::f32, data_type::s8, data_type::s32);
            ok = ok
                    && utils::one_of(dst_dt_, data_type::f32, data_type::s8,
                            data_type::s32);

            ok = ok
                    && IMPLICATION(src_dt_ == data_type::f32,
                            utils::one_of(
                                    dst_dt_, data_type::s8, data_type::s32));
            ok = ok
                    && IMPLICATION(dst_dt_ == data_type::f32,
                            utils::one_of(
                                    src_dt_, data_type::s8, data_type::s32));

            src_float_ = src_dt_ == data_type_t::dnnl_f32;
            dst_float_ = dst_dt_ == data_type_t::dnnl_f32;

            if (!ok) return ok;

            memory_desc_wrapper src_wrap(*src_md());
            memory_desc_wrapper dst_wrap(*dst_md());

            // Only support transforming from plain to blocked format and vice-versa.
            ok = ok && IMPLICATION(src_wrap.is_plain(), !dst_wrap.is_plain());
            ok = ok && IMPLICATION(dst_wrap.is_plain(), !src_wrap.is_plain());
            ok = ok && IMPLICATION(!src_wrap.is_plain(), dst_wrap.is_plain());
            ok = ok && IMPLICATION(!dst_wrap.is_plain(), src_wrap.is_plain());
            ok = ok && IMPLICATION(src_float_, src_wrap.is_plain());
            ok = ok && IMPLICATION(dst_float_, dst_wrap.is_plain());

            if (!ok) return ok;

            auto ndims = src_wrap.ndims();
            if (ndims > 3) { return false; }

            auto check_tag = [&](const memory_desc_wrapper &wrap,
                                     bool &transpose, format_kind_t &kind) {
                kind=format_kind_t::dnnl_blocked;
                if (wrap.is_plain()) {
                    auto tag = wrap.matches_one_of_tag(dnnl_ab, dnnl_abc);
                    if (tag != dnnl_format_tag_undef) {
                        transpose = false;
                        return tag;
                    }
                    tag = wrap.matches_one_of_tag(dnnl_ba, dnnl_acb);
                    if (tag != dnnl_format_tag_undef) {
                        transpose = true;
                        return tag;
                    }
                }
                if (wrap.is_blocking_desc()) {
                    transpose = false;
                    return wrap.matches_one_of_tag(dnnl_Ab32a, dnnl_aBc32b);
                }
                if (wrap.is_cublaslt_blocked_desc()) {
                    transpose = false;
                    kind = format_kind::cublaslt_blocked;
                    return dnnl_format_tag_undef;
                }
                return dnnl_format_tag_undef;
            };
            ok = ok && src_wrap.ndims() == dst_wrap.ndims();
            format_kind_t kind;

            src_tag_ = check_tag(src_wrap, src_trans_, kind);
            ok = IMPLICATION(kind==dnnl_blocked, src_tag_!=dnnl_format_tag_undef);

            dst_tag_ = check_tag(dst_wrap, dst_trans_, kind);
            ok = IMPLICATION(kind==dnnl_blocked, src_tag_!=dnnl_format_tag_undef);

            ok = ok
                    && !utils::one_of(
                            dnnl_format_tag_undef, src_tag_, dst_tag_);
            return ok;
        }

        bool scales_ok() const {
            const auto &scales = attr()->scales_;
            const auto &supported_args = {DNNL_ARG_FROM, DNNL_ARG_TO};
            if (!scales.has_default_values(supported_args)) return false;
            // cuDNN does not support scaling per dimension.
            for (auto arg : supported_args)
                if (scales.get(arg).mask_ != 0) return false;
            return true;
        }

        bool post_ops_ok() const {
            // only sum post-op is supported
            const auto &p = attr()->post_ops_;
            return p.len() == 0 || (p.len() == 1 && p.entry_[0].is_sum(false));
        }

        void init_internal_reorder(
                impl::engine_t *engine, bool swap_src_dst = false) {
            primitive_attr_t r_attr;
            int mask = 0;
            bool is_set = false;
            auto src = swap_src_dst ? DNNL_ARG_DST : DNNL_ARG_SRC;
            auto dst = swap_src_dst ? DNNL_ARG_SRC : DNNL_ARG_DST;
            auto src_d = swap_src_dst ? dst_md() : src_md();
            auto dst_d = swap_src_dst ? src_md() : dst_md();
            attr()->scales_.get(src, &mask, &is_set);
            if (is_set) { r_attr.scales_.set(src, mask); }

            attr()->scales_.get(dst, &mask, &is_set);
            if (is_set) { r_attr.scales_.set(dst, mask); }
            reorder_primitive_desc_create(
                    generic_reorder_desc_, engine, src_d, dst_d, &r_attr);
        }

        void init_scratchpad(impl::engine_t *engine) {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    generic_reorder_desc_->scratchpad_registry());
        }
        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::post_ops;
            bool ok = engine == dst_engine
                    && src_engine->kind() == engine_kind::gpu
                    && valid_data_n_mem_format(engine)
                    && attr()->has_default_values(attr_skip_mask) && scales_ok()
                    && post_ops_ok();

            if (src_float_) { init_internal_reorder(engine, false); }
            if (dst_float_) { init_internal_reorder(engine, true); }

            init_scratchpad(engine);

            if (!ok) return status::unimplemented;
            return status::success;
            // return reorder_->init(this);
        }

        bool src_trans_ = false;
        bool dst_trans_ = false;
        bool src_float_ = false;
        bool dst_float_ = false;

        data_type_t src_dt_;
        data_type_t dst_dt_;
        format_tag_t src_tag_;
        format_tag_t dst_tag_;
        // std::shared_ptr<cublaslt_reorder_generic_t> cublaslt_reorder_;
        std::shared_ptr<impl::primitive_desc_t> generic_reorder_desc_;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t execute(const exec_ctx_t &ctx) const override;
    status_t execute_internal_reorder(const exec_ctx_t &ctx,
            const memory_arg_t &src, const memory_arg_t &dst,
            const memory_arg_t *src_scales, const memory_arg_t *dst_scales) const;

private:
    std::shared_ptr<impl::primitive_t> generic_reorder_;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
