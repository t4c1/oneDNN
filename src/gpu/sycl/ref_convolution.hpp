/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_SYCL_REF_CONVOLUTION_HPP
#define GPU_SYCL_REF_CONVOLUTION_HPP

#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "sycl/sycl_stream.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_convolution_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public convolution_fwd_pd_t {
        using convolution_fwd_pd_t::convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper data_d(src_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper dst_d(dst_md());

            const bool ok = is_fwd() 
                    && check_work_amount(weights_d)
                    && set_default_formats()
                    //&& set_default_params() == status::success
                    && attr_.set_default_formats(dst_md()) == status::success
                    && check_data_types(data_d, weights_d, dst_d)
                    && check_formats(data_d, weights_d, dst_d)
                    && attr()->has_default_values(
                            sm::scales_runtime | sm::zero_points_runtime | sm::post_ops)
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask())
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        sycl_convolution_conf_t conf_;

    private:
        status_t init_conf();

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool check_scales_mask() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
            return attr_scales_ok(supported_args);
        }

        bool post_ops_ok() const {
            for (int i = 0; i < attr()->post_ops_.len(); i++) {
                const auto &e = attr()->post_ops_.entry_[i];
                if (!IMPLICATION(e.is_eltwise(),
                            utils::one_of(e.eltwise.alg, alg_kind::eltwise_relu, 
                                    alg_kind::eltwise_linear, alg_kind::eltwise_clip,
                                    alg_kind::eltwise_clip_v2, alg_kind::eltwise_hardswish))) {
                    return false;
                }
            }
            // Dw conv post-ops are not supported.
            return attr()->post_ops_.len() <= sycl_post_ops_t::max_post_ops
                    && attr()->post_ops_.has_default_values(
                            {primitive_kind::eltwise, primitive_kind::convolution,
                                    primitive_kind::prelu,
                                    primitive_kind::sum});
        }

        static bool check_data_types(const memory_desc_wrapper &src0,
                const memory_desc_wrapper &src1,
                const memory_desc_wrapper &dst) {
            using namespace data_type;

            const auto src0_dt = src0.data_type();
            const auto src1_dt = src1.data_type();
            const auto dst_dt = dst.data_type();

            for (auto t : {src0_dt, src1_dt, dst_dt}) {
                if (!utils::one_of(t, /*f64,*/ f32, bf16, f16, s32, s8, u8)) return false;
            }

            return true/*IMPLICATION(utils::one_of(bf16, src0_dt, src1_dt, dst_dt),
                    src0_dt == dst_dt && src1_dt == dst_dt)*/;
        }

        static bool check_formats(const memory_desc_wrapper &src0,
                const memory_desc_wrapper &src1,
                const memory_desc_wrapper &dst) {
            using namespace format_tag;

            for (const auto &mdw : {src0, src1, dst}) {
                if (!mdw.is_plain()) { return false; }
            }
            return true;
        }

        bool check_work_amount(const memory_desc_wrapper &weights){
            auto elems = weights.nelems();
            auto works_per_output = elems / OC();
            // arbitrarily chosen threshold to avoid unreasonably long runtimes
            // such cases should use a different implementation
            return works_per_output < 100000;
        }
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    intel::compute::kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
