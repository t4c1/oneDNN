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

#include "gpu/sycl/ref_convolution.hpp"
#include "gpu/sycl/convolution_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_convolution_fwd_t::pd_t::init_conf() {
    conf_ = sycl_convolution_conf_t();

    conf_.data_md = xpu::sycl::md_t(src_md());
    conf_.weights_md = xpu::sycl::md_t(weights_md(0));
    conf_.bias_md = xpu::sycl::md_t(weights_md(1));
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.ndims = ndims();

    // XXX: should probably be tuned.
    conf_.block_size = 16;
    conf_.wg_size = 32;

    conf_.wk_size = memory_desc_wrapper(dst_md()).nelems();

    // Limitations:
    // - Only common scale policy is supported.
    conf_.do_scale_data
            = !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.do_scale_weights
            = !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values();

    conf_.post_ops = sycl_post_ops_t(attr());
    
    //conf_.padding = {padFront(), padT(), padL()};
    auto pad = desc()->padding[0];
    //utils::array_copy(conf_.padding, pad,3);
    conf_.padding[0] = static_cast<int>(pad[0]);
    conf_.padding[1] = static_cast<int>(pad[1]);
    conf_.padding[2] = static_cast<int>(pad[2]);
    std::cout << "pad "
              << padFront() << " "
              << padBack() << " "
              << padT() << " "
              << padB() << " "
              << padL() << " "
              << padR() << "\n";
    return status::success;
}

status_t ref_convolution_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<convolution_kernel_vec_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto data_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_0);
        auto weights_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS);
        auto bias_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_BIAS);
        auto data_scale_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0);
        auto weights_scale_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

        auto scales_dt = (pd()->conf_.do_scale_data)
                ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0)
                          .data_type()
                : data_type_t::dnnl_f32;

        ::sycl::stream s(1024*10,1024*10,cgh);

        convolution_kernel_vec_t convolution_kernel(pd()->conf_, data_mem_arg,
                weights_mem_arg, bias_mem_arg, dst_mem_arg, data_scale_mem_arg,
                weights_scale_mem_arg, scales_dt, s);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        const int t_work = pd()->conf_.wk_size;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = utils::div_up(t_work, wg_work);

        cgh.parallel_for(
                ::sycl::nd_range<1>(wg_cnt * wg_size, wg_size), convolution_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
