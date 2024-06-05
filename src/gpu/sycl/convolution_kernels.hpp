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

#ifndef GPU_SYCL_CONVOLUTION_KERNELS_HPP
#define GPU_SYCL_CONVOLUTION_KERNELS_HPP

#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct convolution_kernel_vec_t {
    static constexpr int vec_len = 8;
    static constexpr int max_supported_ndims = 6;

    convolution_kernel_vec_t(const sycl_convolution_conf_t &conf,
            xpu::sycl::in_memory_arg_t &data, xpu::sycl::in_memory_arg_t &weights,
            xpu::sycl::in_memory_arg_t &bias, xpu::sycl::out_memory_arg_t &dst,
            xpu::sycl::in_memory_arg_t &data_scale,
            xpu::sycl::in_memory_arg_t &weights_scale, data_type_t scales_dt, ::sycl::stream s)
        : conf_(conf)
        , data_(data)
        , weights_(weights)
        , bias_(bias)
        , dst_(dst)
        , data_scale_(data_scale)
        , weights_scale_(weights_scale)
        , scales_dt_(scales_dt), s(s) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;
        //size_t vec_base_idx = base_idx / vec_len;

        //size_t sg_base_idx = (wg_offset_t + sg_offset_t) * conf_.block_size;

        const float sm_data = (conf_.do_scale_data
                        ? load_float_value(scales_dt_, data_scale_ptr(), 0)
                        : 1.f);

        const float sm_weights = (conf_.do_scale_weights
                        ? load_float_value(scales_dt_, weights_scale_ptr(), 0)
                        : 1.f);

        dims_t data_dims, weights_dims, dst_dims, dst_strides, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            data_dims[i] = (i < data_md().ndims()) ? data_md().dims()[i] : 1;
            weights_dims[i] = (i < weights_md().ndims()) ? weights_md().dims()[i] : 1;
            dst_dims[i] = (i < dst_md().ndims()) ? dst_md().dims()[i] : 1;
            dst_strides[i] = (i < dst_md().ndims()) ? dst_md().strides()[i] : INT_MAX;
        }
        
        //input strides
        const int SD = conf_.strides[0];
        const int SH = conf_.strides[1];
        const int SW = conf_.strides[2];

        const int G = weights_dims[0];
        
        //per group
        const int OC = weights_dims[1];
        const int IC = weights_dims[2];

        const int KD = weights_dims[3];
        const int KH = weights_dims[4];
        const int KW = weights_dims[5];

        //padding?
        const int PD = conf_.padding[0];
        const int PH = conf_.padding[1];
        const int PW = conf_.padding[2];
        for (int i = 0; i < conf_.block_size; i++) {
            int idx = base_idx + i;
            if (idx < conf_.wk_size) {
                if(idx == 0){
                    s << "G " << G << " IC " << IC << " OC " << OC << "\n";
                    s << "KD " << KD << " KH " << KH << " KW " << KW << "\n";
                    s << "PD " << PD << " PH " << PH << " PW " << PW << "\n";
                    s << "w nd " << weights_md().ndims() << "\n";
                    s << "weights dims "
                    << weights_dims[0] << " "
                    << weights_dims[1] << " "
                    << weights_dims[2] << " "
                    << weights_dims[3] << " "
                    << weights_dims[4] << " "
                    << weights_dims[5] << "\n";
                    s << "d nd " << data_md().ndims() << "\n";
                    s << "data dims "
                    << data_dims[0] << " "
                    << data_dims[1] << " "
                    << data_dims[2] << " "
                    << data_dims[3] << " "
                    << data_dims[4] << " "
                    << data_dims[5] << "\n";
                    s << "dst nd " << dst_md().ndims() << "\n";
                    s << "dst dims "
                    << dst_md().dims()[0] << " "
                    << dst_md().dims()[1] << " "
                    << dst_md().dims()[2] << " "
                    << dst_md().dims()[3] << " "
                    << dst_md().dims()[4] << " "
                    << dst_md().dims()[5] << "\n";
                }
                for (int i = 0; i < max_supported_ndims; i++) {
                    off[i] = idx / dst_strides[i] % dst_dims[i];
                }
                s << "\n\n\non idx " << idx << "\n";

                const int n = off[0];
                const int oc_tot = off[1];
                const int oc = oc_tot % OC;
                const int g = oc_tot / OC;

                const int od = off[2];
                const int oh = off[3];
                const int ow = off[4];
                s << "n " << n << " g " << g << " oc " << oc << " od " << od << " oh " << oh << " ow " << ow << "\n";

                
                //dilation?
                const int DD = 1;
                const int DH = 1;
                const int DW = 1;

                float accumulator = 0;
                for (int ic = 0; ic < IC; ++ic) {
                    for (int kd = 0; kd < KD; ++kd) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                const int id = od * SD - PD + kd * DD;
                                const int ih = oh * SH - PH + kh * DH;
                                const int iw = ow * SW - PW + kw * DW;

                                if (id < 0 || id >= data_dims[2] || ih < 0 || ih >= data_dims[3] || iw < 0
                                        || iw >= data_dims[4]){
                                    continue;
                                }
                                s << "n " << n << " ic " << ic << " id " << id << " ih " << ih << " iw " << iw << "\n";

                                dims_t off_data{n, g * IC + ic, id, ih, iw};
                                const int data_idx = data_md().off_v(off_data);
                                
                                dims_t off_weights{g, oc, ic, kd, kh, kw};
                                const int weights_idx = weights_md().off_v(off_weights);
                                s << "w off "
                                    << off_weights[0] << " "
                                    << off_weights[1] << " "
                                    << off_weights[2] << " "
                                    << off_weights[3] << " "
                                    << off_weights[4] << " "
                                    << off_weights[5] << "\n";

                                s << "d off "
                                    << off_data[0] << " "
                                    << off_data[1] << " "
                                    << off_data[2] << " "
                                    << off_data[3] << " "
                                    << off_data[4] << " "
                                    << off_data[5] << "\n";
                                
                                auto data = load_float_value(
                                        data_md().data_type(), data_ptr(), data_idx);
                                auto weight = load_float_value(
                                        weights_md().data_type(), weights_ptr(), weights_idx);
                                s << "load d " << data << " from " << data_idx << ", w " << weight << " from " << weights_idx << "\n\n";
                                accumulator += data * weight;

                                //TODO zeropoints

                            }
                        }
                    }
                }
                s << "val " << accumulator << "\n";
                //TODO scales

                //bias
                auto bias = load_float_value(
                                        bias_md().data_type(), bias_ptr(), oc_tot);
                s << "load bias " << bias << " from " << oc_tot << "\n";
                accumulator += bias;

                auto dst = load_float_value(
                        dst_md().data_type(), dst_ptr(), idx);

                //if (conf_.do_scale_src0) src0 *= sm_0;
                //if (conf_.do_scale_src1) src1 *= sm_1;

                accumulator = conf_.post_ops.apply(accumulator, dst);
                s << "store on " << idx << " val " << accumulator << "\n";
                store_float_value(
                        dst_md().data_type(), accumulator, dst_ptr(), idx);
            }
        }
    }

private:
    const xpu::sycl::md_t &data_md() const { return conf_.data_md; }
    const xpu::sycl::md_t &weights_md() const { return conf_.weights_md; }
    const xpu::sycl::md_t &bias_md() const { return conf_.bias_md; }
    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }

    void *data_ptr() const { return data_.get_pointer(); }
    void *weights_ptr() const { return weights_.get_pointer(); }
    void *bias_ptr() const { return bias_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    float *data_scale_ptr() const {
        return static_cast<float *>(data_scale_.get_pointer());
    }
    float *weights_scale_ptr() const {
        return static_cast<float *>(weights_scale_.get_pointer());
    }

    inline void l_dims_by_l_offset(dims_t dims_pos, dim_t l_offset,
            const xpu::sycl::md_t::dims32_t &dims, const dim_t &ndims) const {
        for (dim_t rd = 0; rd < ndims; ++rd) {
            const dim_t d = ndims - 1 - rd;
            /* switch to faster 32-bit division when possible. */
            if (l_offset <= INT32_MAX && dims[d] <= INT32_MAX) {
                dims_pos[d] = (int32_t)l_offset % (int32_t)dims[d];
                l_offset = (int32_t)l_offset / (int32_t)dims[d];
            } else {
                dims_pos[d] = l_offset % dims[d];
                l_offset /= dims[d];
            }
        }
    }

    sycl_convolution_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t data_scale_;
    xpu::sycl::in_memory_arg_t weights_scale_;
    data_type_t scales_dt_;
    ::sycl::stream s;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
