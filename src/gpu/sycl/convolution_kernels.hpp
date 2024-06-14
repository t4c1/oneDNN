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
            xpu::sycl::in_memory_arg_t &weights_scale, 
            xpu::sycl::in_memory_arg_t &dst_scale, 
            xpu::sycl::in_memory_arg_t &data_zeropoints,
            xpu::sycl::in_memory_arg_t &dst_zeropoints, 
            data_type_t scales_data_dt, data_type_t scales_weights_dt, 
            data_type_t zeropoints_data_dt, data_type_t zeropoints_dst_dt)
        : conf_(conf)
        , data_(data)
        , weights_(weights)
        , bias_(bias)
        , dst_(dst)
        , data_scale_(data_scale)
        , weights_scale_(weights_scale)
        , dst_scale_(dst_scale)
        , data_zeropoints_(data_zeropoints)
        , dst_zeropoints_(dst_zeropoints)
        , scales_data_dt_(scales_data_dt)
        , scales_weights_dt_(scales_weights_dt)
        , zeropoints_data_dt_(zeropoints_data_dt)
        , zeropoints_dst_dt_(zeropoints_dst_dt) {}

    void operator()(::sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        size_t wg_offset_t = item.get_group(0) * conf_.wg_size;
        size_t sg_offset_t = sg.get_group_id()[0] * sg.get_local_range()[0];
        size_t wi_offset_t = sg.get_local_id();
        size_t offset_t = wg_offset_t + sg_offset_t + wi_offset_t;

        size_t base_idx = offset_t * conf_.block_size;

        const float sm_data = (conf_.do_scale_data
                        ? load_float_value(scales_data_dt_, data_scale_ptr(), 0)
                        : 1.f);

        float sm_weights = (conf_.do_scale_weights && conf_.single_weight_scale
                        ? load_float_value(scales_weights_dt_, weights_scale_ptr(), 0)
                        : 1.f);
                        
        const float sm_dst = (conf_.do_scale_dst
                        ? load_float_value(data_type::f32, dst_scale_ptr(), 0)
                        : 1.f);

        dims_t data_dims, weights_dims, dst_dims, dst_strides, off;
        for (int i = 0; i < max_supported_ndims; i++) {
            data_dims[i] = (i < data_md().ndims()) ? data_md().dims()[i] : 1;
            weights_dims[i] = (i < weights_md().ndims()) ? weights_md().dims()[i] : 1;
            dst_dims[i] = (i < dst_md().ndims()) ? dst_md().dims()[i] : 1;
            dst_strides[i] = (i < dst_md().ndims()) ? dst_md().strides()[i] : INT_MAX;
        }

        bool no_groups = weights_md().ndims() == data_md().ndims();
        
        const int SD = conf_.strides[0];
        const int SH = conf_.strides[1];
        const int SW = conf_.strides[2];
        
        //per group
        int OC = weights_dims[1];
        int IC = weights_dims[2];

        int KD = weights_dims[3];
        int KH = weights_dims[4];
        int KW = weights_dims[5];
        if(no_groups){
            OC = weights_dims[0];
            IC = weights_dims[1];
            KD = weights_dims[2];
            KH = weights_dims[3];
            KW = weights_dims[4];
        }

        const int PD = conf_.padding[0];
        const int PH = conf_.padding[1];
        const int PW = conf_.padding[2];
        
        const int DD = conf_.dilation[0];
        const int DH = conf_.dilation[1];
        const int DW = conf_.dilation[2];

        for (int i = 0; i < conf_.block_size; i++) {
            int idx = base_idx + i;
            if (idx < conf_.wk_size) {
                for (int i = 0; i < max_supported_ndims; i++) {
                    off[i] = idx / dst_strides[i] % dst_dims[i];
                }

                const int n = off[0];
                const int oc_tot = off[1];
                const int oc = oc_tot % OC;
                const int g = oc_tot / OC;

                const int od = off[2];
                const int oh = off[3];
                const int ow = off[4];

                float accumulator = 0;
                for (int ic = 0; ic < IC; ++ic) {
                    for (int kd = 0; kd < KD; ++kd) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                const int id = od * SD - PD + kd * (1 + DD);
                                const int ih = oh * SH - PH + kh * (1 + DH);
                                const int iw = ow * SW - PW + kw * (1 + DW);

                                if (id < 0 || id >= data_dims[2] || ih < 0 || ih >= data_dims[3] || iw < 0
                                        || iw >= data_dims[4]){
                                    continue;
                                }

                                dims_t off_data{n, g * IC + ic, id, ih, iw};
                                const int data_idx = data_md().off_v(off_data);
                                dims_t off_weights{g, oc, ic, kd, kh, kw};
                                dims_t off_weights_no_groups{oc, ic, kd, kh, kw};
                                const int weights_idx = weights_md().off_v(no_groups ? off_weights_no_groups : off_weights);
                                
                                auto data = load_float_value(
                                        data_md().data_type(), data_ptr(), data_idx);
                                auto weight = load_float_value(
                                        weights_md().data_type(), weights_ptr(), weights_idx);
                                        
                                if(conf_.use_data_zeropoints){
                                    int zpoint_idx = conf_.single_data_zeropoint ? 0 : g * IC + ic;
                                    auto data_zeropoint = load_float_value(
                                            zeropoints_data_dt_, data_zeropoint_ptr(), zpoint_idx);
                                    data -= data_zeropoint;
                                }
                                accumulator += data * weight;
                            }
                        }
                    }
                }
                if(conf_.do_scale_data){
                    accumulator *= sm_data;
                }
                if(conf_.do_scale_weights){
                    if(!conf_.single_weight_scale){
                        sm_weights = load_float_value(scales_weights_dt_, weights_scale_ptr(), oc_tot);
                    }
                    accumulator *= sm_weights;
                }

                if(bias_md().ndims()!=0){
                    auto bias = load_float_value(
                                            bias_md().data_type(), bias_ptr(), oc_tot);
                    accumulator += bias;
                }

                auto dst = load_float_value(
                        conf_.post_ops.sum_dt_ == dnnl_data_type_undef ? dst_md().data_type() : conf_.post_ops.sum_dt_, dst_ptr(), idx);
                accumulator = conf_.post_ops.apply(accumulator, dst);

                if(conf_.do_scale_dst){
                    accumulator /= sm_dst;
                }
                if(conf_.use_dst_zeropoints){
                    int zpoint_idx = conf_.single_dst_zeropoint ? 0 : oc_tot;
                    auto dst_zeropoint = load_float_value(
                            zeropoints_dst_dt_, dst_zeropoint_ptr(), zpoint_idx);
                    accumulator += dst_zeropoint;
                }
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
    void *data_scale_ptr() const {
        return data_scale_.get_pointer();
    }
    void *weights_scale_ptr() const {
        return weights_scale_.get_pointer();
    }
    void *dst_scale_ptr() const {
        return dst_scale_.get_pointer();
    }
    void *data_zeropoint_ptr() const {
        return data_zeropoints_.get_pointer();
    }
    void *dst_zeropoint_ptr() const {
        return dst_zeropoints_.get_pointer();
    }

    sycl_convolution_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t data_scale_;
    xpu::sycl::in_memory_arg_t weights_scale_;
    xpu::sycl::in_memory_arg_t dst_scale_;
    xpu::sycl::in_memory_arg_t data_zeropoints_;
    xpu::sycl::in_memory_arg_t dst_zeropoints_;
    data_type_t scales_data_dt_;
    data_type_t scales_weights_dt_;
    data_type_t zeropoints_data_dt_;
    data_type_t zeropoints_dst_dt_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
