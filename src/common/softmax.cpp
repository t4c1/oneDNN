/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

namespace {
status_t softmax_desc_init(softmax_desc_t *softmax_desc,
        primitive_kind_t prim_kind, prop_kind_t prop_kind,
        const memory_desc_t *data_desc, const memory_desc_t *diff_desc,
        int softmax_axis) {
    bool args_ok = !any_null(softmax_desc, data_desc)
            && IMPLICATION(prop_kind == backward_data, diff_desc != nullptr)
            && 0 <= softmax_axis && softmax_axis < data_desc->ndims
            && IMPLICATION(
                    one_of(prop_kind, forward_training, forward_inference),
                    !memory_desc_wrapper(data_desc).format_any());
    if (!args_ok) return invalid_arguments;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(data_desc).has_runtime_dims_or_strides();
    if (prop_kind == backward_data)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_desc).has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    auto sd = softmax_desc_t();
    sd.primitive_kind = prim_kind;
    sd.prop_kind = prop_kind;

    sd.data_desc = *data_desc;
    if (sd.prop_kind == backward_data) sd.diff_desc = *diff_desc;
    sd.softmax_axis = softmax_axis;

    *softmax_desc = sd;
    return success;
}

status_t softmax_v2_desc_init(softmax_v2_desc_t *softmax_v2_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *diff_src_desc, const memory_desc_t *diff_dst_desc,
        int softmax_axis) {
    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    bool args_ok = !any_null(softmax_v2_desc, dst_desc)
            && IMPLICATION(is_fwd, src_desc != nullptr)
            && IMPLICATION(!is_fwd, !any_null(diff_src_desc, diff_dst_desc))
            && one_of(alg_kind, softmax_accurate, softmax_log)
            && 0 <= softmax_axis && softmax_axis < dst_desc->ndims
            && IMPLICATION(is_fwd, !memory_desc_wrapper(src_desc).format_any())
            && IMPLICATION(
                    !is_fwd, !memory_desc_wrapper(dst_desc).format_any());
    if (!args_ok) return invalid_arguments;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (is_fwd) {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(src_desc).has_runtime_dims_or_strides();
    } else {
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_src_desc)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(diff_dst_desc)
                           .has_runtime_dims_or_strides();
    }
    if (runtime_dims_or_strides) return unimplemented;

    auto sd = softmax_v2_desc_t();
    sd.primitive_kind = primitive_kind::softmax_v2;
    sd.prop_kind = prop_kind;

    if (is_fwd) sd.src_desc = *src_desc;
    if (!is_fwd) sd.diff_src_desc = *diff_src_desc;
    sd.softmax_axis = softmax_axis;
    sd.alg_kind = alg_kind;
    sd.dst_desc = *dst_desc;
    if (!is_fwd) sd.diff_dst_desc = *diff_dst_desc;

    *softmax_v2_desc = sd;
    return success;
}
} // namespace

status_t dnnl_softmax_forward_desc_init(softmax_desc_t *softmax_desc,
        prop_kind_t prop_kind, const memory_desc_t *data_desc,
        int softmax_axis) {
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;
    return softmax_desc_init(softmax_desc, primitive_kind::softmax, prop_kind,
            data_desc, nullptr, softmax_axis);
}

status_t dnnl_softmax_backward_desc_init(softmax_desc_t *softmax_desc,
        const memory_desc_t *diff_desc, const memory_desc_t *data_desc,
        int softmax_axis) {
    return softmax_desc_init(softmax_desc, primitive_kind::softmax,
            prop_kind::backward_data, data_desc, diff_desc, softmax_axis);
}

status_t dnnl_logsoftmax_forward_desc_init(logsoftmax_desc_t *logsoftmax_desc,
        prop_kind_t prop_kind, const memory_desc_t *data_desc,
        int logsoftmax_axis) {
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;
    return softmax_desc_init(logsoftmax_desc, primitive_kind::logsoftmax,
            prop_kind, data_desc, nullptr, logsoftmax_axis);
}

status_t dnnl_logsoftmax_backward_desc_init(logsoftmax_desc_t *logsoftmax_desc,
        const memory_desc_t *diff_desc, const memory_desc_t *data_desc,
        int logsoftmax_axis) {
    return softmax_desc_init(logsoftmax_desc, primitive_kind::logsoftmax,
            prop_kind::backward_data, data_desc, diff_desc, logsoftmax_axis);
}

status_t dnnl_softmax_v2_forward_desc_init(softmax_v2_desc_t *softmax_v2_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        int softmax_axis) {
    if (!one_of(prop_kind, forward_inference, forward_training))
        return invalid_arguments;
    return softmax_v2_desc_init(softmax_v2_desc, prop_kind, alg_kind, src_desc,
            dst_desc, nullptr, nullptr, softmax_axis);
}

status_t dnnl_softmax_v2_backward_desc_init(softmax_v2_desc_t *softmax_v2_desc,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *dst_desc,
        int softmax_axis) {
    return softmax_v2_desc_init(softmax_v2_desc, prop_kind::backward_data,
            alg_kind, nullptr, dst_desc, diff_src_desc, diff_dst_desc,
            softmax_axis);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
