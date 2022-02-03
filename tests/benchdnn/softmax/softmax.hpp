/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/perf_report.hpp"

namespace softmax {

enum alg_t {
    UNDEF,
    SOFTMAX,
    LOGSOFTMAX,
    softmax_accurate = SOFTMAX,
    softmax_log = LOGSOFTMAX,
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_dims_t prb_dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> sdt {dnnl_f32}, ddt {dnnl_f32};
    std::vector<std::string> stag {tag::abx}, dtag {tag::any};
    std::vector<alg_t> alg {SOFTMAX};
    std::vector<int> axis {1};
    std::vector<int64_t> mb {0};
    std::vector<bool> inplace {false};
    std::vector<attr_t::scale_t> oscale {attr_t::scale_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%dir%,%sdt%,%ddt%,%stag%,%dtag%,%alg%,%"
              "axis%,%DESC%,%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_dims_t {
    prb_t(const prb_dims_t &prb_dims, dir_t dir, dnnl_data_type_t sdt,
            dnnl_data_type_t ddt, const std::string &stag,
            const std::string &dtag, alg_t alg, int axis, bool inplace,
            const attr_t &attr, int64_t mb = 0)
        : prb_dims_t(prb_dims)
        , dir(dir)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , alg(alg)
        , axis(axis)
        , inplace(inplace)
        , attr(attr)
        , user_mb(mb)
        , scales(NULL) {
        if (mb) dims[0] = mb;
        generate_oscales();
    }
    ~prb_t() {
        if (scales) zfree(scales);
    }

    dir_t dir;
    dnnl_data_type_t sdt, ddt;
    std::string stag, dtag;
    alg_t alg;
    int axis;
    bool inplace;
    attr_t attr;
    int64_t user_mb;

    float *scales;
    void generate_oscales();
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , sdt_({p_->sdt})
        , ddt_(p_->ddt)
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const int *axis() const override { return &p_->axis; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &ddt_; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<dnnl_data_type_t> sdt_;
    dnnl_data_type_t ddt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

inline void map_off_to_mb_ic(
        const prb_t *prb, int64_t off, int64_t &mb, int64_t &ic) {
    for (int i = prb->ndims - 1; i > 1; i--)
        off /= prb->dims[i];

    ic = off % prb->dims[1];
    off /= prb->dims[1];
    mb = off % prb->dims[0];
    off /= prb->dims[0];
    assert(off == 0);
}

inline void get_sizes(const prb_t *prb, int64_t &outer_size,
        int64_t &inner_size, int64_t &axis_size) {
    outer_size = inner_size = axis_size = 1;
    for (int i = 0; i < prb->axis; i++)
        outer_size *= prb->dims[i];
    for (int i = prb->axis + 1; i < prb->ndims; i++)
        inner_size *= prb->dims[i];
    axis_size = prb->dims[prb->axis];
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &dst,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace softmax

#endif
