#include "common-ggml.h"

#include <regex>
#include <map>

static const std::map<std::string, enum ggml_dadbed9_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
};

void ggml_dadbed9_print_ftypes(FILE * fp) {
    for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
        fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
    }
}

enum ggml_dadbed9_ftype ggml_dadbed9_parse_ftype(const char * str) {
    enum ggml_dadbed9_ftype ftype;
    if (str[0] == 'q') {
        const auto it = GGML_FTYPE_MAP.find(str);
        if (it == GGML_FTYPE_MAP.end()) {
            fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
            return GGML_FTYPE_UNKNOWN;
        }
        ftype = it->second;
    } else {
        ftype = (enum ggml_dadbed9_ftype) atoi(str);
    }

    return ftype;
}

bool ggml_dadbed9_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_dadbed9_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip) {

    ggml_dadbed9_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };

    if (!ggml_dadbed9_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ggml_dadbed9_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_dadbed9_fp16_t> data_f16;
    std::vector<float>       data_f32;

    std::vector<int64_t> hist_all(1 << 4, 0);

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read (&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_dadbed9_type_name((ggml_dadbed9_type) ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_dadbed9_type_name((ggml_dadbed9_type) ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_dadbed9_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_dadbed9_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        } else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements*bpe);
            finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            switch ((ggml_dadbed9_type) ttype) {
                case GGML_TYPE_Q4_0:
                    {
                        cur_size = ggml_dadbed9_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q4_1:
                    {
                        cur_size = ggml_dadbed9_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q5_0:
                    {
                        cur_size = ggml_dadbed9_quantize_q5_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q5_1:
                    {
                        cur_size = ggml_dadbed9_quantize_q5_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q8_0:
                    {
                        cur_size = ggml_dadbed9_quantize_q8_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_dadbed9_type_name((ggml_dadbed9_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                hist_all[i] += hist_cur[i];
            }

            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                printf("%5.3f ", hist_cur[i] / (float)nelements);
            }
            printf("\n");
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ggml_dadbed9_type_name(qtype));

    {
        int64_t sum_all = 0;
        for (int i = 0; i < (int) hist_all.size(); ++i) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (int i = 0; i < (int) hist_all.size(); ++i) {
            printf("%5.3f ", hist_all[i] / (float)sum_all);
        }
        printf("\n");
    }

    return true;
}
