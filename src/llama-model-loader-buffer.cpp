// Buffer-based model loader constructor
// Separated for easier upstream merge management.

#include "llama-model-loader.h"

#include "llama-impl.h"

#include "gguf.h"

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

llama_model_loader::llama_model_loader(
        const void * buffer,
        size_t size,
        bool check_tensors,
        bool no_alloc,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc*/ true,
        /*.ctx*/ &ctx
    };

    meta.reset(gguf_init_from_buffer(buffer, size, params));
    if (!meta) {
        throw std::runtime_error("failed to load model from buffer");
    }

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));

    contexts.emplace_back(ctx);

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string tensor_name = std::string(cur->name);
        if (weights_map.find(tensor_name) != weights_map.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
        }

        n_elements += ggml_nelements(cur);
        n_bytes    += ggml_nbytes(cur);
        weights_map.emplace(tensor_name, llama_tensor_weight(meta.get(), cur, size));
    }

    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    if (n_split > 1) {
        throw std::runtime_error("split model is not supported");
    }

    n_kv      = gguf_get_n_kv(meta.get());
    n_tensors = weights_map.size();

    fver = (enum llama_fver) gguf_get_version(meta.get());

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors\n",
            __func__, n_kv, n_tensors);

    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w = it.second;
            const ggml_tensor * tensor = w.tensor;

            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__,
                        sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(meta.get(), i);
            const enum gguf_type type   = gguf_get_kv_type(meta.get(), i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta.get(), i)), gguf_get_arr_n(meta.get(), i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(meta.get(), i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    this->use_mmap = false;
    this->use_direct_io = false;
    this->check_tensors = check_tensors;
    this->no_alloc = no_alloc;
    this->buffer_addr = buffer;
    this->buffer_size = size;
}
