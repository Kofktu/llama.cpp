// Buffer-based GGUF reader and parser
// This file is included from gguf.cpp to access internal types (gguf_kv, gguf_tensor_info, gguf_context).
// Separated for easier upstream merge management.

struct gguf_reader_buffer {

    const uint8_t *data;
    size_t size;
    size_t offset;

    gguf_reader_buffer(const void *buffer, const size_t size)
        : data((const uint8_t *)buffer), size(size), offset(0) {}

    template <typename T>
    bool read(T & dst) {
        return read(&dst, sizeof(T));
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) {
        dst.resize(n);

        if constexpr (std::is_same<T, bool>::value) {
            for (size_t i = 0; i < n; i ++) {
                bool tmp;
                if (!read(tmp)) {
                    return false;
                }
                dst[i] = tmp;
            }
            return true;
        } else {
            size_t total_bytes = n * sizeof(T);

            if (offset + total_bytes > size) {
                return false;
            }

            memcpy(dst.data(), data + offset, total_bytes);

            offset += total_bytes;
            return true;
        }
    }

    bool read(bool & dst) {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum ggml_type & dst) {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = ggml_type(tmp);
        return true;
    }

    bool read(enum gguf_type & dst) {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = gguf_type(tmp);
        return true;
    }

    bool read(std::string & dst) {
        uint64_t str_len = 0;

        if (!read(&str_len, sizeof(str_len))) {
            return false;
        }

        dst.resize(str_len);
        return read(dst.data(), str_len);
    }

    bool read(void * dst, const size_t n) {
        if (offset + n > size) {
            return false;
        }

        memcpy(dst, data + offset, n);
        offset += n;

        return true;
    }

    size_t tell() const {
        return offset;
    }

    bool seek(size_t pos) {
        if (pos > size) {
            return false;
        }
        offset = pos;
        return true;
    }

};

// batch load for scalar types using fast memcpy
template <typename T>
static void load_gguf_data_batch(std::vector<T> & dst, const uint8_t * src, size_t n) {
    memcpy(dst.data(), src, n * sizeof(T));
}

// batch load specialization for bool (std::vector<bool> has no .data())
static void load_gguf_data_batch(std::vector<bool> & dst, const uint8_t * src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (src[i] != 0);
    }
}

template<typename T>
bool gguf_read_buffer_emplace_helper(struct gguf_reader_buffer & gr, std::vector<struct gguf_kv> & kv, const std::string & key, const bool is_array, const size_t n) {
    // Array path: optimized scan-then-copy to avoid memory spikes on large arrays (e.g. tokenizer)
    if (is_array) {
        const uint8_t *start_ptr = gr.data + gr.offset;
        const uint8_t *curr_ptr = start_ptr;
        const uint8_t *end_ptr = gr.data + gr.size;

        if constexpr (std::is_same_v<T, std::string>) {
            std::vector<std::string> values;
            values.reserve(n);

            for (size_t i = 0; i < n; ++ i) {
                uint64_t len;
                if (curr_ptr + sizeof(len) > end_ptr) return false;

                memcpy(&len, curr_ptr, sizeof(len));
                curr_ptr += sizeof(len);

                if (curr_ptr + len > end_ptr) return false;
                values.emplace_back((const char*)curr_ptr, len);
                curr_ptr += len;
            }

            kv.emplace_back(key, values);
        } else {
            size_t total_bytes = n * sizeof(T);
            if (curr_ptr + total_bytes > end_ptr) return false;

            std::vector<T> values(n);

            load_gguf_data_batch(values, curr_ptr, n);

            curr_ptr += total_bytes;
            kv.emplace_back(key, values);
        }

        gr.offset += (curr_ptr - start_ptr);
    } else {
        T value;
        if (!gr.read(value)) {
            return false;
        }
        kv.emplace_back(key, value);
    }

    return true;
}

struct gguf_context * gguf_init_from_buffer(const void * buffer, size_t size, struct gguf_init_params params) {
    struct gguf_reader_buffer gr(buffer, size);
    struct gguf_context * ctx = new gguf_context;

    bool ok = true;

    // file magic
    {
        std::vector<char> magic;
        ok = ok && gr.read(magic, 4);

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                char c0 = isprint(magic[0]) ? magic[0] : '?';
                char c1 = isprint(magic[1]) ? magic[1] : '?';
                char c2 = isprint(magic[2]) ? magic[2] : '?';
                char c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                return nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    if (ok && gr.read(ctx->version)) {
        if (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = false;
        }

        if (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = false;
        }

        if (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        if (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_tensors)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/sizeof(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/sizeof(gguf_tensor_info));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_kv)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_kv < 0 || n_kv > int64_t(SIZE_MAX/sizeof(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/sizeof(gguf_kv));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            try {
                ok = ok && gr.read(key);
            } catch (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                if (key == ctx->kv[j].key) {
                    GGML_LOG_ERROR("%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                    ok = false;
                }
            }

            if (!ok) {
                break;
            }

            ok = ok && gr.read(type);
            if (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }

            if (!ok) {
                break;
            }

            switch (type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_read_buffer_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_read_buffer_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_read_buffer_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_read_buffer_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_read_buffer_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_read_buffer_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_read_buffer_emplace_helper<float>      (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_read_buffer_emplace_helper<bool>       (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_read_buffer_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_read_buffer_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_read_buffer_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_read_buffer_emplace_helper<double>     (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_ARRAY:
                default:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        ok = false;
                    } break;
            }
        }

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        const int alignment_idx = gguf_find_key(ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        if (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            return nullptr;
        }
    }

    // read the tensor info
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        struct gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            try {
                ok = ok && gr.read(name);
            } catch (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            if (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                break;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                    GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.read(n_dims);
            if (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                break;
            }
            for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                if (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                if (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    break;
                }
            }

            if (ok && ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                       (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                       (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2]))) {

                GGML_LOG_ERROR("%s: total number of elements in tensor '%s' with shape "
                    "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d (%s)\n",
                    __func__, info.t.name, info.t.type, ggml_type_name(info.t.type));
                ok = false;
                break;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            if (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = false;
                break;
            }

            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }

    if (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    size_t current_pos = gr.tell();
    size_t aligned_pos = GGML_PAD(current_pos, ctx->alignment);

    if (!gr.seek(aligned_pos)) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    ctx->offset = aligned_pos;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            size_t padded_size = GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
            if (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                return nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    if (params.ctx != nullptr) {
        const size_t mem_size =
            params.no_alloc ?
            (n_tensors    )*ggml_tensor_overhead() :
            (n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        if (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = nullptr;

        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            if (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            ok = ok && gr.read(data->data, ctx->size);

            if (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                return nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const struct gguf_tensor_info & info = ctx->info[i];

            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            if (!ok) {
                break;
            }

            ggml_set_name(cur, info.t.name);

            if (!params.no_alloc) {
                cur->data = (char *) data->data + info.offset;
            }
        }

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            return nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    return ctx;
}
