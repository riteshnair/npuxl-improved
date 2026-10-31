/*
 * vnpu.h - Virtual NPU public C ABI (v1)
 *
 * A standalone, dependency-free software NPU for CPU-only machines.  It is a
 * *virtual device*: command streams, device memory, completion events, status
 * registers, DMA, telemetry and multiple heterogeneous compute-unit styles.
 * No PCem, no SDL, no graphics stack, no BLAS and no model runtime is needed.
 *
 * ABI rules:
 *   - All public structs are plain-old-data and versioned by abi_version.
 *   - Pointers marked const are read-only inputs.
 *   - All functions return vnpu_status_t unless documented otherwise.
 *   - Calls are thread safe on distinct streams; a single stream is ordered.
 *   - New capabilities are added via opcodes/flags, not by changing v1 structs.
 */
#ifndef VNPU_H
#define VNPU_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
#  if defined(VNPU_BUILD_SHARED)
#    define VNPU_API __declspec(dllexport)
#  elif defined(VNPU_USE_SHARED)
#    define VNPU_API __declspec(dllimport)
#  else
#    define VNPU_API
#  endif
#  define VNPU_CALL __cdecl
#else
#  if defined(__GNUC__) || defined(__clang__)
#    define VNPU_API __attribute__((visibility("default")))
#  else
#    define VNPU_API
#  endif
#  define VNPU_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VNPU_ABI_VERSION        1u
#define VNPU_VERSION_MAJOR      1u
#define VNPU_VERSION_MINOR      0u
#define VNPU_VERSION_PATCH      0u
#define VNPU_MAX_DIMS           4u
#define VNPU_MAX_TOKENS         64u
#define VNPU_CACHELINE_BYTES    64u
#define VNPU_LOCAL_STORE_BYTES  (256u * 1024u)

/* Virtual MMIO register offsets. */
#define VNPU_REG_ID            0x00u
#define VNPU_REG_VERSION       0x04u
#define VNPU_REG_CONTROL       0x08u
#define VNPU_REG_STATUS        0x0cu
#define VNPU_REG_ERROR         0x10u
#define VNPU_REG_CMD_HEAD      0x14u
#define VNPU_REG_CMD_TAIL      0x18u
#define VNPU_REG_DOORBELL      0x1cu
#define VNPU_REG_IRQ_STATUS    0x20u
#define VNPU_REG_IRQ_MASK      0x24u
#define VNPU_REG_ACTIVE_CUS    0x28u
#define VNPU_REG_THERMAL       0x2cu
#define VNPU_REG_CLOCK_MHZ     0x30u
#define VNPU_REG_FEATURE_LO    0x34u
#define VNPU_REG_FEATURE_HI    0x38u

#define VNPU_REG_COUNT         16u
#define VNPU_MAGIC             0x56504e55u /* "VPNU" */

/* Feature bits returned in vnpu_device_info_t.features. */
#define VNPU_FEATURE_SCALAR       (1ull << 0)
#define VNPU_FEATURE_SSE2         (1ull << 1)
#define VNPU_FEATURE_AVX2_FMA     (1ull << 2)
#define VNPU_FEATURE_AVX512F      (1ull << 3)
#define VNPU_FEATURE_AVX512VNNI   (1ull << 4)
#define VNPU_FEATURE_NEON         (1ull << 5)
#define VNPU_FEATURE_BF16         (1ull << 6)
#define VNPU_FEATURE_THREADS      (1ull << 7)
#define VNPU_FEATURE_CLUSTER      (1ull << 8)
#define VNPU_FEATURE_REMOTE       (1ull << 9)
#define VNPU_FEATURE_TRACE        (1ull << 10)
#define VNPU_FEATURE_GPU_FUTURE   (1ull << 31)

/* Configuration flags. */
#define VNPU_CONFIG_TRACE          (1u << 0)
#define VNPU_CONFIG_TRACE_CHROME   (1u << 1)
#define VNPU_CONFIG_PIN_THREADS    (1u << 2)
#define VNPU_CONFIG_NUMA_AWARE     (1u << 3)
#define VNPU_CONFIG_EAGER_SYNC     (1u << 4)
#define VNPU_CONFIG_LOW_LATENCY    (1u << 5)
#define VNPU_CONFIG_HIGH_THROUGHPUT (1u << 6)

/* Command flags. */
#define VNPU_CMD_FLAG_ASYNC        (1u << 0)
#define VNPU_CMD_FLAG_PROFILE      (1u << 1)
#define VNPU_CMD_FLAG_WILDCAT_CULL (1u << 2)
#define VNPU_CMD_FLAG_CELL_DMA     (1u << 3)
#define VNPU_CMD_FLAG_VOODOO_FIFO  (1u << 4)

typedef enum vnpu_status_t {
    VNPU_OK                       = 0,
    VNPU_ERR_INVALID_ARGUMENT     = -1,
    VNPU_ERR_OUT_OF_MEMORY        = -2,
    VNPU_ERR_UNSUPPORTED          = -3,
    VNPU_ERR_INVALID_COMMAND      = -4,
    VNPU_ERR_DEVICE_BUSY          = -5,
    VNPU_ERR_DEVICE_ERROR         = -6,
    VNPU_ERR_TIMEOUT              = -7,
    VNPU_ERR_NETWORK              = -8,
    VNPU_ERR_AUTH                 = -9,
    VNPU_ERR_PROTOCOL             = -10,
    VNPU_ERR_NOT_FOUND            = -11,
    VNPU_ERR_IO                   = -12,
    VNPU_ERR_INTERNAL             = -100
} vnpu_status_t;

typedef enum vnpu_dtype_t {
    VNPU_DTYPE_F32   = 0,
    VNPU_DTYPE_F16   = 1,
    VNPU_DTYPE_BF16  = 2,
    VNPU_DTYPE_I8    = 3,
    VNPU_DTYPE_U8    = 4,
    VNPU_DTYPE_I32   = 5
} vnpu_dtype_t;

/*
 * Compute-unit styles.  These are real scheduling paths, not cosmetic names:
 *   VOODOO  - FIFO command-stream tile pipeline with persistent per-tile bases.
 *   WILDCAT - programmable work-stealing tile scheduler with optional cull mask.
 *   CELL    - SPU-style workers with explicit DMA into per-worker local stores.
 *   GPU     - reserved for the future heterogeneous backend.
 */
typedef enum vnpu_compute_unit_t {
    VNPU_UNIT_AUTO     = 0,
    VNPU_UNIT_VOODOO   = 1,
    VNPU_UNIT_WILDCAT  = 2,
    VNPU_UNIT_CELL     = 3,
    VNPU_UNIT_GPU      = 16
} vnpu_compute_unit_t;

typedef enum vnpu_epilogue_t {
    VNPU_EPILOGUE_NONE       = 0,
    VNPU_EPILOGUE_RELU       = 1,
    VNPU_EPILOGUE_SILU       = 2,
    VNPU_EPILOGUE_GELU_TANH  = 3,
    VNPU_EPILOGUE_GELU_ERF   = 4
} vnpu_epilogue_t;

typedef enum vnpu_quant_axis_t {
    VNPU_QUANT_PER_ROW      = 0,
    VNPU_QUANT_PER_COLUMN   = 1,
    VNPU_QUANT_PER_TENSOR   = 2
} vnpu_quant_axis_t;

typedef enum vnpu_copy_direction_t {
    VNPU_COPY_HOST_TO_HOST   = 0,
    VNPU_COPY_HOST_TO_DEVICE = 1,
    VNPU_COPY_DEVICE_TO_HOST = 2,
    VNPU_COPY_DEVICE_TO_DEVICE = 3
} vnpu_copy_direction_t;

typedef enum vnpu_opcode_t {
    VNPU_OP_NOP                 = 0,
    VNPU_OP_MATMUL_F32          = 1,
    VNPU_OP_MATMUL_I8           = 2,
    VNPU_OP_RMSNORM_F32         = 3,
    VNPU_OP_SOFTMAX_F32         = 4,
    VNPU_OP_ROPE_F32            = 5,
    VNPU_OP_SWIGLU_F32          = 6,
    VNPU_OP_GELU_F32            = 7,
    VNPU_OP_ADD_F32             = 8,
    VNPU_OP_MUL_F32             = 9,
    VNPU_OP_ATTENTION_F32       = 10,
    VNPU_OP_QUANTIZE_I8         = 11,
    VNPU_OP_PACK_WEIGHTS_I8     = 12,
    VNPU_OP_MTP_HEAD_F32        = 13,
    VNPU_OP_COPY                = 14,
    VNPU_OP_MEMSET              = 15,
    VNPU_OP_BARRIER             = 16,
    VNPU_OP_COUNT               = 17
} vnpu_opcode_t;

typedef struct vnpu_config_t {
    uint32_t abi_version;
    uint32_t num_threads;          /* 0 = hardware concurrency */
    uint32_t num_compute_units;    /* 0 = one virtual CU per worker */
    uint32_t tile_m;               /* 0 = autotune */
    uint32_t tile_k;
    uint32_t tile_n;
    uint32_t flags;
    uint32_t reserved0;
    const char *trace_file;        /* NULL = stderr for text dumps */
    const char *node_name;         /* NULL = auto */
} vnpu_config_t;

typedef struct vnpu_device_info_t {
    uint32_t abi_version;
    char name[64];
    char vendor[64];
    char cpu_arch[32];
    char backend[64];
    uint32_t simd_bits;
    uint32_t num_threads;
    uint32_t num_compute_units;
    uint32_t local_store_bytes;
    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;
    /* Physical cores behind num_threads' logical count.  SMT siblings share
     * L1/L2, so the default unit count follows this value. */
    uint32_t physical_cores;
    uint64_t features;
    uint64_t l1d_bytes;
    uint64_t l2_bytes;
    uint64_t l3_bytes;
    uint64_t total_memory_bytes;
} vnpu_device_info_t;

typedef struct vnpu_device_status_t {
    uint64_t commands_submitted;
    uint64_t commands_completed;
    uint64_t kernel_time_ns;
    uint64_t dispatch_time_ns;
    uint64_t memory_time_ns;
    uint64_t sync_time_ns;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t flops;
    uint64_t tiles_dispatched;
    uint64_t tiles_culled;
    uint64_t dma_bytes;
    uint64_t interrupts;
    uint32_t busy;
    uint32_t error;
    uint32_t last_opcode;
    uint32_t active_workers;
    uint64_t reserved[4];
} vnpu_device_status_t;

typedef struct vnpu_trace_summary_t {
    uint64_t events;
    uint64_t total_time_ns;
    uint64_t dispatch_time_ns;
    uint64_t memory_time_ns;
    uint64_t compute_time_ns;
    uint64_t sync_time_ns;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t flops;
    double achieved_gflops;
    double achieved_gbytes_s;
    uint32_t hot_opcode;
    uint64_t hot_time_ns;
    double trace_overhead_ns;
} vnpu_trace_summary_t;

typedef struct vnpu_device vnpu_device_t;
typedef struct vnpu_stream vnpu_stream_t;
typedef struct vnpu_event vnpu_event_t;
typedef struct vnpu_buffer vnpu_buffer_t;
typedef struct vnpu_packed_weights vnpu_packed_weights_t;
typedef struct vnpu_cluster vnpu_cluster_t;
typedef struct vnpu_remote_node vnpu_remote_node_t;
typedef struct vnpu_weight_handle vnpu_weight_handle_t;

typedef void (VNPU_CALL *vnpu_interrupt_cb_t)(vnpu_device_t *device,
                                              uint32_t irq_status,
                                              void *user_data);

/* ------------------------------------------------------------------------- */
/* Command packet - one fixed-size, POD command for the virtual command CPU.  */
/* ------------------------------------------------------------------------- */

typedef struct vnpu_matmul_f32_args_t {
    const float *A;
    const float *B;              /* row-major KxN, or NxK when trans_b */
    const float *bias;           /* optional, length N */
    float *C;                    /* MxN */
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t lda;                /* 0 = K */
    uint32_t ldb;                /* 0 = N or K depending on trans_b */
    uint32_t ldc;                /* 0 = N */
    uint32_t trans_a;            /* 0/1 */
    uint32_t trans_b;            /* 0/1 */
    float alpha;
    float beta;
    uint32_t epilogue;           /* vnpu_epilogue_t */
    uint32_t unit;               /* vnpu_compute_unit_t */
    uint32_t reserved0;
    const uint8_t *tile_mask;    /* optional Wildcat cull mask, 1 byte/tile */
    uint32_t mask_tile_m;
    uint32_t mask_tile_n;
} vnpu_matmul_f32_args_t;

typedef struct vnpu_matmul_i8_args_t {
    const int8_t *A;             /* symmetric signed int8, MxK */
    vnpu_packed_weights_t *B;    /* packed signed int8 weights */
    const float *scale_a;        /* length M, NULL if all 1.0 */
    const float *bias;           /* optional, length N */
    float *C;                    /* MxN */
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t lda;                /* 0 = K */
    uint32_t ldc;                /* 0 = N */
    uint32_t epilogue;
    uint32_t unit;
} vnpu_matmul_i8_args_t;

typedef struct vnpu_rmsnorm_f32_args_t {
    const float *X;
    const float *W;
    float *Y;
    uint32_t rows;
    uint32_t cols;
    float eps;
    uint32_t unit;
} vnpu_rmsnorm_f32_args_t;

typedef struct vnpu_softmax_f32_args_t {
    const float *X;
    float *Y;
    uint32_t rows;
    uint32_t cols;
    float temperature;
    uint32_t unit;
} vnpu_softmax_f32_args_t;

typedef struct vnpu_rope_f32_args_t {
    float *Q;                    /* in-place [batch, heads, q_seq, head_dim] */
    float *K;                    /* in-place [batch, heads, k_seq, head_dim] */
    const int32_t *positions;    /* optional per-token positions */
    uint32_t batch;
    uint32_t heads;
    uint32_t q_seq;
    uint32_t k_seq;
    uint32_t head_dim;
    uint32_t position_offset;
    uint32_t interleaved;
    float theta_base;
    float scaling;
} vnpu_rope_f32_args_t;

typedef struct vnpu_binary_f32_args_t {
    const float *A;
    const float *B;
    float *Y;
    uint64_t count;
    float alpha;
    float beta;
    uint32_t unit;
    uint32_t reserved0;
} vnpu_binary_f32_args_t;

typedef struct vnpu_unary_f32_args_t {
    const float *X;
    float *Y;
    uint64_t count;
    float param;
    uint32_t unit;
    uint32_t reserved0;
} vnpu_unary_f32_args_t;

typedef struct vnpu_swiglu_f32_args_t {
    const float *gate;
    const float *up;
    float *Y;
    uint64_t count;
    float alpha;
    uint32_t unit;
    uint32_t reserved0;
} vnpu_swiglu_f32_args_t;

typedef struct vnpu_attention_f32_args_t {
    const float *Q;              /* [batch, heads, seq_q, head_dim] */
    const float *K;              /* [batch, heads, seq_k, head_dim] */
    const float *V;              /* [batch, heads, seq_k, head_dim] */
    const float *mask;           /* optional additive [batch, heads, seq_q, seq_k] */
    float *O;                    /* [batch, heads, seq_q, head_dim] */
    float *workspace;            /* optional, else allocated internally */
    uint64_t workspace_bytes;
    uint32_t batch;
    uint32_t heads;
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t head_dim;
    uint32_t causal;
    float scale;
    uint32_t unit;
} vnpu_attention_f32_args_t;

typedef struct vnpu_quantize_i8_args_t {
    const float *X;
    int8_t *Y;
    float *scales;
    uint64_t rows;
    uint64_t cols;
    uint32_t axis;
    uint32_t unit;
} vnpu_quantize_i8_args_t;

typedef struct vnpu_pack_weights_i8_args_t {
    const int8_t *B;             /* KxN (trans_b=0) or NxK (trans_b=1) */
    const float *scales;         /* optional, length N; else computed from B */
    vnpu_packed_weights_t **out;
    uint32_t K;
    uint32_t N;
    uint32_t ldb;                /* 0 = N or K depending on trans_b */
    uint32_t trans_b;
    uint32_t unit;
    uint32_t reserved0;
} vnpu_pack_weights_i8_args_t;

typedef struct vnpu_mtp_head_f32_args_t {
    const float *hidden;         /* [gamma, hidden_dim] */
    const float *W;              /* [vocab, hidden_dim] or [hidden_dim, vocab] */
    const float *bias;           /* optional, length vocab */
    float *logits;               /* [gamma, vocab] */
    uint32_t gamma;
    uint32_t hidden_dim;
    uint32_t vocab;
    uint32_t trans_w;
    uint32_t unit;
    uint32_t reserved0;
} vnpu_mtp_head_f32_args_t;

typedef struct vnpu_copy_args_t {
    void *dst;
    const void *src;
    uint64_t bytes;
    uint32_t direction;
    uint32_t reserved0;
} vnpu_copy_args_t;

typedef struct vnpu_memset_args_t {
    void *dst;
    uint64_t bytes;
    uint32_t value;
    uint32_t reserved0;
} vnpu_memset_args_t;

typedef struct vnpu_command_t {
    uint32_t opcode;
    uint32_t flags;
    uint64_t id;
    union {
        vnpu_matmul_f32_args_t matmul_f32;
        vnpu_matmul_i8_args_t matmul_i8;
        vnpu_rmsnorm_f32_args_t rmsnorm;
        vnpu_softmax_f32_args_t softmax;
        vnpu_rope_f32_args_t rope;
        vnpu_binary_f32_args_t binary;
        vnpu_unary_f32_args_t unary;
        vnpu_swiglu_f32_args_t swiglu;
        vnpu_attention_f32_args_t attention;
        vnpu_quantize_i8_args_t quantize_i8;
        vnpu_pack_weights_i8_args_t pack_weights_i8;
        vnpu_mtp_head_f32_args_t mtp_head;
        vnpu_copy_args_t copy;
        vnpu_memset_args_t memset;
        uint8_t raw[192];
    } args;
} vnpu_command_t;

/* ------------------------------------------------------------------------- */
/* Device, stream, event and memory API                                       */
/* ------------------------------------------------------------------------- */

VNPU_API void VNPU_CALL vnpu_config_init(vnpu_config_t *config);
VNPU_API const char * VNPU_CALL vnpu_status_string(vnpu_status_t status);
VNPU_API const char * VNPU_CALL vnpu_version_string(void);
VNPU_API uint32_t VNPU_CALL vnpu_abi_version(void);

VNPU_API vnpu_status_t VNPU_CALL vnpu_device_create(const vnpu_config_t *config,
                                                    vnpu_device_t **out_device);
VNPU_API void VNPU_CALL vnpu_device_destroy(vnpu_device_t *device);
VNPU_API vnpu_status_t VNPU_CALL vnpu_device_get_info(const vnpu_device_t *device,
                                                      vnpu_device_info_t *info);
VNPU_API vnpu_status_t VNPU_CALL vnpu_device_get_status(const vnpu_device_t *device,
                                                        vnpu_device_status_t *status);
VNPU_API vnpu_status_t VNPU_CALL vnpu_device_reset(vnpu_device_t *device);
VNPU_API vnpu_status_t VNPU_CALL vnpu_device_set_tiles(vnpu_device_t *device,
                                                       uint32_t tile_m,
                                                       uint32_t tile_k,
                                                       uint32_t tile_n);
VNPU_API uint32_t VNPU_CALL vnpu_mmio_read32(vnpu_device_t *device, uint32_t offset);
VNPU_API void VNPU_CALL vnpu_mmio_write32(vnpu_device_t *device, uint32_t offset,
                                          uint32_t value);
VNPU_API void VNPU_CALL vnpu_device_set_interrupt_callback(vnpu_device_t *device,
                                                           vnpu_interrupt_cb_t cb,
                                                           void *user_data);

VNPU_API vnpu_status_t VNPU_CALL vnpu_stream_create(vnpu_device_t *device,
                                                    int priority,
                                                    vnpu_stream_t **out_stream);
VNPU_API void VNPU_CALL vnpu_stream_destroy(vnpu_stream_t *stream);
VNPU_API vnpu_status_t VNPU_CALL vnpu_stream_sync(vnpu_stream_t *stream);
VNPU_API vnpu_status_t VNPU_CALL vnpu_stream_submit(vnpu_stream_t *stream,
                                                    const vnpu_command_t *commands,
                                                    uint64_t count,
                                                    vnpu_event_t *event);

VNPU_API vnpu_status_t VNPU_CALL vnpu_event_create(vnpu_event_t **out_event);
VNPU_API void VNPU_CALL vnpu_event_destroy(vnpu_event_t *event);
VNPU_API vnpu_status_t VNPU_CALL vnpu_event_wait(vnpu_event_t *event,
                                                 uint32_t timeout_ms);
VNPU_API vnpu_status_t VNPU_CALL vnpu_event_query(const vnpu_event_t *event);

VNPU_API vnpu_status_t VNPU_CALL vnpu_buffer_alloc(vnpu_device_t *device,
                                                   uint64_t bytes,
                                                   vnpu_buffer_t **out_buffer);
VNPU_API void VNPU_CALL vnpu_buffer_free(vnpu_buffer_t *buffer);
VNPU_API void * VNPU_CALL vnpu_buffer_host_ptr(vnpu_buffer_t *buffer);
VNPU_API uint64_t VNPU_CALL vnpu_buffer_size(const vnpu_buffer_t *buffer);
VNPU_API vnpu_status_t VNPU_CALL vnpu_buffer_copy(vnpu_stream_t *stream,
                                                  void *dst,
                                                  const void *src,
                                                  uint64_t bytes,
                                                  uint32_t direction);
VNPU_API vnpu_status_t VNPU_CALL vnpu_buffer_memset(vnpu_stream_t *stream,
                                                    void *dst,
                                                    uint32_t value,
                                                    uint64_t bytes);

/* Packed INT8 weights stay resident on a node and are reused across calls. */
VNPU_API vnpu_status_t VNPU_CALL vnpu_pack_weights_i8(
    vnpu_device_t *device,
    const int8_t *B,
    const float *scales,
    uint32_t K,
    uint32_t N,
    uint32_t ldb,
    uint32_t trans_b,
    vnpu_packed_weights_t **out_weights);
VNPU_API void VNPU_CALL vnpu_packed_weights_destroy(vnpu_packed_weights_t *weights);
VNPU_API vnpu_status_t VNPU_CALL vnpu_packed_weights_info(
    const vnpu_packed_weights_t *weights,
    uint32_t *K,
    uint32_t *N,
    uint64_t *bytes,
    const float **scales,
    const int32_t **column_sums);

/* ------------------------------------------------------------------------- */
/* Synchronous convenience API - thin wrappers over the command stream.       */
/* ------------------------------------------------------------------------- */

VNPU_API vnpu_status_t VNPU_CALL vnpu_matmul_f32(vnpu_stream_t *stream,
                                                 const vnpu_matmul_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_matmul_i8(vnpu_stream_t *stream,
                                                const vnpu_matmul_i8_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_quantize_i8(vnpu_stream_t *stream,
                                                  const vnpu_quantize_i8_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_rmsnorm_f32(vnpu_stream_t *stream,
                                                  const vnpu_rmsnorm_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_softmax_f32(vnpu_stream_t *stream,
                                                  const vnpu_softmax_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_rope_f32(vnpu_stream_t *stream,
                                               const vnpu_rope_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_swiglu_f32(vnpu_stream_t *stream,
                                                 const vnpu_swiglu_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_gelu_f32(vnpu_stream_t *stream,
                                               const vnpu_unary_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_add_f32(vnpu_stream_t *stream,
                                              const vnpu_binary_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_mul_f32(vnpu_stream_t *stream,
                                              const vnpu_binary_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_attention_f32(vnpu_stream_t *stream,
                                                    const vnpu_attention_f32_args_t *args);
VNPU_API vnpu_status_t VNPU_CALL vnpu_mtp_head_f32(vnpu_stream_t *stream,
                                                   const vnpu_mtp_head_f32_args_t *args);

/* ------------------------------------------------------------------------- */
/* Sherlock tracing and autotuning                                            */
/* ------------------------------------------------------------------------- */

VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_enable(vnpu_device_t *device, int enable);
VNPU_API void VNPU_CALL vnpu_trace_reset(vnpu_device_t *device);
VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_get_summary(const vnpu_device_t *device,
                                                        vnpu_trace_summary_t *summary);
VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_dump(const vnpu_device_t *device,
                                                 const char *path);
VNPU_API vnpu_status_t VNPU_CALL vnpu_trace_write_chrome(const vnpu_device_t *device,
                                                         const char *path);

typedef struct vnpu_tune_result_t {
    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;
    uint32_t unit;
    uint32_t threads;
    double best_ms;
    double baseline_ms;
    double speedup;
    double gflops;
    uint64_t bytes_read;
    uint64_t bytes_written;
} vnpu_tune_result_t;

VNPU_API vnpu_status_t VNPU_CALL vnpu_autotune_gemm(vnpu_device_t *device,
                                                    uint32_t M,
                                                    uint32_t K,
                                                    uint32_t N,
                                                    uint32_t iterations,
                                                    vnpu_dtype_t dtype,
                                                    vnpu_tune_result_t *result);

/* ------------------------------------------------------------------------- */
/* Local cluster, remote nodes and speculative decoding                       */
/* ------------------------------------------------------------------------- */

typedef struct vnpu_cluster_config_t {
    uint32_t abi_version;
    uint32_t num_streams;
    uint32_t flags;
    uint32_t request_timeout_ms;
    const char *auth_token;
    const char *cluster_name;
} vnpu_cluster_config_t;

typedef struct vnpu_remote_node_config_t {
    const char *host;
    uint16_t port;
    const char *auth_token;
    uint32_t weight;
    uint32_t flags;
} vnpu_remote_node_config_t;

typedef struct vnpu_cluster_stats_t {
    uint64_t local_jobs;
    uint64_t remote_jobs;
    uint64_t total_jobs;
    uint64_t network_bytes_sent;
    uint64_t network_bytes_received;
    uint64_t failed_jobs;
    uint64_t culled_tiles;
    double local_time_ns;
    double remote_time_ns;
    double speedup_estimate;
} vnpu_cluster_stats_t;

VNPU_API void VNPU_CALL vnpu_cluster_config_init(vnpu_cluster_config_t *config);
VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_create(const vnpu_cluster_config_t *config,
                                                     vnpu_cluster_t **out_cluster);
VNPU_API void VNPU_CALL vnpu_cluster_destroy(vnpu_cluster_t *cluster);
VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_add_device(vnpu_cluster_t *cluster,
                                                         vnpu_device_t *device,
                                                         uint32_t weight);
VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_add_remote(vnpu_cluster_t *cluster,
                                                         const vnpu_remote_node_config_t *config,
                                                         vnpu_remote_node_t **out_node);
VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_get_stats(const vnpu_cluster_t *cluster,
                                                        vnpu_cluster_stats_t *stats);

/* Cache a weight matrix on every worker.  Handle is owned by the cluster. */
VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_cache_weights_f32(
    vnpu_cluster_t *cluster,
    const char *key,
    const float *B,
    uint32_t K,
    uint32_t N,
    uint32_t trans_b,
    vnpu_weight_handle_t **out_handle);

/* Split GEMM rows across local devices and remote nodes. */
VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_matmul_f32(
    vnpu_cluster_t *cluster,
    vnpu_weight_handle_t *weights,
    const float *A,
    float *C,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t unit);

/* Node server: never returns until vnpu_node_stop() is called from a signal
 * handler or another thread.  embed=1 is for library embedding. */
typedef struct vnpu_node_config_t {
    uint32_t abi_version;
    uint16_t port;
    uint16_t reserved0;
    uint32_t num_threads;
    uint32_t flags;
    const char *bind_address;
    const char *auth_token;
    const char *node_name;
} vnpu_node_config_t;

VNPU_API void VNPU_CALL vnpu_node_config_init(vnpu_node_config_t *config);
VNPU_API vnpu_status_t VNPU_CALL vnpu_node_run(const vnpu_node_config_t *config);
VNPU_API void VNPU_CALL vnpu_node_stop(void);

/* Generic DFlash/MTP-style greedy speculative decoding.  Both callbacks fill
 * a row-major [n_positions, vocab] logits matrix.  The verifier may live on a
 * different vnpu-node, and the draft head may run on the local cluster. */
typedef vnpu_status_t (VNPU_CALL *vnpu_logits_fn_t)(
    void *user_data,
    const int32_t *tokens,
    uint32_t num_tokens,
    uint32_t first_position,
    uint32_t num_positions,
    float *logits,
    uint32_t vocab);

typedef struct vnpu_speculative_config_t {
    uint32_t abi_version;
    uint32_t gamma;             /* number of draft tokens per verification */
    uint32_t vocab;
    uint32_t max_tokens;
    uint32_t eos_token;
    uint32_t flags;
    float temperature;          /* 0 = greedy; >0 enables stochastic sampling */
    uint32_t seed;
} vnpu_speculative_config_t;

typedef struct vnpu_speculative_result_t {
    uint32_t accepted;
    uint32_t produced;
    uint32_t eos;
    int32_t tokens[VNPU_MAX_TOKENS];
    double draft_time_ms;
    double verify_time_ms;
    double total_time_ms;
    double tokens_per_second;
    double theoretical_speedup;
} vnpu_speculative_result_t;

VNPU_API void VNPU_CALL vnpu_speculative_config_init(vnpu_speculative_config_t *config);
VNPU_API vnpu_status_t VNPU_CALL vnpu_speculative_step(
    vnpu_cluster_t *cluster,
    const vnpu_speculative_config_t *config,
    const int32_t *context,
    uint32_t context_len,
    vnpu_logits_fn_t draft_fn,
    void *draft_user,
    vnpu_logits_fn_t verify_fn,
    void *verify_user,
    vnpu_speculative_result_t *result);

/* Misc helpers used by engine adapters. */
VNPU_API uint16_t VNPU_CALL vnpu_float_to_bf16(float value);
VNPU_API uint16_t VNPU_CALL vnpu_float_to_f16(float value);
VNPU_API float VNPU_CALL vnpu_bf16_to_float(uint16_t value);
VNPU_API float VNPU_CALL vnpu_f16_to_float(uint16_t value);
VNPU_API const char * VNPU_CALL vnpu_opcode_name(uint32_t opcode);
VNPU_API const char * VNPU_CALL vnpu_unit_name(uint32_t unit);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VNPU_H */
