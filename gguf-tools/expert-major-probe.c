#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

enum {
    PROBE_HEADER_BYTES = 4096,
    PROBE_GUARD_BYTES = 64,
    PROBE_SELECTED = 6,
    PROBE_MAX_TASKS = PROBE_SELECTED * 3,
};

static const uint8_t PROBE_MAGIC[8] = {'D', 'S', '4', 'P', 'R', 'B', '0', '1'};
static volatile sig_atomic_t g_interrupted;

typedef enum {
    PROBE_COMMAND_BUILD,
    PROBE_COMMAND_VERIFY,
    PROBE_COMMAND_REPLAY,
} probe_command;

typedef struct {
    probe_command command;
    const char *model_path;
    const char *sidecar_path;
    const char *trace_path;
    const char *csv_path;
    const char *cache_mode;
    const char *order;
    uint32_t layer;
    uint32_t experts;
    uint32_t threads;
    uint64_t min_request;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_bytes;
    uint64_t down_bytes;
    bool layer_set;
    bool experts_set;
} probe_options;

typedef struct {
    uint64_t record_bytes;
    uint64_t payload_bytes;
    uint64_t file_bytes;
} probe_layout;

typedef struct {
    uint32_t experts[PROBE_SELECTED];
    uint32_t missing[PROBE_SELECTED];
    uint32_t n_missing;
} probe_trace_row;

typedef struct {
    probe_trace_row *rows;
    size_t len;
    size_t cap;
    uint64_t misses;
} probe_trace;

typedef struct {
    int fd;
    uint64_t offset;
    uint64_t len;
    uint8_t *dst;
    bool ok;
} probe_io_task;

typedef struct probe_pool probe_pool;

typedef struct {
    probe_pool *pool;
    uint32_t index;
} probe_worker_arg;

struct probe_pool {
    pthread_t *threads;
    probe_worker_arg *args;
    uint32_t n_threads;
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    bool stop;
    bool failed;
    uint64_t generation;
    probe_io_task *tasks;
    uint32_t n_tasks;
    uint32_t n_active;
    uint32_t active_remaining;
};

typedef struct {
    char mode;
    uint64_t rows;
    uint64_t miss_rows;
    uint64_t misses;
    uint64_t tasks;
    uint64_t logical_bytes;
    uint64_t advice_calls;
    double advice_ms;
    double pread_ms;
    double total_ms;
    double p50_ms;
    double p95_ms;
    uint64_t hash;
} probe_run_result;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage:\n"
            "  expert-major-probe build|verify --model FILE --sidecar FILE --layer N\n"
            "      --experts N --gate-offset N --up-offset N --down-offset N\n"
            "      --gate-bytes N --down-bytes N\n"
            "  expert-major-probe replay --model FILE --sidecar FILE --trace FILE --layer N\n"
            "      --min-request N --experts N --gate-offset N --up-offset N --down-offset N\n"
            "      --gate-bytes N --down-bytes N [--threads N]\n"
            "      [--cache-mode normal|nocache]\n"
            "      [--order AB|BA|ABBA|BAAB|AC|CA|ACCA|CAAC] [--csv FILE]\n"
            "  replay modes: A=model 3-read, B=sidecar 1-read, C=sidecar 2-read\n");
}

static void die(const char *message) {
    fprintf(stderr, "expert-major-probe: %s\n", message);
    exit(1);
}

static void die_errno(const char *operation, const char *path) {
    fprintf(stderr,
            "expert-major-probe: %s %s: %s\n",
            operation,
            path ? path : "",
            strerror(errno));
    exit(1);
}

static void *xmalloc(size_t size) {
    void *p = malloc(size ? size : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *old, size_t size) {
    void *p = realloc(old, size ? size : 1);
    if (!p) die("out of memory");
    return p;
}

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool parse_u64(const char *s, uint64_t *out) {
    if (!s || !s[0] || s[0] == '-') return false;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0') return false;
    *out = (uint64_t)value;
    return true;
}

static const char *need_value(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        fprintf(stderr,
                "expert-major-probe: %s requires a value\n",
                argv[*index]);
        exit(2);
    }
    return argv[++*index];
}

static probe_options parse_options(int argc, char **argv) {
    if (argc < 2) {
        usage(stderr);
        exit(2);
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(stdout);
        exit(0);
    }
    probe_options opt = {
        .threads = 9,
        .cache_mode = "normal",
        .order = "ABBA",
    };
    if (!strcmp(argv[1], "build")) opt.command = PROBE_COMMAND_BUILD;
    else if (!strcmp(argv[1], "verify")) opt.command = PROBE_COMMAND_VERIFY;
    else if (!strcmp(argv[1], "replay")) opt.command = PROBE_COMMAND_REPLAY;
    else {
        fprintf(stderr, "expert-major-probe: unknown command: %s\n", argv[1]);
        usage(stderr);
        exit(2);
    }

    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        }
        if (!strcmp(arg, "--model")) opt.model_path = need_value(&i, argc, argv);
        else if (!strcmp(arg, "--sidecar")) opt.sidecar_path = need_value(&i, argc, argv);
        else if (!strcmp(arg, "--trace")) opt.trace_path = need_value(&i, argc, argv);
        else if (!strcmp(arg, "--csv")) opt.csv_path = need_value(&i, argc, argv);
        else if (!strcmp(arg, "--cache-mode")) opt.cache_mode = need_value(&i, argc, argv);
        else if (!strcmp(arg, "--order")) opt.order = need_value(&i, argc, argv);
        else {
            uint64_t value = 0;
            if (strcmp(arg, "--layer") && strcmp(arg, "--experts") &&
                strcmp(arg, "--min-request") && strcmp(arg, "--threads") &&
                strcmp(arg, "--gate-offset") && strcmp(arg, "--up-offset") &&
                strcmp(arg, "--down-offset") && strcmp(arg, "--gate-bytes") &&
                strcmp(arg, "--down-bytes")) {
                fprintf(stderr, "expert-major-probe: unknown option: %s\n", arg);
                usage(stderr);
                exit(2);
            }
            const char *text = need_value(&i, argc, argv);
            if (!parse_u64(text, &value)) {
                fprintf(stderr,
                        "expert-major-probe: invalid value for %s: %s\n",
                        arg,
                        text);
                exit(2);
            }
            if (!strcmp(arg, "--layer")) {
                if (value > UINT32_MAX) goto range_error;
                opt.layer = (uint32_t)value;
                opt.layer_set = true;
            } else if (!strcmp(arg, "--experts")) {
                if (value == 0 || value > UINT32_MAX) goto range_error;
                opt.experts = (uint32_t)value;
                opt.experts_set = true;
            } else if (!strcmp(arg, "--min-request")) {
                opt.min_request = value;
            } else if (!strcmp(arg, "--threads")) {
                if (value == 0 || value > 64) goto range_error;
                opt.threads = (uint32_t)value;
            } else if (!strcmp(arg, "--gate-offset")) opt.gate_offset = value;
            else if (!strcmp(arg, "--up-offset")) opt.up_offset = value;
            else if (!strcmp(arg, "--down-offset")) opt.down_offset = value;
            else if (!strcmp(arg, "--gate-bytes")) opt.gate_bytes = value;
            else opt.down_bytes = value;
            continue;
range_error:
            fprintf(stderr, "expert-major-probe: value out of range for %s\n", arg);
            exit(2);
        }
    }
    return opt;
}

static probe_layout validate_options(const probe_options *opt, uint64_t model_size) {
    if (!opt->model_path || !opt->sidecar_path || !opt->layer_set ||
        !opt->experts_set || opt->gate_bytes == 0 || opt->down_bytes == 0) {
        usage(stderr);
        die("missing required layout option");
    }
    if (opt->command == PROBE_COMMAND_REPLAY && !opt->trace_path) {
        die("replay requires --trace");
    }
    if (strcmp(opt->cache_mode, "normal") && strcmp(opt->cache_mode, "nocache")) {
        die("--cache-mode must be normal or nocache");
    }
    if (strcmp(opt->order, "AB") && strcmp(opt->order, "BA") &&
        strcmp(opt->order, "ABBA") && strcmp(opt->order, "BAAB") &&
        strcmp(opt->order, "AC") && strcmp(opt->order, "CA") &&
        strcmp(opt->order, "ACCA") && strcmp(opt->order, "CAAC")) {
        die("unsupported --order");
    }

    probe_layout layout = {0};
    uint64_t gate_up = 0;
    if (!checked_mul_u64(opt->gate_bytes, 2, &gate_up) ||
        !checked_add_u64(gate_up, opt->down_bytes, &layout.record_bytes) ||
        !checked_mul_u64(layout.record_bytes, opt->experts, &layout.payload_bytes) ||
        !checked_add_u64(PROBE_HEADER_BYTES,
                         layout.payload_bytes,
                         &layout.file_bytes)) {
        die("layout size overflow");
    }
    if (layout.record_bytes > SIZE_MAX ||
        layout.record_bytes > (uint64_t)LLONG_MAX) {
        die("record is too large for this process");
    }
    uint64_t guarded_record = 0;
    uint64_t selected_payload = 0;
    uint64_t guarded_selected = 0;
    if (!checked_add_u64(layout.record_bytes,
                         2u * PROBE_GUARD_BYTES,
                         &guarded_record) ||
        !checked_mul_u64(layout.record_bytes,
                         PROBE_SELECTED,
                         &selected_payload) ||
        !checked_add_u64(selected_payload,
                         2u * PROBE_GUARD_BYTES,
                         &guarded_selected) ||
        guarded_record > SIZE_MAX || guarded_selected > SIZE_MAX) {
        die("guarded replay buffer size overflow");
    }
    uint64_t gate_span = 0;
    uint64_t down_span = 0;
    uint64_t end = 0;
    if (!checked_mul_u64(opt->gate_bytes, opt->experts, &gate_span) ||
        !checked_mul_u64(opt->down_bytes, opt->experts, &down_span) ||
        !checked_add_u64(opt->gate_offset, gate_span, &end) || end > model_size ||
        !checked_add_u64(opt->up_offset, gate_span, &end) || end > model_size ||
        !checked_add_u64(opt->down_offset, down_span, &end) || end > model_size) {
        die("expert tensor range is outside the model");
    }
    return layout;
}

static uint64_t stat_mtime_sec(const struct stat *st) {
#if defined(__APPLE__)
    return (uint64_t)st->st_mtimespec.tv_sec;
#else
    return (uint64_t)st->st_mtim.tv_sec;
#endif
}

static uint64_t stat_mtime_nsec(const struct stat *st) {
#if defined(__APPLE__)
    return (uint64_t)st->st_mtimespec.tv_nsec;
#else
    return (uint64_t)st->st_mtim.tv_nsec;
#endif
}

static bool stat_identity_equal(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_size == b->st_size && stat_mtime_sec(a) == stat_mtime_sec(b) &&
           stat_mtime_nsec(a) == stat_mtime_nsec(b);
}

static void put_u32_le(uint8_t *p, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(value >> (8u * i));
}

static void put_u64_le(uint8_t *p, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) p[i] = (uint8_t)(value >> (8u * i));
}

static uint32_t get_u32_le(const uint8_t *p) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; i++) value |= (uint32_t)p[i] << (8u * i);
    return value;
}

static uint64_t get_u64_le(const uint8_t *p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) value |= (uint64_t)p[i] << (8u * i);
    return value;
}

static void make_header(uint8_t header[PROBE_HEADER_BYTES],
                        const probe_options *opt,
                        const probe_layout *layout,
                        const struct stat *model_st) {
    memset(header, 0, PROBE_HEADER_BYTES);
    memcpy(header, PROBE_MAGIC, sizeof(PROBE_MAGIC));
    put_u32_le(header + 8, 1);
    put_u32_le(header + 12, PROBE_HEADER_BYTES);
    put_u64_le(header + 16, (uint64_t)model_st->st_size);
    put_u64_le(header + 24, (uint64_t)model_st->st_dev);
    put_u64_le(header + 32, (uint64_t)model_st->st_ino);
    put_u64_le(header + 40, stat_mtime_sec(model_st));
    put_u64_le(header + 48, stat_mtime_nsec(model_st));
    put_u32_le(header + 56, opt->layer);
    put_u32_le(header + 60, opt->experts);
    put_u64_le(header + 64, opt->gate_offset);
    put_u64_le(header + 72, opt->up_offset);
    put_u64_le(header + 80, opt->down_offset);
    put_u64_le(header + 88, opt->gate_bytes);
    put_u64_le(header + 96, opt->down_bytes);
    put_u64_le(header + 104, layout->record_bytes);
    put_u64_le(header + 112, layout->payload_bytes);
    put_u64_le(header + 120, layout->file_bytes);
}

static bool pread_full(int fd, void *buffer, uint64_t len, uint64_t offset) {
    if (offset > (uint64_t)LLONG_MAX || len > (uint64_t)LLONG_MAX - offset) {
        return false;
    }
    uint8_t *dst = buffer;
    uint64_t done = 0;
    while (done < len) {
        uint64_t remaining = len - done;
        size_t want = remaining > (uint64_t)SSIZE_MAX ? SSIZE_MAX : (size_t)remaining;
        ssize_t n;
        do {
            n = pread(fd, dst + done, want, (off_t)(offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return false;
        done += (uint64_t)n;
    }
    return true;
}

static bool pwrite_full(int fd, const void *buffer, uint64_t len, uint64_t offset) {
    if (offset > (uint64_t)LLONG_MAX || len > (uint64_t)LLONG_MAX - offset) {
        return false;
    }
    const uint8_t *src = buffer;
    uint64_t done = 0;
    while (done < len) {
        uint64_t remaining = len - done;
        size_t want = remaining > (uint64_t)SSIZE_MAX ? SSIZE_MAX : (size_t)remaining;
        ssize_t n;
        do {
            n = pwrite(fd, src + done, want, (off_t)(offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return false;
        done += (uint64_t)n;
    }
    return true;
}

static bool set_nocache(int fd, bool required) {
#if defined(F_NOCACHE)
    if (fcntl(fd, F_NOCACHE, 1) == 0) return true;
    if (required) return false;
    fprintf(stderr,
            "expert-major-probe: warning: F_NOCACHE unavailable on fd: %s\n",
            strerror(errno));
    return true;
#else
    (void)fd;
    if (required) errno = ENOTSUP;
    return !required;
#endif
}

static int open_regular(const char *path, struct stat *st) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) die_errno("open", path);
    if (fstat(fd, st) != 0) die_errno("fstat", path);
    if (!S_ISREG(st->st_mode) || st->st_size < 0) die("input is not a regular file");
    return fd;
}

static bool header_matches(const uint8_t header[PROBE_HEADER_BYTES],
                           const probe_options *opt,
                           const probe_layout *layout,
                           const struct stat *model_st,
                           const struct stat *sidecar_st) {
    return !memcmp(header, PROBE_MAGIC, sizeof(PROBE_MAGIC)) &&
           get_u32_le(header + 8) == 1 &&
           get_u32_le(header + 12) == PROBE_HEADER_BYTES &&
           get_u64_le(header + 16) == (uint64_t)model_st->st_size &&
           get_u64_le(header + 24) == (uint64_t)model_st->st_dev &&
           get_u64_le(header + 32) == (uint64_t)model_st->st_ino &&
           get_u64_le(header + 40) == stat_mtime_sec(model_st) &&
           get_u64_le(header + 48) == stat_mtime_nsec(model_st) &&
           get_u32_le(header + 56) == opt->layer &&
           get_u32_le(header + 60) == opt->experts &&
           get_u64_le(header + 64) == opt->gate_offset &&
           get_u64_le(header + 72) == opt->up_offset &&
           get_u64_le(header + 80) == opt->down_offset &&
           get_u64_le(header + 88) == opt->gate_bytes &&
           get_u64_le(header + 96) == opt->down_bytes &&
           get_u64_le(header + 104) == layout->record_bytes &&
           get_u64_le(header + 112) == layout->payload_bytes &&
           get_u64_le(header + 120) == layout->file_bytes &&
           (uint64_t)sidecar_st->st_size == layout->file_bytes;
}

static bool read_model_record(int fd,
                              const probe_options *opt,
                              const probe_layout *layout,
                              uint32_t expert,
                              uint8_t *record) {
    uint64_t gate_delta = 0;
    uint64_t down_delta = 0;
    if (!checked_mul_u64(expert, opt->gate_bytes, &gate_delta) ||
        !checked_mul_u64(expert, opt->down_bytes, &down_delta) ||
        !pread_full(fd,
                    record,
                    opt->gate_bytes,
                    opt->gate_offset + gate_delta) ||
        !pread_full(fd,
                    record + opt->gate_bytes,
                    opt->gate_bytes,
                    opt->up_offset + gate_delta) ||
        !pread_full(fd,
                    record + opt->gate_bytes * 2,
                    opt->down_bytes,
                    opt->down_offset + down_delta)) {
        (void)layout;
        return false;
    }
    return true;
}

static void interrupt_handler(int signal_number) {
    (void)signal_number;
    g_interrupted = 1;
}

static void command_build(const probe_options *opt) {
    struct stat model_st;
    int model_fd = open_regular(opt->model_path, &model_st);
    probe_layout layout = validate_options(opt, (uint64_t)model_st.st_size);
    (void)set_nocache(model_fd, false);

    if (strlen(opt->sidecar_path) > SIZE_MAX - 32) die("sidecar path is too long");
    size_t temp_len = strlen(opt->sidecar_path) + 32;
    char *temp_path = xmalloc(temp_len);
    snprintf(temp_path, temp_len, "%s.tmp.XXXXXX", opt->sidecar_path);
    int temp_fd = mkstemp(temp_path);
    if (temp_fd < 0) die_errno("mkstemp", temp_path);
    (void)fcntl(temp_fd, F_SETFD, FD_CLOEXEC);
    (void)set_nocache(temp_fd, false);
    bool keep_temp = true;
    uint8_t *record = xmalloc((size_t)layout.record_bytes);
    uint8_t zero_header[PROBE_HEADER_BYTES] = {0};
    if (!pwrite_full(temp_fd, zero_header, sizeof(zero_header), 0)) {
        goto build_error;
    }

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    for (uint32_t expert = 0; expert < opt->experts; expert++) {
        if (g_interrupted) {
            errno = EINTR;
            goto build_error;
        }
        if (!read_model_record(model_fd, opt, &layout, expert, record)) {
            errno = EIO;
            goto build_error;
        }
        uint64_t offset = PROBE_HEADER_BYTES + (uint64_t)expert * layout.record_bytes;
        if (!pwrite_full(temp_fd, record, layout.record_bytes, offset)) {
            goto build_error;
        }
        if ((expert + 1) % 32 == 0 || expert + 1 == opt->experts) {
            fprintf(stderr,
                    "expert-major-probe: build %u/%u experts\r",
                    expert + 1,
                    opt->experts);
        }
    }
    fputc('\n', stderr);
    struct stat model_after;
    if (fstat(model_fd, &model_after) != 0 ||
        !stat_identity_equal(&model_st, &model_after)) {
        errno = ESTALE;
        goto build_error;
    }
    uint8_t header[PROBE_HEADER_BYTES];
    make_header(header, opt, &layout, &model_st);
    if (!pwrite_full(temp_fd, header, sizeof(header), 0) || fsync(temp_fd) != 0) {
        goto build_error;
    }
    if (close(temp_fd) != 0) {
        temp_fd = -1;
        goto build_error;
    }
    temp_fd = -1;
    if (link(temp_path, opt->sidecar_path) != 0) goto build_error;
    if (unlink(temp_path) != 0) {
        fprintf(stderr,
                "expert-major-probe: warning: cannot unlink temp hardlink %s: %s\n",
                temp_path,
                strerror(errno));
    }
    keep_temp = false;
    fprintf(stdout,
            "built sidecar=%s layer=%u experts=%u record_bytes=%" PRIu64
            " file_bytes=%" PRIu64 "\n",
            opt->sidecar_path,
            opt->layer,
            opt->experts,
            layout.record_bytes,
            layout.file_bytes);
    free(record);
    free(temp_path);
    close(model_fd);
    return;

build_error: {
        int saved_errno = errno;
        if (temp_fd >= 0) close(temp_fd);
        if (keep_temp) unlink(temp_path);
        close(model_fd);
        free(record);
        fprintf(stderr,
                "expert-major-probe: sidecar build failed: %s\n",
                strerror(saved_errno));
        free(temp_path);
        exit(1);
    }
}

static probe_layout open_and_validate_header(const probe_options *opt,
                                             int *model_fd_out,
                                             int *sidecar_fd_out,
                                             struct stat *model_st_out) {
    struct stat model_st;
    struct stat sidecar_st;
    int model_fd = open_regular(opt->model_path, &model_st);
    probe_layout layout = validate_options(opt, (uint64_t)model_st.st_size);
    int sidecar_fd = open_regular(opt->sidecar_path, &sidecar_st);
    uint8_t header[PROBE_HEADER_BYTES];
    if (!pread_full(sidecar_fd, header, sizeof(header), 0) ||
        !header_matches(header, opt, &layout, &model_st, &sidecar_st)) {
        close(sidecar_fd);
        close(model_fd);
        die("sidecar header/source/layout mismatch");
    }
    *model_fd_out = model_fd;
    *sidecar_fd_out = sidecar_fd;
    if (model_st_out) *model_st_out = model_st;
    return layout;
}

static probe_layout verify_artifact(const probe_options *opt, bool require_nocache) {
    int model_fd = -1;
    int sidecar_fd = -1;
    struct stat model_before;
    probe_layout layout = open_and_validate_header(opt,
                                                   &model_fd,
                                                   &sidecar_fd,
                                                   &model_before);
    if (!set_nocache(model_fd, require_nocache) ||
        !set_nocache(sidecar_fd, require_nocache)) {
        close(sidecar_fd);
        close(model_fd);
        die_errno("F_NOCACHE", "verification fds");
    }
    uint8_t *expected = xmalloc((size_t)layout.record_bytes);
    size_t guarded_size = (size_t)layout.record_bytes + 2 * PROBE_GUARD_BYTES;
    uint8_t *guarded = xmalloc(guarded_size);
    for (uint32_t expert = 0; expert < opt->experts; expert++) {
        memset(guarded, 0xa5, guarded_size);
        if (!read_model_record(model_fd, opt, &layout, expert, expected)) {
            die("short model expert read during verification");
        }
        uint64_t offset = PROBE_HEADER_BYTES + (uint64_t)expert * layout.record_bytes;
        if (!pread_full(sidecar_fd,
                        guarded + PROBE_GUARD_BYTES,
                        layout.record_bytes,
                        offset) ||
            memcmp(expected,
                   guarded + PROBE_GUARD_BYTES,
                   (size_t)layout.record_bytes) != 0) {
            fprintf(stderr, "expert-major-probe: verify mismatch at expert %u\n", expert);
            exit(1);
        }
        for (size_t i = 0; i < PROBE_GUARD_BYTES; i++) {
            if (guarded[i] != 0xa5 ||
                guarded[PROBE_GUARD_BYTES + layout.record_bytes + i] != 0xa5) {
                die("verification redzone changed");
            }
        }
    }
    struct stat model_after;
    if (fstat(model_fd, &model_after) != 0 ||
        !stat_identity_equal(&model_before, &model_after)) {
        die("model identity changed during verification");
    }
    free(guarded);
    free(expected);
    close(sidecar_fd);
    close(model_fd);
    fprintf(stdout,
            "verified sidecar=%s layer=%u experts=%u bytes=%" PRIu64 "\n",
            opt->sidecar_path,
            opt->layer,
            opt->experts,
            layout.payload_bytes);
    return layout;
}

static void trace_push(probe_trace *trace, const probe_trace_row *row) {
    if (trace->len == trace->cap) {
        size_t next = trace->cap ? trace->cap * 2 : 1024;
        if (next < trace->cap || next > SIZE_MAX / sizeof(*trace->rows)) {
            die("trace is too large");
        }
        trace->rows = xrealloc(trace->rows, next * sizeof(*trace->rows));
        trace->cap = next;
    }
    if (trace->misses > UINT64_MAX - row->n_missing) die("trace miss count overflow");
    trace->rows[trace->len++] = *row;
    trace->misses += row->n_missing;
}

static bool split_nine_fields(char *line, char *fields[9]) {
    if (strchr(line, '"')) return false;
    fields[0] = line;
    for (size_t i = 1; i < 9; i++) {
        char *comma = strchr(fields[i - 1], ',');
        if (!comma) return false;
        *comma = '\0';
        fields[i] = comma + 1;
    }
    return strchr(fields[8], ',') == NULL;
}

static bool parse_selected(char *text,
                           uint32_t expert_limit,
                           uint32_t out[PROBE_SELECTED]) {
    char *part = text;
    for (uint32_t i = 0; i < PROBE_SELECTED; i++) {
        char *separator = strchr(part, ';');
        if ((i + 1 < PROBE_SELECTED && !separator) ||
            (i + 1 == PROBE_SELECTED && separator) ||
            part[0] == '\0') {
            return false;
        }
        if (separator) *separator = '\0';
        uint64_t value = 0;
        if (!parse_u64(part, &value) || value >= expert_limit) return false;
        out[i] = (uint32_t)value;
        for (uint32_t j = 0; j < i; j++) {
            if (out[j] == out[i]) return false;
        }
        if (separator) part = separator + 1;
    }
    return true;
}

static probe_trace parse_trace(const probe_options *opt,
                               const probe_layout *layout) {
    FILE *fp = fopen(opt->trace_path, "rb");
    if (!fp) die_errno("open", opt->trace_path);
    probe_trace trace = {0};
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    uint64_t line_number = 0;
    const char *header =
        "trace_version,phase,request,token,layer,selected_ids,hit_mask,cache_size,expert_bytes";
    while ((line_len = getline(&line, &line_cap, fp)) >= 0) {
        line_number++;
        if (line_len > 0 && line[line_len - 1] == '\n') line[--line_len] = '\0';
        if (line_len > 0 && line[line_len - 1] == '\r') line[--line_len] = '\0';
        if (line_number == 1) {
            if (strcmp(line, header)) die("trace header mismatch");
            continue;
        }
        char *fields[9];
        if (!split_nine_fields(line, fields) || strcmp(fields[0], "1")) {
            goto malformed;
        }
        bool decode = !strcmp(fields[1], "decode");
        if (!decode && strcmp(fields[1], "prefill")) goto malformed;
        uint64_t request = 0, token = 0, layer = 0, cache_size = 0, expert_bytes = 0;
        if (!parse_u64(fields[2], &request) || !parse_u64(fields[3], &token) ||
            !parse_u64(fields[4], &layer) || !parse_u64(fields[7], &cache_size) ||
            !parse_u64(fields[8], &expert_bytes) ||
            expert_bytes != layout->record_bytes) {
            goto malformed;
        }
        (void)token;
        (void)cache_size;
        probe_trace_row row = {0};
        if (!parse_selected(fields[5], opt->experts, row.experts) ||
            strlen(fields[6]) != PROBE_SELECTED) {
            goto malformed;
        }
        for (uint32_t i = 0; i < PROBE_SELECTED; i++) {
            if (fields[6][i] != '0' && fields[6][i] != '1') goto malformed;
            if (fields[6][i] == '0') row.missing[row.n_missing++] = row.experts[i];
        }
        if (decode && layer == opt->layer && request >= opt->min_request) {
            trace_push(&trace, &row);
        }
        continue;
malformed:
        fprintf(stderr,
                "expert-major-probe: malformed trace row at line %" PRIu64 "\n",
                line_number);
        exit(1);
    }
    free(line);
    if (ferror(fp)) die_errno("read", opt->trace_path);
    fclose(fp);
    if (line_number < 2 || trace.len == 0) die("trace selection is empty");
    return trace;
}

static void *pool_worker(void *opaque) {
    probe_worker_arg *arg = opaque;
    probe_pool *pool = arg->pool;
    uint64_t seen_generation = 0;
    pthread_mutex_lock(&pool->mutex);
    for (;;) {
        while (!pool->stop && pool->generation == seen_generation) {
            pthread_cond_wait(&pool->work_cond, &pool->mutex);
        }
        if (pool->stop) break;
        seen_generation = pool->generation;
        probe_io_task *tasks = pool->tasks;
        uint32_t n_tasks = pool->n_tasks;
        uint32_t n_active = pool->n_active;
        bool participate = arg->index < n_active;
        pthread_mutex_unlock(&pool->mutex);

        bool ok = true;
        if (participate) {
            for (uint32_t i = arg->index; i < n_tasks; i += n_active) {
                tasks[i].ok = pread_full(tasks[i].fd,
                                         tasks[i].dst,
                                         tasks[i].len,
                                         tasks[i].offset);
                if (!tasks[i].ok) ok = false;
            }
        }

        pthread_mutex_lock(&pool->mutex);
        if (participate) {
            if (!ok) pool->failed = true;
            if (--pool->active_remaining == 0) {
                pthread_cond_signal(&pool->done_cond);
            }
        }
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static bool pool_init(probe_pool *pool, uint32_t n_threads) {
    memset(pool, 0, sizeof(*pool));
    pool->n_threads = n_threads;
    if (pthread_mutex_init(&pool->mutex, NULL) != 0 ||
        pthread_cond_init(&pool->work_cond, NULL) != 0 ||
        pthread_cond_init(&pool->done_cond, NULL) != 0) {
        return false;
    }
    pool->threads = xmalloc((size_t)n_threads * sizeof(*pool->threads));
    pool->args = xmalloc((size_t)n_threads * sizeof(*pool->args));
    uint32_t started = 0;
    for (; started < n_threads; started++) {
        pool->args[started] = (probe_worker_arg){pool, started};
        if (pthread_create(&pool->threads[started],
                           NULL,
                           pool_worker,
                           &pool->args[started]) != 0) {
            break;
        }
    }
    if (started == n_threads) return true;
    pthread_mutex_lock(&pool->mutex);
    pool->stop = true;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mutex);
    for (uint32_t i = 0; i < started; i++) pthread_join(pool->threads[i], NULL);
    return false;
}

static bool pool_run(probe_pool *pool, probe_io_task *tasks, uint32_t n_tasks) {
    if (n_tasks == 0 || n_tasks > PROBE_MAX_TASKS) return n_tasks == 0;
    pthread_mutex_lock(&pool->mutex);
    if (pool->active_remaining != 0) {
        pthread_mutex_unlock(&pool->mutex);
        return false;
    }
    pool->tasks = tasks;
    pool->n_tasks = n_tasks;
    pool->n_active = n_tasks < pool->n_threads ? n_tasks : pool->n_threads;
    pool->active_remaining = pool->n_active;
    pool->failed = false;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cond);
    while (pool->active_remaining != 0) {
        pthread_cond_wait(&pool->done_cond, &pool->mutex);
    }
    bool ok = !pool->failed;
    pool->tasks = NULL;
    pool->n_tasks = 0;
    pthread_mutex_unlock(&pool->mutex);
    return ok;
}

static void pool_destroy(probe_pool *pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->stop = true;
    pthread_cond_broadcast(&pool->work_cond);
    pthread_mutex_unlock(&pool->mutex);
    for (uint32_t i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_cond_destroy(&pool->done_cond);
    pthread_cond_destroy(&pool->work_cond);
    pthread_mutex_destroy(&pool->mutex);
    free(pool->args);
    free(pool->threads);
}

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) die("clock_gettime failed");
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static bool advise_range(int fd, uint64_t offset, uint64_t len) {
#if defined(F_RDADVISE)
    while (len > 0) {
        if (offset > (uint64_t)LLONG_MAX) return false;
        int chunk = len > (uint64_t)INT_MAX ? INT_MAX : (int)len;
        struct radvisory advice = {(off_t)offset, chunk};
        if (fcntl(fd, F_RDADVISE, &advice) != 0) return false;
        offset += (uint64_t)chunk;
        len -= (uint64_t)chunk;
    }
    return true;
#else
    (void)fd;
    (void)offset;
    (void)len;
    errno = ENOTSUP;
    return false;
#endif
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t len) {
    const uint8_t *p = data;
    while (len >= sizeof(uint64_t)) {
        uint64_t word;
        memcpy(&word, p, sizeof(word));
        hash ^= word;
        hash *= UINT64_C(0x9e3779b185ebca87);
        hash ^= hash >> 29;
        p += sizeof(word);
        len -= sizeof(word);
    }
    while (len-- > 0) {
        hash ^= *p++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int compare_double(const void *a, const void *b) {
    double av = *(const double *)a;
    double bv = *(const double *)b;
    return (av > bv) - (av < bv);
}

static double percentile(double *values, size_t n, double fraction) {
    if (n == 0) return 0.0;
    qsort(values, n, sizeof(*values), compare_double);
    size_t index = (size_t)((double)(n - 1) * fraction + 0.5);
    if (index >= n) index = n - 1;
    return values[index];
}

static probe_run_result run_mode(char mode,
                                 const probe_options *opt,
                                 const probe_layout *layout,
                                 const probe_trace *trace,
                                 int model_fd,
                                 int sidecar_fd,
                                 probe_pool *pool) {
    probe_run_result result = {.mode = mode, .rows = trace->len};
    size_t payload_size = (size_t)(layout->record_bytes * PROBE_SELECTED);
    size_t guarded_size = payload_size + 2 * PROBE_GUARD_BYTES;
    uint8_t *guarded = xmalloc(guarded_size);
    uint8_t *payload = guarded + PROBE_GUARD_BYTES;
    double *latencies = xmalloc(trace->len * sizeof(*latencies));
    size_t n_latencies = 0;
    result.hash = UINT64_C(1469598103934665603);

    for (size_t row_i = 0; row_i < trace->len; row_i++) {
        const probe_trace_row *row = &trace->rows[row_i];
        if (row->n_missing == 0) continue;
        result.miss_rows++;
        memset(guarded, 0xa5, guarded_size);
        memset(payload, 0xcc, payload_size);
        probe_io_task tasks[PROBE_MAX_TASKS] = {0};
        uint32_t n_tasks = 0;
        double t0 = now_ms();
        for (uint32_t i = 0; i < row->n_missing; i++) {
            uint32_t expert = row->missing[i];
            uint8_t *dst = payload + (uint64_t)i * layout->record_bytes;
            uint64_t gate_delta = (uint64_t)expert * opt->gate_bytes;
            uint64_t down_delta = (uint64_t)expert * opt->down_bytes;
            if (mode == 'A') {
                if (!advise_range(model_fd,
                                  opt->gate_offset + gate_delta,
                                  opt->gate_bytes) ||
                    !advise_range(model_fd,
                                  opt->up_offset + gate_delta,
                                  opt->gate_bytes) ||
                    !advise_range(model_fd,
                                  opt->down_offset + down_delta,
                                  opt->down_bytes)) {
                    die_errno("F_RDADVISE", opt->model_path);
                }
                result.advice_calls += 3;
                tasks[n_tasks++] = (probe_io_task){model_fd,
                                                   opt->gate_offset + gate_delta,
                                                   opt->gate_bytes,
                                                   dst,
                                                   false};
                tasks[n_tasks++] = (probe_io_task){model_fd,
                                                   opt->up_offset + gate_delta,
                                                   opt->gate_bytes,
                                                   dst + opt->gate_bytes,
                                                   false};
                tasks[n_tasks++] = (probe_io_task){model_fd,
                                                   opt->down_offset + down_delta,
                                                   opt->down_bytes,
                                                   dst + opt->gate_bytes * 2,
                                                   false};
            } else if (mode == 'B') {
                uint64_t source = PROBE_HEADER_BYTES +
                                  (uint64_t)expert * layout->record_bytes;
                if (!advise_range(sidecar_fd, source, layout->record_bytes)) {
                    die_errno("F_RDADVISE", opt->sidecar_path);
                }
                result.advice_calls++;
                tasks[n_tasks++] = (probe_io_task){sidecar_fd,
                                                   source,
                                                   layout->record_bytes,
                                                   dst,
                                                   false};
            } else {
                uint64_t source = PROBE_HEADER_BYTES +
                                  (uint64_t)expert * layout->record_bytes;
                uint64_t gate_up_bytes = opt->gate_bytes * 2;
                if (!advise_range(sidecar_fd, source, gate_up_bytes) ||
                    !advise_range(sidecar_fd,
                                  source + gate_up_bytes,
                                  opt->down_bytes)) {
                    die_errno("F_RDADVISE", opt->sidecar_path);
                }
                result.advice_calls += 2;
                tasks[n_tasks++] = (probe_io_task){sidecar_fd,
                                                   source,
                                                   gate_up_bytes,
                                                   dst,
                                                   false};
                tasks[n_tasks++] = (probe_io_task){sidecar_fd,
                                                   source + gate_up_bytes,
                                                   opt->down_bytes,
                                                   dst + gate_up_bytes,
                                                   false};
            }
        }
        double t1 = now_ms();
        if (!pool_run(pool, tasks, n_tasks)) die("persistent pread pool failed");
        double t2 = now_ms();
        result.advice_ms += t1 - t0;
        result.pread_ms += t2 - t1;
        result.total_ms += t2 - t0;
        latencies[n_latencies++] = t2 - t0;
        if (result.misses > UINT64_MAX - row->n_missing ||
            result.tasks > UINT64_MAX - n_tasks) {
            die("replay counter overflow");
        }
        uint64_t row_bytes = 0;
        if (!checked_mul_u64(row->n_missing, layout->record_bytes, &row_bytes) ||
            result.logical_bytes > UINT64_MAX - row_bytes) {
            die("replay logical byte count overflow");
        }
        result.misses += row->n_missing;
        result.tasks += n_tasks;
        result.logical_bytes += row_bytes;
        result.hash = hash_bytes(result.hash,
                                 row->missing,
                                 row->n_missing * sizeof(row->missing[0]));
        result.hash = hash_bytes(result.hash,
                                 payload,
                                 (size_t)((uint64_t)row->n_missing *
                                          layout->record_bytes));
        for (size_t i = 0; i < PROBE_GUARD_BYTES; i++) {
            if (guarded[i] != 0xa5 || guarded[PROBE_GUARD_BYTES + payload_size + i] != 0xa5) {
                die("replay redzone changed");
            }
        }
    }
    double *copy = xmalloc(n_latencies * sizeof(*copy));
    memcpy(copy, latencies, n_latencies * sizeof(*copy));
    result.p50_ms = percentile(copy, n_latencies, 0.50);
    memcpy(copy, latencies, n_latencies * sizeof(*copy));
    result.p95_ms = percentile(copy, n_latencies, 0.95);
    free(copy);
    free(latencies);
    free(guarded);
    return result;
}

static void write_csv_header(FILE *fp) {
    fprintf(fp,
            "run,mode,cache_mode,rows,miss_rows,misses,tasks,logical_bytes,"
            "advice_calls,advice_ms,pread_ms,total_ms,row_p50_ms,row_p95_ms,hash\n");
}

static void print_result(FILE *csv,
                         size_t run_index,
                         const char *cache_mode,
                         const probe_run_result *r) {
    fprintf(stdout,
            "run=%zu mode=%c cache=%s rows=%" PRIu64 " miss_rows=%" PRIu64
            " misses=%" PRIu64 " tasks=%" PRIu64 " bytes=%" PRIu64
            " advice_ms=%.3f pread_ms=%.3f total_ms=%.3f p50_ms=%.3f"
            " p95_ms=%.3f hash=%016" PRIx64 "\n",
            run_index,
            r->mode,
            cache_mode,
            r->rows,
            r->miss_rows,
            r->misses,
            r->tasks,
            r->logical_bytes,
            r->advice_ms,
            r->pread_ms,
            r->total_ms,
            r->p50_ms,
            r->p95_ms,
            r->hash);
    if (csv) {
        fprintf(csv,
                "%zu,%c,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%.6f,%.6f,%.6f,%.6f,%.6f,%016"
                PRIx64 "\n",
                run_index,
                r->mode,
                cache_mode,
                r->rows,
                r->miss_rows,
                r->misses,
                r->tasks,
                r->logical_bytes,
                r->advice_calls,
                r->advice_ms,
                r->pread_ms,
                r->total_ms,
                r->p50_ms,
                r->p95_ms,
                r->hash);
        fflush(csv);
    }
}

static void command_replay(const probe_options *opt) {
    probe_layout verified = verify_artifact(opt, true);
    probe_trace trace = parse_trace(opt, &verified);
    uint64_t expected_logical_bytes = 0;
    if (!checked_mul_u64(trace.misses,
                         verified.record_bytes,
                         &expected_logical_bytes)) {
        die("trace logical byte count overflow");
    }
    int model_fd = -1;
    int sidecar_fd = -1;
    probe_layout layout = open_and_validate_header(opt,
                                                   &model_fd,
                                                   &sidecar_fd,
                                                   NULL);
    bool nocache = !strcmp(opt->cache_mode, "nocache");
    if (nocache &&
        (!set_nocache(model_fd, true) || !set_nocache(sidecar_fd, true))) {
        die_errno("F_NOCACHE", "replay fds");
    }
    fprintf(stderr,
            "expert-major-probe: replay cache=%s; physical SSD I/O is not"
            " directly observed, so compare paired loader time only\n",
            opt->cache_mode);

    probe_pool pool;
    if (!pool_init(&pool, opt->threads)) die("cannot start persistent pread pool");
    size_t n_runs = strlen(opt->order);
    probe_run_result *results = xmalloc(n_runs * sizeof(*results));
    FILE *csv = NULL;
    if (opt->csv_path) {
        int csv_fd = open(opt->csv_path,
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          0644);
        if (csv_fd < 0) die_errno("create non-clobbering CSV", opt->csv_path);
        csv = fdopen(csv_fd, "w");
        if (!csv) {
            int saved_errno = errno;
            close(csv_fd);
            unlink(opt->csv_path);
            errno = saved_errno;
            die_errno("fdopen", opt->csv_path);
        }
        write_csv_header(csv);
    }
    for (size_t i = 0; i < n_runs; i++) {
        results[i] = run_mode(opt->order[i],
                              opt,
                              &layout,
                              &trace,
                              model_fd,
                              sidecar_fd,
                              &pool);
        print_result(csv, i + 1, opt->cache_mode, &results[i]);
        uint64_t task_multiplier = results[i].mode == 'A' ? 3u :
                                   results[i].mode == 'B' ? 1u : 2u;
        uint64_t expected_tasks = 0;
        if (!checked_mul_u64(results[i].misses,
                             task_multiplier,
                             &expected_tasks)) {
            die("expected task count overflow");
        }
        if (results[i].rows != trace.len || results[i].misses != trace.misses ||
            results[i].tasks != expected_tasks ||
            results[i].logical_bytes != expected_logical_bytes) {
            die("replay counter invariant failed");
        }
        if (i > 0 && (results[i].hash != results[0].hash ||
                      results[i].logical_bytes != results[0].logical_bytes)) {
            die("A/B destination byte hash mismatch");
        }
    }
    if (csv && fclose(csv) != 0) die_errno("close", opt->csv_path);
    double a_total = 0.0, candidate_total = 0.0;
    uint32_t a_count = 0, candidate_count = 0;
    char candidate_mode = '\0';
    for (size_t i = 0; i < n_runs; i++) {
        if (results[i].mode == 'A') {
            a_total += results[i].total_ms;
            a_count++;
        } else {
            if (candidate_mode != '\0' && candidate_mode != results[i].mode) {
                die("order contains multiple candidate modes");
            }
            candidate_mode = results[i].mode;
            candidate_total += results[i].total_ms;
            candidate_count++;
        }
    }
    if (!a_count || !candidate_count) die("order must contain A and a candidate");
    double a_mean = a_total / a_count;
    double candidate_mean = candidate_total / candidate_count;
    fprintf(stdout,
            "summary A_mean_ms=%.3f %c_mean_ms=%.3f sidecar_delta=%.2f%%"
            " rows=%zu misses=%" PRIu64 " logical_bytes=%" PRIu64 "\n",
            a_mean,
            candidate_mode,
            candidate_mean,
            100.0 * (candidate_mean / a_mean - 1.0),
            trace.len,
            trace.misses,
            expected_logical_bytes);
    free(results);
    pool_destroy(&pool);
    close(sidecar_fd);
    close(model_fd);
    free(trace.rows);
}

int main(int argc, char **argv) {
    probe_options opt = parse_options(argc, argv);
    if (opt.command == PROBE_COMMAND_BUILD) {
        command_build(&opt);
    } else if (opt.command == PROBE_COMMAND_VERIFY) {
        (void)verify_artifact(&opt, false);
    } else {
        command_replay(&opt);
    }
    return 0;
}
