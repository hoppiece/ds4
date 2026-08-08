#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
        __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000

enum {
    DS4_METAL_IO_TEST_BYTES = 64 * 1024,
    DS4_METAL_IO_BUFFER_PADDING = 4 * 1024,
    DS4_METAL_IO_TIMEOUT_SECONDS = 5,
};

typedef enum {
    DS4_METAL_IO_SOURCE_RANGE_ERROR = -1,
    DS4_METAL_IO_SOURCE_RANGE_OUTSIDE_FILE = 0,
    DS4_METAL_IO_SOURCE_RANGE_VALID = 1,
} ds4_metal_io_source_range_result;

static const char *ds4_metal_io_status_name(MTLIOStatus status)
        API_AVAILABLE(macos(13.0)) {
    switch (status) {
        case MTLIOStatusPending: return "pending";
        case MTLIOStatusCancelled: return "cancelled";
        case MTLIOStatusError: return "error";
        case MTLIOStatusComplete: return "complete";
    }
    return "unknown";
}

static int ds4_write_all(int fd, const uint8_t *data, size_t size) {
    size_t written = 0;
    while (written < size) {
        const ssize_t n = write(fd, data + written, size - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        written += (size_t)n;
    }
    return 1;
}

/*
 * MTLIOCommandBuffer does not reliably report a source range beyond EOF as an
 * error.  macOS 15.6 has been observed to complete such a command with no
 * NSError.  The production loader must therefore validate every source range
 * against the already-open model fd before encoding a Metal IO command.
 *
 * Subtraction after the offset check avoids offset + size overflow.  Keeping
 * the fd open also avoids resolving the path to a different file between the
 * check and file-handle creation; production must still treat later truncation
 * as an external mutation that invalidates the model.
 */
static ds4_metal_io_source_range_result ds4_metal_io_source_range_in_file(
        int fd,
        uint64_t source_offset,
        uint64_t size,
        uint64_t *file_size_out) {
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 || !S_ISREG(st.st_mode)) {
        return DS4_METAL_IO_SOURCE_RANGE_ERROR;
    }

    const uint64_t file_size = (uint64_t)st.st_size;
    if (file_size_out) *file_size_out = file_size;
    if (source_offset > file_size || size > file_size - source_offset) {
        return DS4_METAL_IO_SOURCE_RANGE_OUTSIDE_FILE;
    }
    return DS4_METAL_IO_SOURCE_RANGE_VALID;
}

static id<MTLIOFileHandle> ds4_new_io_file_handle(
        id<MTLDevice> device,
        NSURL *url,
        NSError **error) API_AVAILABLE(macos(13.0)) {
    if (@available(macOS 14.0, *)) {
        return [device newIOFileHandleWithURL:url error:error];
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return [device newIOHandleWithURL:url error:error];
#pragma clang diagnostic pop
}

static int ds4_wait_for_io(
        id<MTLIOCommandBuffer> command_buffer,
        const char *label) API_AVAILABLE(macos(13.0)) {
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    [command_buffer addCompletedHandler:^(id<MTLIOCommandBuffer> buffer) {
        (void)buffer;
        dispatch_semaphore_signal(completed);
    }];
    [command_buffer commit];

    const dispatch_time_t deadline = dispatch_time(
            DISPATCH_TIME_NOW,
            (int64_t)DS4_METAL_IO_TIMEOUT_SECONDS * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(completed, deadline) != 0) {
        fprintf(stderr,
                "FAIL: %s Metal IO command timed out after %d seconds\n",
                label,
                DS4_METAL_IO_TIMEOUT_SECONDS);
        [command_buffer tryCancel];
        return 0;
    }

    /* Completion already happened, so this also checks the synchronous API
     * without making the error-path test vulnerable to an unbounded wait. */
    [command_buffer waitUntilCompleted];
    return 1;
}

static int ds4_test_missing_file_handle(id<MTLDevice> device, NSString *path)
        API_AVAILABLE(macos(13.0)) {
    NSString *missing_path = [path stringByAppendingString:@".missing"];
    (void)unlink(missing_path.fileSystemRepresentation);

    NSError *error = nil;
    id<MTLIOFileHandle> handle = ds4_new_io_file_handle(
            device,
            [NSURL fileURLWithPath:missing_path],
            &error);
    if (handle != nil || error == nil) {
        fprintf(stderr,
                "FAIL: nonexistent Metal IO source returned handle=%s error=%s\n",
                handle ? "yes" : "no",
                error ? error.localizedDescription.UTF8String : "none");
        return 0;
    }
    return 1;
}

static int ds4_test_known_load(
        id<MTLDevice> device,
        id<MTLIOCommandQueue> queue,
        NSString *path,
        const uint8_t *expected) API_AVAILABLE(macos(13.0)) {
    NSError *error = nil;
    id<MTLIOFileHandle> handle = ds4_new_io_file_handle(
            device,
            [NSURL fileURLWithPath:path],
            &error);
    if (!handle) {
        fprintf(stderr,
                "FAIL: Metal IO could not open valid source: %s\n",
                error ? error.localizedDescription.UTF8String : "unknown error");
        return 0;
    }

    const NSUInteger padding = DS4_METAL_IO_BUFFER_PADDING;
    const NSUInteger data_size = DS4_METAL_IO_TEST_BYTES;
    id<MTLBuffer> destination = [device
            newBufferWithLength:data_size + 2 * padding
            options:MTLResourceStorageModeShared];
    if (!destination || !destination.contents) {
        fprintf(stderr, "FAIL: could not allocate shared Metal IO buffer\n");
        return 0;
    }
    memset(destination.contents, 0xa5, destination.length);

    id<MTLSharedEvent> event = [device newSharedEvent];
    if (!event) {
        fprintf(stderr, "FAIL: could not allocate Metal shared event\n");
        return 0;
    }
    const uint64_t event_value = event.signaledValue + 1;

    id<MTLIOCommandBuffer> command_buffer = [queue commandBuffer];
    if (!command_buffer) {
        fprintf(stderr, "FAIL: Metal IO queue returned no command buffer\n");
        return 0;
    }
    command_buffer.label = @"ds4 Metal IO known-byte smoke";

    const NSUInteger half = data_size / 2;
    [command_buffer loadBuffer:destination
                        offset:padding
                          size:half
                  sourceHandle:handle
            sourceHandleOffset:0];
    [command_buffer addBarrier];
    [command_buffer loadBuffer:destination
                        offset:padding + half
                          size:data_size - half
                  sourceHandle:handle
            sourceHandleOffset:half];
    [command_buffer addBarrier];
    [command_buffer signalEvent:event value:event_value];

    if (!ds4_wait_for_io(command_buffer, "known-byte load")) return 0;
    if (command_buffer.status != MTLIOStatusComplete || command_buffer.error) {
        fprintf(stderr,
                "FAIL: known-byte load status=%s error=%s\n",
                ds4_metal_io_status_name(command_buffer.status),
                command_buffer.error ?
                    command_buffer.error.localizedDescription.UTF8String : "none");
        return 0;
    }
    if (event.signaledValue < event_value) {
        fprintf(stderr,
                "FAIL: Metal IO shared event value=%llu, expected at least %llu\n",
                (unsigned long long)event.signaledValue,
                (unsigned long long)event_value);
        return 0;
    }

    const uint8_t *actual = destination.contents;
    for (NSUInteger i = 0; i < padding; i++) {
        if (actual[i] != 0xa5 || actual[padding + data_size + i] != 0xa5) {
            fprintf(stderr, "FAIL: Metal IO load wrote outside destination range\n");
            return 0;
        }
    }
    if (memcmp(actual + padding, expected, data_size) != 0) {
        NSUInteger mismatch = 0;
        while (mismatch < data_size &&
               actual[padding + mismatch] == expected[mismatch]) {
            mismatch++;
        }
        fprintf(stderr,
                "FAIL: Metal IO byte mismatch at offset %llu\n",
                (unsigned long long)mismatch);
        return 0;
    }
    return 1;
}

static int ds4_test_truncated_range_rejected(int fd) {
    uint64_t file_size = 0;
    ds4_metal_io_source_range_result range =
            ds4_metal_io_source_range_in_file(fd,
                                              0,
                                              DS4_METAL_IO_TEST_BYTES,
                                              &file_size);
    if (range != DS4_METAL_IO_SOURCE_RANGE_VALID ||
        file_size != DS4_METAL_IO_TEST_BYTES) {
        fprintf(stderr,
                "FAIL: complete source range was not accepted before truncate "
                "(result=%d size=%llu)\n",
                range,
                (unsigned long long)file_size);
        return 0;
    }

    const uint64_t truncated_size = DS4_METAL_IO_TEST_BYTES / 4;
    if (ftruncate(fd, (off_t)truncated_size) != 0) {
        fprintf(stderr, "FAIL: truncate test setup: %s\n", strerror(errno));
        return 0;
    }

    range = ds4_metal_io_source_range_in_file(fd,
                                              0,
                                              DS4_METAL_IO_TEST_BYTES,
                                              &file_size);
    if (range != DS4_METAL_IO_SOURCE_RANGE_OUTSIDE_FILE ||
        file_size != truncated_size) {
        fprintf(stderr,
                "FAIL: truncated source range was not rejected before Metal IO "
                "encoding (result=%d size=%llu)\n",
                range,
                (unsigned long long)file_size);
        return 0;
    }

    if (ds4_metal_io_source_range_in_file(fd, 0, truncated_size, NULL) !=
            DS4_METAL_IO_SOURCE_RANGE_VALID ||
        ds4_metal_io_source_range_in_file(fd, truncated_size, 0, NULL) !=
            DS4_METAL_IO_SOURCE_RANGE_VALID ||
        ds4_metal_io_source_range_in_file(fd, truncated_size, 1, NULL) !=
            DS4_METAL_IO_SOURCE_RANGE_OUTSIDE_FILE ||
        ds4_metal_io_source_range_in_file(fd, UINT64_MAX, 2, NULL) !=
            DS4_METAL_IO_SOURCE_RANGE_OUTSIDE_FILE) {
        fprintf(stderr, "FAIL: source range boundary/overflow checks failed\n");
        return 0;
    }

    /* No MTLIOCommandBuffer is obtained in this helper: rejection happens
     * before any loadBuffer command can be encoded or committed. */
    return 1;
}

static int ds4_test_source_range_validation(void) {
    char path_template[] = "/tmp/ds4-metal-io-range.XXXXXX";
    const int fd = mkstemp(path_template);
    if (fd < 0) {
        fprintf(stderr, "FAIL: range-test mkstemp: %s\n", strerror(errno));
        return 0;
    }
    int ok = 1;
    if (ftruncate(fd, DS4_METAL_IO_TEST_BYTES) != 0) {
        fprintf(stderr, "FAIL: range-test sizing: %s\n", strerror(errno));
        ok = 0;
    }
    if (ok) ok = ds4_test_truncated_range_rejected(fd);
    if (close(fd) != 0) {
        fprintf(stderr, "FAIL: range-test close: %s\n", strerror(errno));
        ok = 0;
    }
    if (unlink(path_template) != 0) {
        fprintf(stderr, "FAIL: range-test cleanup: %s\n", strerror(errno));
        ok = 0;
    }
    return ok;
}

int main(void) {
    @autoreleasepool {
        if (!ds4_test_source_range_validation()) return 1;

        if (@available(macOS 13.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                puts("PASS: source range rejection; SKIP: Metal device unavailable");
                return 0;
            }
            if (![device respondsToSelector:
                    @selector(newIOCommandQueueWithDescriptor:error:)]) {
                puts("PASS: source range rejection; "
                     "SKIP: Metal IO queue API unavailable on this device");
                return 0;
            }

            MTLIOCommandQueueDescriptor *descriptor =
                    [[MTLIOCommandQueueDescriptor alloc] init];
            descriptor.type = MTLIOCommandQueueTypeConcurrent;
            descriptor.priority = MTLIOPriorityNormal;
            descriptor.maxCommandBufferCount = 2;
            descriptor.maxCommandsInFlight = 4;

            NSError *queue_error = nil;
            id<MTLIOCommandQueue> queue =
                    [device newIOCommandQueueWithDescriptor:descriptor
                                                     error:&queue_error];
            if (!queue) {
                printf("PASS: source range rejection; "
                       "SKIP: Metal IO queue unavailable: %s\n",
                       queue_error ?
                           queue_error.localizedDescription.UTF8String :
                           "unknown error");
                return 0;
            }
            queue.label = @"ds4 standalone Metal IO smoke queue";

            uint8_t expected[DS4_METAL_IO_TEST_BYTES];
            for (size_t i = 0; i < sizeof(expected); i++) {
                expected[i] = (uint8_t)((i * 131u + (i >> 3) * 17u + 0x5au) & 0xffu);
            }

            char path_template[] = "/tmp/ds4-metal-io-smoke.XXXXXX";
            const int fd = mkstemp(path_template);
            if (fd < 0) {
                fprintf(stderr, "FAIL: mkstemp: %s\n", strerror(errno));
                return 1;
            }
            int ok = ds4_write_all(fd, expected, sizeof(expected));
            if (ok && fsync(fd) != 0) ok = 0;
            const int close_ok = close(fd) == 0;
            if (!ok || !close_ok) {
                fprintf(stderr, "FAIL: temporary source write: %s\n", strerror(errno));
                (void)unlink(path_template);
                return 1;
            }

            NSString *path = [NSString stringWithUTF8String:path_template];
            ok = ds4_test_missing_file_handle(device, path) &&
                 ds4_test_known_load(device, queue, path, expected);

            /* MTLIOFileHandle and MTLIOCommandBuffer have no close method.
             * Their ARC lifetimes end with the helper scopes/autorelease pool;
             * unlinking the exact mkstemp path is the filesystem cleanup. */
            if (unlink(path_template) != 0) {
                fprintf(stderr, "FAIL: temporary source cleanup: %s\n", strerror(errno));
                ok = 0;
            }
            if (!ok) return 1;

            puts("PASS: Metal IO known-byte/event and pre-encode range-rejection smoke");
            return 0;
        }

        puts("PASS: source range rejection; "
             "SKIP: Metal IO requires macOS 13 or newer");
        return 0;
    }
}

#else

int main(void) {
    puts("SKIP: build SDK does not expose Metal IO (macOS 13+ SDK required)");
    return 0;
}

#endif
