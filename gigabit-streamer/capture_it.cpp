/*
 * capture_it.cpp — High-performance DIFI/VITA-49.2 UDP capture
 *                  for Moku:Delta Gigabit Streamer
 *
 * Designed for zero-packet-loss capture at up to 312.5 MSa/s over 10GbE.
 *
 * Performance techniques vs. the Python version:
 *   recvmmsg()     — receive up to 64 packets per syscall (64× fewer syscalls)
 *   Buffer pool    — pre-allocated, mlock'd buffers; zero allocation in hot path
 *   Writer thread  — disk I/O overlaps with packet receive
 *   SCHED_FIFO     — real-time scheduling for the receive thread (--rt)
 *   CPU affinity   — pin receive thread to a dedicated core (--cpu N)
 *   Seq tracking   — DIFI 4-bit packet counter detects any dropped packet
 *
 * Build:
 *   make capture_it
 *   # or: g++ -O3 -std=c++17 -o capture_it capture_it.cpp -lpthread
 *
 * Usage:
 *   sudo ./capture_it --outfile capture.bin --seconds 5 --verbose
 *   sudo ./capture_it --outfile capture.bin --seconds 10 \
 *        --socket-buffer 268435456 --rt --cpu 2 --verbose
 *
 * Disk write speed requirements (approximate):
 *   156.25 MSa/s mono  16-bit :  312 MB/s  — fast SATA SSD or NVMe
 *   156.25 MSa/s stereo 16-bit:  625 MB/s  — fast SATA SSD or NVMe
 *   312.5  MSa/s mono  16-bit :  625 MB/s  — NVMe recommended
 *   312.5  MSa/s stereo 16-bit: 1250 MB/s  — fast NVMe required
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

// recvmmsg is Linux-specific (kernel 2.6.33+).
// MSG_WAITFORONE: block until at least one packet arrives, then return all
// available packets up to the batch limit in a single syscall.
#ifndef MSG_WAITFORONE
#  define MSG_WAITFORONE 0x10000
#endif

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr uint32_t DIFI_OUI      = 0x006A621E;
static constexpr uint32_t DIFI_OUI_MASK = 0x00FFFFFFu;
static constexpr uint8_t  PKT_DATA      = 0x1;
static constexpr uint8_t  PKT_CONTEXT   = 0x4;
static constexpr int      HEADER_BYTES  = 28;          // 7 × 32-bit words

// Batch size for recvmmsg().  64 is a good balance: large enough to amortise
// syscall overhead at 865K pkt/s, small enough to keep latency low.
static constexpr int      RECV_BATCH    = 64;

// Max packet size: 9216 handles jumbo frames as well as standard 1500-byte MTU.
static constexpr int      MAX_PKT_BYTES = 9216;

// ─── Global stop flag set by SIGINT / SIGTERM ─────────────────────────────────

static std::atomic<bool> g_stop{false};
static void sig_handler(int) { g_stop.store(true, std::memory_order_relaxed); }

// ─── CLI arguments ────────────────────────────────────────────────────────────

struct Args {
    std::string bind_addr    = "0.0.0.0";
    int         port         = 4991;
    std::string outfile;
    double      seconds      = 0.0;     // max capture duration from first packet (0=unlimited)
    int64_t     max_packets  = 0;       // stop after N data packets (0=unlimited)
    double      wait_timeout = 0.0;     // give up if no packet arrives within this many seconds (0=wait forever)
    int         socket_buf   = 128 * 1024 * 1024;   // 128 MiB
    int         ram_buf_mib  = 256;                  // MiB per buffer
    int         queue_depth  = 4;                    // filled buffers in queue
    bool        rt           = false;
    int         rt_priority  = 80;
    int         cpu          = -1;
    bool        verbose      = false;
    bool        vita49       = false;                // keep full VITA-49.2 packets
};

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --outfile PATH [options]\n"
        "\n"
        "  --bind IP              Bind address (default 0.0.0.0)\n"
        "  --port N               UDP port (default 4991)\n"
        "  --outfile PATH         Output file path\n"
        "  --max-packets N        Stop after N data packets (0 = unlimited).\n"
        "                         Timer does not start until first packet arrives,\n"
        "                         so gaps between bursts are handled correctly.\n"
        "  --seconds F            Max capture duration in seconds, measured from\n"
        "                         the first packet received (0 = unlimited).\n"
        "                         Acts as a safety valve alongside --max-packets.\n"
        "  --wait-timeout F       Give up and exit if no packet arrives within F\n"
        "                         seconds of launch (0 = wait forever, default).\n"
        "  --socket-buffer N      SO_RCVBUF in bytes (default 128 MiB)\n"
        "  --ram-buffer N         Size of each RAM buffer in MiB (default 256)\n"
        "  --write-queue-depth N  Filled buffers queued for writer thread (default 4)\n"
        "                         Total RAM = (N + 2) × ram-buffer MiB\n"
        "  --rt                   Enable SCHED_FIFO real-time scheduling\n"
        "  --rt-priority N        SCHED_FIFO priority 1-99 (default 80)\n"
        "  --cpu N                Pin receive thread to CPU N\n"
        "  --verbose              Print statistics every 5 seconds\n"
        "  --vita49               Preserve full VITA-49.2 packets in output\n"
        "                         (header + payload per packet instead of\n"
        "                          payload-only).  Allows timestamp recovery.\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nxt = [&]() -> std::string {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for %s\n", argv[i-1]);
                exit(1);
            }
            return argv[i];
        };
        if      (s == "--bind")              a.bind_addr    = nxt();
        else if (s == "--port")              a.port         = std::stoi(nxt());
        else if (s == "--outfile")           a.outfile      = nxt();
        else if (s == "--seconds")           a.seconds      = std::stod(nxt());
        else if (s == "--max-packets")       a.max_packets  = std::stoll(nxt());
        else if (s == "--wait-timeout")      a.wait_timeout = std::stod(nxt());
        else if (s == "--socket-buffer")     a.socket_buf   = std::stoi(nxt());
        else if (s == "--ram-buffer")        a.ram_buf_mib  = std::stoi(nxt());
        else if (s == "--write-queue-depth") a.queue_depth  = std::stoi(nxt());
        else if (s == "--rt")                a.rt           = true;
        else if (s == "--rt-priority")       a.rt_priority  = std::stoi(nxt());
        else if (s == "--cpu")               a.cpu          = std::stoi(nxt());
        else if (s == "--verbose")           a.verbose      = true;
        else if (s == "--vita49")            a.vita49       = true;
        else if (s == "--help" || s == "-h") { usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "Unknown option: %s\n", s.c_str());
            usage(argv[0]);
            exit(1);
        }
    }
    if (a.outfile.empty()) {
        fprintf(stderr, "--outfile is required\n");
        usage(argv[0]);
        exit(1);
    }
    return a;
}

// ─── Buffer pool ──────────────────────────────────────────────────────────────
//
// Each Buffer wraps a pre-allocated, page-faulted block of memory.
// The pool is split between two queues:
//   free_q  — empty buffers available for the receive loop
//   write_q — filled buffers waiting for the writer thread
//
// The writer thread pops from write_q, writes to disk, then pushes back to
// free_q.  The receive loop pops from free_q when its current buffer is full.
// pop_interruptible() on free_q respects g_stop so Ctrl-C never hangs.

struct Buffer {
    uint8_t* data     = nullptr;
    size_t   capacity = 0;
    size_t   used     = 0;
};

class BufferQueue {
public:
    // Push a buffer pointer (may be nullptr as a writer-shutdown sentinel).
    void push(Buffer* b) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.push_back(b);
        }
        cv_.notify_one();
    }

    // Blocking pop — used by the writer thread on write_q.
    // Returns nullptr only when the nullptr sentinel is dequeued (shutdown).
    Buffer* pop() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return !q_.empty(); });
        Buffer* b = q_.front();
        q_.pop_front();
        return b;
    }

    // Interruptible pop — used by the receive loop on free_q.
    // Returns nullptr if g_stop is set before a buffer becomes available.
    Buffer* pop_interruptible() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{
            return !q_.empty() || g_stop.load(std::memory_order_relaxed);
        });
        if (q_.empty()) return nullptr;
        Buffer* b = q_.front();
        q_.pop_front();
        return b;
    }

    // Wake any thread blocked in pop_interruptible() (used during shutdown).
    void wake_all() { cv_.notify_all(); }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

private:
    std::mutex              mu_;
    std::condition_variable cv_;
    std::deque<Buffer*>     q_;
};

// ─── Writer thread ────────────────────────────────────────────────────────────

struct WriterCtx {
    BufferQueue*           write_q;
    BufferQueue*           free_q;
    int                    fd;
    std::atomic<uint64_t>  bytes_written{0};
    std::atomic<int>       write_errors{0};
};

static void writer_thread(WriterCtx* ctx) {
    while (true) {
        Buffer* b = ctx->write_q->pop();
        if (!b) break;                        // nullptr sentinel — time to stop

        const uint8_t* ptr  = b->data;
        size_t         left = b->used;
        while (left > 0) {
            ssize_t n = write(ctx->fd, ptr, left);
            if (n > 0) {
                ctx->bytes_written.fetch_add((uint64_t)n,
                                             std::memory_order_relaxed);
                ptr  += n;
                left -= (size_t)n;
            } else if (n < 0 && errno == EINTR) {
                continue;
            } else {
                perror("write");
                ctx->write_errors.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        b->used = 0;
        ctx->free_q->push(b);                 // return buffer to pool
    }
}

// ─── DIFI packet parser ───────────────────────────────────────────────────────
//
// Returns a pointer to the payload inside pkt[], and sets payload_len.
// Returns nullptr for context packets (is_ctx = true) or malformed packets.
// seq_out receives the 4-bit rolling packet counter from the VITA header.

static const uint8_t* parse_difi(const uint8_t* pkt,  int pkt_len,
                                  int&           payload_len,
                                  bool&          is_ctx,
                                  uint8_t&       seq_out) {
    is_ctx      = false;
    payload_len = 0;
    seq_out     = 0;

    if (pkt_len < HEADER_BYTES) return nullptr;

    // VITA-49 header words are big-endian.
    auto be32 = [&](int off) -> uint32_t {
        return  (uint32_t)pkt[off+0] << 24 | (uint32_t)pkt[off+1] << 16
              | (uint32_t)pkt[off+2] <<  8 | (uint32_t)pkt[off+3];
    };

    uint32_t w1 = be32(0);
    uint32_t w3 = be32(8);

    if ((w3 & DIFI_OUI_MASK) != (DIFI_OUI & DIFI_OUI_MASK)) return nullptr;

    uint8_t  pkt_type   = (uint8_t)((w1 >> 28) & 0xF);
    seq_out             = (uint8_t)((w1 >> 16) & 0xF);
    int      tot_bytes  = (int)(w1 & 0xFFFF) * 4;

    if (tot_bytes > pkt_len || tot_bytes <= HEADER_BYTES) return nullptr;

    if (pkt_type == PKT_CONTEXT) { is_ctx = true; return nullptr; }
    if (pkt_type != PKT_DATA)    return nullptr;

    payload_len = tot_bytes - HEADER_BYTES;
    return pkt + HEADER_BYTES;
}

// ─── Real-time setup ──────────────────────────────────────────────────────────

static void setup_realtime(const Args& a) {
    if (a.rt) {
        struct sched_param sp{};
        sp.sched_priority = std::max(1, std::min(99, a.rt_priority));
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
            perror("sched_setscheduler (need root for --rt)");
        else
            fprintf(stderr, "SCHED_FIFO priority %d enabled.\n",
                    sp.sched_priority);
    }
    if (a.cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET((size_t)a.cpu, &cs);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
            perror("pthread_setaffinity_np");
        else
            fprintf(stderr, "Receive thread pinned to CPU %d.\n", a.cpu);
    }
}

// ─── Timing helper ───────────────────────────────────────────────────────────

static inline double monotonic_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    // ── Lock all memory ───────────────────────────────────────────────────────
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        perror("mlockall (non-fatal — run as root for full effect)");
    else
        fprintf(stderr, "Memory locked (mlockall).\n");

    // ── Allocate buffer pool ──────────────────────────────────────────────────
    // Pool = queue_depth + 2:
    //   1 being filled  +  queue_depth queued for writer  +  1 being written
    int    num_bufs   = a.queue_depth + 2;
    size_t buf_bytes  = (size_t)a.ram_buf_mib * 1024 * 1024;
    double total_mib  = (double)num_bufs * a.ram_buf_mib;

    fprintf(stderr, "Allocating %d × %d MiB buffers (%.0f MiB total)...\n",
            num_bufs, a.ram_buf_mib, total_mib);

    std::vector<Buffer> pool((size_t)num_bufs);
    for (auto& b : pool) {
        // aligned_alloc for potential O_DIRECT use and cache-line alignment
        b.data = static_cast<uint8_t*>(aligned_alloc(4096, buf_bytes));
        if (!b.data) { perror("aligned_alloc"); return 1; }
        b.capacity = buf_bytes;
        b.used     = 0;
        // Pre-fault every page — prevents page faults during capture
        memset(b.data, 0, buf_bytes);
    }
    fprintf(stderr, "Buffer pages pre-faulted.\n");

    BufferQueue write_q, free_q;
    for (auto& b : pool) free_q.push(&b);

    // ── Open output file ──────────────────────────────────────────────────────
    int fd = open(a.outfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }

    // ── Start writer thread ───────────────────────────────────────────────────
    WriterCtx wctx{&write_q, &free_q, fd, {0}, {0}};
    std::thread writer(writer_thread, &wctx);

    // ── Static receive batch buffers ──────────────────────────────────────────
    // Static (not stack) to avoid touching ~576 KiB of stack, and to keep
    // the recv buffers in a fixed, TLB-hot region of memory.
    static struct mmsghdr     batch_msgs[RECV_BATCH];
    static struct iovec       batch_iovs[RECV_BATCH];
    static uint8_t            batch_bufs[RECV_BATCH][MAX_PKT_BYTES];
    static struct sockaddr_in batch_addrs[RECV_BATCH];

    memset(batch_msgs, 0, sizeof(batch_msgs));
    for (int i = 0; i < RECV_BATCH; ++i) {
        batch_iovs[i].iov_base              = batch_bufs[i];
        batch_iovs[i].iov_len               = MAX_PKT_BYTES;
        batch_msgs[i].msg_hdr.msg_iov       = &batch_iovs[i];
        batch_msgs[i].msg_hdr.msg_iovlen    = 1;
        batch_msgs[i].msg_hdr.msg_name      = &batch_addrs[i];
        batch_msgs[i].msg_hdr.msg_namelen   = sizeof(batch_addrs[i]);
    }

    // ── Create and configure UDP socket ──────────────────────────────────────
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    // Request large receive buffer — kernel may cap it at rmem_max.
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
               &a.socket_buf, sizeof(a.socket_buf));

    // Report actual buffer size and hint if it was capped.
    int    actual_buf = 0;
    socklen_t optlen  = sizeof(actual_buf);
    getsockopt(sock, SOL_SOCKET, SO_RCVBUF, &actual_buf, &optlen);
    fprintf(stderr, "Socket receive buffer: requested %d MiB, got %d MiB\n",
            a.socket_buf / (1024*1024), actual_buf / (1024*1024));
    if (actual_buf < a.socket_buf / 2)
        fprintf(stderr,
            "  NOTE: kernel capped SO_RCVBUF — raise limit with:\n"
            "  sudo sysctl -w net.core.rmem_max=%d\n"
            "  sudo sysctl -w net.core.rmem_default=%d\n",
            a.socket_buf * 2, a.socket_buf);

    // 500 ms receive timeout so the loop can check stop conditions.
    struct timeval tv{};
    tv.tv_sec  = 0;
    tv.tv_usec = 500 * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // ── Set real-time scheduling BEFORE bind ──────────────────────────────────
    // We want RT priority in place before the first packet arrives.
    setup_realtime(a);

    // ── Bind — this is the moment packets start flowing ───────────────────────
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)a.port);
    if (inet_pton(AF_INET, a.bind_addr.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "Bad bind address: %s\n", a.bind_addr.c_str());
        return 1;
    }
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    fprintf(stderr, "Listening on %s:%d  (Ctrl-C to stop)\n",
            a.bind_addr.c_str(), a.port);

    // ── Receive loop ──────────────────────────────────────────────────────────

    // Grab the first buffer for the receive loop to fill.
    Buffer* cur = free_q.pop_interruptible();
    if (!cur) { fprintf(stderr, "No buffers available.\n"); return 1; }

    // Statistics
    uint64_t n_data    = 0;   // data packets received
    uint64_t n_ctx     = 0;   // context packets (skipped)
    uint64_t n_bad     = 0;   // malformed / wrong OUI
    uint64_t n_seqgaps = 0;   // DIFI sequence number gaps detected
    uint64_t n_seqlost = 0;   // estimated samples lost to seq gaps

    uint8_t  last_seq    = 0;
    bool     seq_valid   = false;

    // t_launch: when the program started (used only for --wait-timeout).
    // t_start:  when the FIRST data packet arrived (used for --seconds and stats).
    //           Set to 0 until first packet; negative seconds check is skipped.
    double   t_launch    = monotonic_s();
    double   t_start     = 0.0;
    bool     first_pkt   = false;          // true once one data packet is received

    double   t_stat      = 0.0;            // next periodic stats print (set on first pkt)
    double   t_prev_stat = 0.0;
    uint64_t n_data_prev = 0;
    uint64_t bytes_prev  = 0;

    if (a.wait_timeout > 0.0)
        fprintf(stderr, "Waiting for first packet  (timeout %.0f s)...\n",
                a.wait_timeout);
    else
        fprintf(stderr, "Waiting for first packet  (no timeout)...\n");

    while (!g_stop.load(std::memory_order_relaxed)) {

        // ── Check stop conditions ─────────────────────────────────────────
        double now = monotonic_s();

        if (!first_pkt) {
            // Still waiting for transmitter to start.
            // Only --wait-timeout applies here.
            if (a.wait_timeout > 0.0 && (now - t_launch) >= a.wait_timeout) {
                fprintf(stderr,
                    "Wait timeout: no data packets received in %.1f s — exiting.\n",
                    a.wait_timeout);
                break;
            }
        } else {
            // Capturing — apply duration and packet-count limits.
            if (a.seconds > 0.0 && (now - t_start) >= a.seconds) break;
        }
        if (a.max_packets > 0 && (int64_t)n_data >= a.max_packets) break;

        // ── Receive a batch of UDP packets ────────────────────────────────
        // MSG_WAITFORONE: block until at least one packet is available,
        // then return all currently buffered packets (up to RECV_BATCH).
        // This gives us the low latency of blocking I/O with the throughput
        // of batch processing — critical at 865K packets/second.
        int n = recvmmsg(sock, batch_msgs, RECV_BATCH, MSG_WAITFORONE, nullptr);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                continue;   // timeout or signal — loop to check conditions
            perror("recvmmsg");
            break;
        }

        // ── Process each packet in the batch ─────────────────────────────
        for (int i = 0; i < n; ++i) {
            if (g_stop.load(std::memory_order_relaxed)) break;

            const uint8_t* pkt     = batch_bufs[i];
            int            pkt_len = (int)batch_msgs[i].msg_len;

            int     payload_len = 0;
            bool    is_ctx      = false;
            uint8_t seq         = 0;
            const uint8_t* payload = parse_difi(pkt, pkt_len,
                                                 payload_len, is_ctx, seq);
            if (is_ctx)   { n_ctx++; continue; }
            if (!payload) { n_bad++; continue; }

            // ── DIFI sequence gap detection ───────────────────────────────
            // The 4-bit rolling counter in the VITA header increments by 1
            // per packet.  Any skip means one or more packets were dropped
            // somewhere between the Moku and this recvmmsg() call.
            if (seq_valid) {
                uint8_t expected = (last_seq + 1u) & 0xFu;
                if (seq != expected) {
                    uint8_t gap = (seq - expected) & 0xFu;
                    n_seqgaps++;
                    // Each missing packet = payload_len bytes of samples
                    n_seqlost += (uint64_t)gap * (uint64_t)(payload_len / 2);
                    if (a.verbose)
                        fprintf(stderr,
                            "[GAP] seq %u→%u  (~%u pkt(s) lost,  "
                            "~%llu samples)\n",
                            expected, seq, gap,
                            (unsigned long long)((uint64_t)gap
                                                 * (payload_len / 2)));
                }
            }
            last_seq  = seq;
            seq_valid = true;

            // ── Flush buffer when full, get next free one ─────────────────
            // free_q.pop_interruptible() respects g_stop so Ctrl-C never
            // hangs waiting for a slow disk to free a buffer.
            //
            // In --vita49 mode each record is the full VITA-49.2 packet
            // (HEADER_BYTES + payload_len); otherwise just the payload.
            size_t         write_len = a.vita49
                                       ? (size_t)(HEADER_BYTES + payload_len)
                                       : (size_t)payload_len;
            const uint8_t* write_src = a.vita49 ? pkt : payload;

            if (cur->used + write_len > cur->capacity) {
                write_q.push(cur);
                cur = free_q.pop_interruptible();
                if (!cur) break;   // g_stop was set while waiting
                cur->used = 0;
            }

            // ── Copy packet data into the RAM buffer ──────────────────────
            // Default (raw) mode : strip the 28-byte VITA header, write
            //   only the little-endian int16 sample payload — compatible
            //   with validate_capture.py and send_it.
            // --vita49 mode      : write the complete VITA-49.2 packet
            //   (header + payload) so TAI timestamps and stream metadata
            //   are preserved for post-processing.
            memcpy(cur->data + cur->used, write_src, write_len);
            cur->used += write_len;
            n_data++;

            // ── Start capture timer on first data packet ──────────────────
            // The timer is intentionally deferred so that --seconds measures
            // actual signal duration rather than time spent waiting for the
            // transmitter to start streaming.
            if (!first_pkt) {
                first_pkt    = true;
                t_start      = monotonic_s();
                t_prev_stat  = t_start;
                t_stat       = t_start + 5.0;
                bytes_prev   = wctx.bytes_written.load();
                fprintf(stderr, "First packet received — capture timer started.\n");
                if (a.max_packets > 0)
                    fprintf(stderr, "  Collecting %lld packets%s.\n",
                            (long long)a.max_packets,
                            a.seconds > 0.0 ? " (or max-duration limit)" : "");
                else if (a.seconds > 0.0)
                    fprintf(stderr, "  Collecting for %.1f s.\n", a.seconds);
                else
                    fprintf(stderr, "  Collecting until Stop or Ctrl-C.\n");
            }

            if (a.max_packets > 0 && (int64_t)n_data >= a.max_packets) break;
        }

        // ── Periodic stats ────────────────────────────────────────────────
        if (a.verbose) {
            double t_now = monotonic_s();
            if (!first_pkt) {
                // Print a "still waiting" heartbeat every 5 s
                if (t_now >= t_launch + 5.0 && t_now >= t_stat) {
                    fprintf(stderr,
                        "Waiting... %.0f s elapsed (no data packets yet)\n",
                        t_now - t_launch);
                    t_stat = t_now + 5.0;
                }
            } else if (t_now >= t_stat) {
                double   elapsed  = t_now - t_start;
                double   dt       = t_now - t_prev_stat;
                uint64_t bw       = wctx.bytes_written.load();
                double   pkt_rate = (double)(n_data - n_data_prev) / dt / 1e3;
                double   mb_rate  = (double)(bw - bytes_prev) / dt / (1024.0*1024.0);
                // Show packet-count progress if --max-packets was set
                char progress[64] = "";
                if (a.max_packets > 0)
                    snprintf(progress, sizeof(progress), " | %lld/%lld pkts",
                             (long long)n_data, (long long)a.max_packets);
                fprintf(stderr,
                    "t=%6.1fs | pkts: %7llu%s | ctx: %4llu | bad: %4llu | "
                    "gaps: %4llu | %.0f Kpkt/s | %.0f MB/s | "
                    "free bufs: %zu/%d\n",
                    elapsed,
                    (unsigned long long)n_data,
                    progress,
                    (unsigned long long)n_ctx,
                    (unsigned long long)n_bad,
                    (unsigned long long)n_seqgaps,
                    pkt_rate, mb_rate,
                    free_q.size(), num_bufs);
                t_prev_stat  = t_now;
                n_data_prev  = n_data;
                bytes_prev   = bw;
                t_stat       = t_now + 5.0;
            }
        }
    }

    // ── Flush remaining data ──────────────────────────────────────────────────
    if (cur && cur->used > 0)
        write_q.push(cur);
    else if (cur)
        free_q.push(cur);   // empty buffer — just return it

    // Send nullptr sentinel to stop the writer thread, then wait for it.
    write_q.push(nullptr);
    free_q.wake_all();      // unblock pop_interruptible() if it's sleeping
    writer.join();

    close(sock);
    close(fd);

    // ── Final summary ─────────────────────────────────────────────────────────
    double t_end        = monotonic_s();
    double wall_elapsed = t_end - t_launch;                        // total time including wait
    double cap_elapsed  = first_pkt ? (t_end - t_start) : 0.0;   // time from first packet
    double mib_out      = (double)wctx.bytes_written.load() / (1024.0 * 1024.0);
    double mb_s         = (cap_elapsed > 0.0) ? mib_out / cap_elapsed : 0.0;

    fprintf(stderr, "\n── Capture complete ──────────────────────────────────────────\n");
    if (!first_pkt) {
        fprintf(stderr, "  No data packets received.\n");
        fprintf(stderr, "  Total wall time    : %.3f s\n", wall_elapsed);
    } else {
        fprintf(stderr, "  Wait for 1st pkt   : %.3f s\n", t_start - t_launch);
        fprintf(stderr, "  Capture duration   : %.3f s\n", cap_elapsed);
    }
    fprintf(stderr, "  Data packets       : %llu\n",
            (unsigned long long)n_data);
    fprintf(stderr, "  Context packets    : %llu\n",
            (unsigned long long)n_ctx);
    fprintf(stderr, "  Malformed/dropped  : %llu\n",
            (unsigned long long)n_bad);
    fprintf(stderr, "  Seq gaps detected  : %llu  (~%llu samples lost)\n",
            (unsigned long long)n_seqgaps,
            (unsigned long long)n_seqlost);
    fprintf(stderr, "  Written to disk    : %.2f MiB\n", mib_out);
    fprintf(stderr, "  Avg write rate     : %.1f MB/s\n", mb_s);
    fprintf(stderr, "  File format        : %s\n",
            a.vita49 ? "VITA-49.2 packets (header + payload)"
                     : "Raw payload (int16 samples, no headers)");
    if (wctx.write_errors.load() > 0)
        fprintf(stderr, "  Write errors       : %d\n",
                wctx.write_errors.load());
    fprintf(stderr, "  Output             : %s\n", a.outfile.c_str());

    bool ok = (n_seqgaps == 0 && wctx.write_errors.load() == 0);
    fprintf(stderr, "  Result             : %s\n",
            ok ? "PASS ✓  No packets lost." : "FAIL ✗  Gaps detected.");
    return ok ? 0 : 1;
}
