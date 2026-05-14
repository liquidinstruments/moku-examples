/*
 * send_it_buffer.cpp  —  DIFI UDP transmitter for Moku:Delta Gigabit Streamer
 *
 * Complete rewrite based on:
 *   • Moku:Delta App Note "From Setup to Streaming: Moku:Delta Gigabit Streamer Guide"
 *   • send_it_repeat_buffer.py (working Python reference)
 *
 * Sends a raw payload binary file (captured by captureDataBuffer.py) back to
 * the Moku:Delta as a properly-formed DIFI/VITA-49.2 stream at the specified
 * sample rate.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o send_it send_it_buffer.cpp -lpthread -lm
 *
 * ── Standard mode (clock_nanosleep + sendto): ─────────────────────────────────
 *   sudo ./send_it --file capture.bin --fs 20833333.333 --dest 10.10.1.1 --rt --cpu 3
 *
 * ── SO_TXTIME mode (NIC hardware fires each packet at the exact right time): ──
 *   Requires one-time qdisc setup (software ETF — no PHC sync needed):
 *     sudo tc qdisc replace dev enp5s0f0np0 root etf clockid CLOCK_TAI delta 500000
 *   Then run:
 *     sudo ./send_it --file capture.bin --fs 20833333.333 --dest 10.10.1.1 \
 *                    --txtime --rt --cpu 3
 *
 *   For hardware-offloaded ETF (requires PHC→CLOCK_TAI sync, best precision):
 *     sudo phc2sys -s CLOCK_REALTIME -c /dev/ptp0 -O -37 -m &
 *     sudo tc qdisc replace dev enp5s0f0np0 root etf clockid CLOCK_TAI delta 200000 offload
 *
 *   To restore default qdisc afterwards:
 *     sudo tc qdisc delete dev enp5s0f0np0 root
 *
 * Flags:
 *   --file PATH          Raw binary payload file from captureDataBuffer.py
 *   --fs RATE            Sample rate in Hz  (e.g. 20833333.333 or 312.5e6)
 *   --dest IP            Moku IP address (Local IP configured in Moku App)
 *   --port N             Destination UDP port (default 4991)
 *   --stream-id 0xN      DIFI stream ID (default 0x00000001)
 *   --loops N            Repeat count; 0 = infinite (default 1)
 *   --bits N             Sample bit width: 16 (Normal) or 32 (Precision). Default 16.
 *   --channels N         Number of channels: 1 or 2 (default 2)
 *   --context-every N    Re-send context packet every N data packets (default 0 = only at start)
 *   --socket-buf N       SO_SNDBUF in bytes (default 0 = OS default)
 *   --max-catchup-ms F   Re-anchor threshold in ms (default 2.0).
 *   --txtime             Use SO_TXTIME hardware TX scheduling (requires ETF qdisc)
 *   --txtime-delta-us N  Lead time in µs for SO_TXTIME submissions (default 500).
 *                        Must be >= the 'delta' value set in the tc qdisc command.
 *   --vita49             Input file contains full VITA-49.2 packets (header +
 *                        payload) as written by capture_it --vita49.  The 28-byte
 *                        VITA header is stripped from each 1472-byte record before
 *                        transmission so the existing transmit loop is unchanged.
 *   --verbose            Print statistics
 *   --rt                 Enable SCHED_FIFO real-time scheduling (requires root)
 *   --rt-priority N      SCHED_FIFO priority 1-99 (default 80)
 *   --cpu N              Pin to CPU core N (default -1 = no pinning)
 */

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

// ── SO_TXTIME / SCM_TXTIME definitions ───────────────────────────────────────
// Available in <linux/net_tstamp.h> on kernel 4.19+. We define fallbacks so
// the file compiles even if the header isn't in the include path.
#ifdef __has_include
#  if __has_include(<linux/net_tstamp.h>)
#    include <linux/net_tstamp.h>
#  endif
#endif
#ifndef SO_TXTIME
#  define SO_TXTIME 61
#endif
#ifndef SCM_TXTIME
#  define SCM_TXTIME SO_TXTIME
#endif
// Provide a fallback struct only when the kernel header wasn't included.
// The guard macro _NET_TIMESTAMPING_H is set by <linux/net_tstamp.h>.
#ifndef _NET_TIMESTAMPING_H
struct sock_txtime { clockid_t clockid; uint32_t flags; };
#endif

// ─────────────────────────────────────────────────────────────────────────────
// DIFI / VITA-49.2 packet constants
// Verified against app note Figure 4/5 and send_it_repeat_buffer.py
// ─────────────────────────────────────────────────────────────────────────────

// Stream class identifiers
static constexpr uint32_t DIFI_OUI         = 0x006A621E;   // 24-bit OUI, padded to 32

// Data packet header Word 1 base (type=1, C=1, TSI=0, TSF flags = 0xE<<20)
// Matches Python: (1<<28)|(1<<27)|(0<<24)|(0xE<<20)
static constexpr uint32_t DATA_W0_BASE     = (1u<<28)|(1u<<27)|(0u<<24)|(0xEu<<20);  // 0x18E00000

// Context packet header Word 1 base (type=4, C=1, TSI=3, TSF flags = 0xE<<20)
// Matches Python: (4<<28)|(1<<27)|(0b011<<24)|(0xE<<20)
static constexpr uint32_t CTX_W0_BASE      = (4u<<28)|(1u<<27)|(3u<<24)|(0xEu<<20);  // 0x4BE00000

// Context Indicator Field — all fields present, "no change" indicator
// Per app note Figure 4, Word 8
static constexpr uint32_t CIF0             = 0xFBB98000u;

// Context packet is always exactly 27 words = 108 bytes
static constexpr int CTX_WORDS             = 27;
static constexpr int CTX_BYTES             = CTX_WORDS * 4;

// Data packet header is 7 words = 28 bytes
static constexpr int DATA_HEADER_WORDS     = 7;
static constexpr int DATA_HEADER_BYTES     = DATA_HEADER_WORDS * 4;  // 28

// MTU-based packet sizing (MTU 1500, IP=20, UDP=8, VITA header=28)
static constexpr int MTU                   = 1500;
static constexpr int IP_UDP_OVERHEAD       = 28;           // 20 IP + 8 UDP
static constexpr int MAX_PAYLOAD           = MTU - IP_UDP_OVERHEAD - DATA_HEADER_BYTES;  // 1444
// Payload must be a whole number of 32-bit words
static constexpr int PAYLOAD_BYTES         = (MAX_PAYLOAD / 4) * 4;     // 1444
static constexpr int SAMPLE_PAIRS_PER_PKT  = PAYLOAD_BYTES / 4;         // 361
static constexpr int DATA_PKT_BYTES        = DATA_HEADER_BYTES + PAYLOAD_BYTES;  // 1472
static constexpr int DATA_PKT_WORDS        = DATA_PKT_BYTES / 4;         // 368

// Sleep headroom: the send loop sleeps until (target - SLEEP_HEADROOM_NS) then
// busy-waits the remaining time.  This decouples the imprecise nanosleep from
// the actual sendto call, giving ±30 ns timing instead of ±1-5 µs.
// Capped at interval/3 so we always sleep at least 2/3 of the interval.
static constexpr int64_t SLEEP_HEADROOM_NS      = 5'000LL;      // 5 µs default

// Max-catchup is auto-scaled in main() to 10 × interval_ns.
// At 44.643 MSa/s that's 80 µs (10 pkts); at 20.833 MSa/s it's 173 µs (10 pkts).
// User can override with --max-catchup-ms.  A value of 0 means "auto".
static constexpr int64_t AUTO_CATCHUP_PKTS      = 10;           // packets per re-anchor

// ─────────────────────────────────────────────────────────────────────────────
// Big-endian write helpers (no alignment requirement)
// ─────────────────────────────────────────────────────────────────────────────
static inline void be32(uint8_t* p, uint32_t v) {
    p[0] = (v >> 24) & 0xFF;  p[1] = (v >> 16) & 0xFF;
    p[2] = (v >>  8) & 0xFF;  p[3] =  v        & 0xFF;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI Arguments
// ─────────────────────────────────────────────────────────────────────────────
struct Args {
    std::string file;
    double      fs             = 0.0;
    std::string dest           = "";
    int         port           = 4991;
    uint32_t    stream_id      = 0x00000001;
    int         loops          = 1;
    int         bits           = 16;
    int         channels       = 2;
    int         context_every  = 0;    // 0 = only at stream start
    int         socket_buf     = 0;
    double      max_catchup_ms = 0.0;   // 0 = auto (10 × interval_ns)
    bool        vita49         = false;   // input file has VITA-49.2 headers (strip on load)
    bool        txtime         = false;   // use SO_TXTIME hardware scheduling
    int64_t     txtime_delta_us = 500;    // lead time for SO_TXTIME submissions (µs)
    bool        verbose        = false;
    bool        rt             = false;
    int         rt_priority    = 80;
    int         cpu            = -1;
};

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --file PATH --fs RATE --dest IP [options]\n"
        "  --file PATH         Raw payload binary (from captureDataBuffer.py)\n"
        "  --fs RATE           Sample rate Hz (e.g. 20833333.333 or 312.5e6)\n"
        "  --dest IP           Moku IP address\n"
        "  --port N            UDP port (default 4991)\n"
        "  --stream-id 0xN     DIFI stream ID (default 0x00000001)\n"
        "  --loops N           Repeat count; 0=infinite (default 1)\n"
        "  --bits N            16=Normal mode, 32=Precision mode (default 16)\n"
        "  --channels N        1 or 2 channels (default 2)\n"
        "  --context-every N   Re-send context every N data pkts (default 0 = only at start)\n"
        "  --socket-buf N      SO_SNDBUF bytes (default 0 = OS default)\n"
        "  --max-catchup-ms F  Re-anchor threshold ms (default 0 = auto: 10 x interval)\n"
        "  --txtime            Use SO_TXTIME hardware TX scheduling\n"
        "                      Requires: sudo tc qdisc replace dev <iface> root etf\n"
        "                                         clockid CLOCK_TAI delta 500000\n"
        "  --txtime-delta-us N Lead time µs for SO_TXTIME (default 500, must >= tc delta)\n"
        "  --vita49            Input file has VITA-49.2 headers (capture_it --vita49 output).\n"
        "                      Strips the 28-byte header from each 1472-byte record.\n"
        "  --verbose           Print statistics\n"
        "  --rt                Enable SCHED_FIFO (requires root)\n"
        "  --rt-priority N     SCHED_FIFO priority 1-99 (default 80)\n"
        "  --cpu N             Pin to CPU N (default -1 = no pin)\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nxt = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); exit(1); }
            return argv[i];
        };
        if      (s == "--file")           a.file           = nxt();
        else if (s == "--fs")             a.fs             = std::stod(nxt());
        else if (s == "--dest")           a.dest           = nxt();
        else if (s == "--port")           a.port           = std::stoi(nxt());
        else if (s == "--stream-id")      a.stream_id      = (uint32_t)std::stoul(nxt(), nullptr, 0);
        else if (s == "--loops")          a.loops          = std::stoi(nxt());
        else if (s == "--bits")           a.bits           = std::stoi(nxt());
        else if (s == "--channels")       a.channels       = std::stoi(nxt());
        else if (s == "--context-every")  a.context_every  = std::stoi(nxt());
        else if (s == "--socket-buf")      a.socket_buf      = std::stoi(nxt());
        else if (s == "--max-catchup-ms") a.max_catchup_ms  = std::stod(nxt());
        else if (s == "--vita49")         a.vita49          = true;
        else if (s == "--txtime")         a.txtime          = true;
        else if (s == "--txtime-delta-us") a.txtime_delta_us = std::stoll(nxt());
        else if (s == "--verbose")        a.verbose         = true;
        else if (s == "--rt")             a.rt             = true;
        else if (s == "--rt-priority")    a.rt_priority    = std::stoi(nxt());
        else if (s == "--cpu")            a.cpu            = std::stoi(nxt());
        else if (s == "--help" || s == "-h") { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", s.c_str()); usage(argv[0]); exit(1); }
    }
    if (a.file.empty())   { fprintf(stderr, "--file is required\n");       usage(argv[0]); exit(1); }
    if (a.fs <= 0.0)      { fprintf(stderr, "--fs is required\n");         usage(argv[0]); exit(1); }
    if (a.dest.empty())   { fprintf(stderr, "--dest is required\n");       usage(argv[0]); exit(1); }
    if (a.bits != 16 && a.bits != 32) { fprintf(stderr, "--bits must be 16 or 32\n"); exit(1); }
    if (a.channels < 1 || a.channels > 4) { fprintf(stderr, "--channels must be 1-4\n"); exit(1); }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Real-time scheduling + CPU affinity
// ─────────────────────────────────────────────────────────────────────────────
static void setup_realtime(const Args& a) {
    if (a.rt) {
        struct sched_param sp{};
        sp.sched_priority = std::max(1, std::min(99, a.rt_priority));
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
            perror("sched_setscheduler (need root for --rt)");
        else
            fprintf(stderr, "SCHED_FIFO priority %d enabled.\n", sp.sched_priority);
    }
    if (a.cpu >= 0) {
        cpu_set_t cs;  CPU_ZERO(&cs);  CPU_SET((size_t)a.cpu, &cs);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
            perror("pthread_setaffinity_np");
        else
            fprintf(stderr, "Pinned to CPU %d.\n", a.cpu);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timestamp arithmetic
// ─────────────────────────────────────────────────────────────────────────────
// DIFI fractional timestamp: 64-bit integer in units of 1/2^64 seconds.
// We compute the per-packet increment as a fixed integer to avoid drift.

struct Timestamp {
    uint32_t int_sec;
    uint32_t frac_hi;   // high 32 bits of 64-bit fractional
    uint32_t frac_lo;   // low  32 bits
};

static uint64_t ticks_per_packet(double fs) {
    // SAMPLE_PAIRS_PER_PKT / fs * 2^64  (as double, then round)
    return (uint64_t)llround((double)SAMPLE_PAIRS_PER_PKT / fs * std::ldexp(1.0, 64));
}

static void advance_ts(Timestamp& ts, uint64_t tpp) {
    uint64_t old_frac = ((uint64_t)ts.frac_hi << 32) | ts.frac_lo;
    uint64_t new_frac = old_frac + tpp;
    if (new_frac < old_frac) ts.int_sec++;   // carry into integer seconds
    ts.frac_hi = (uint32_t)(new_frac >> 32);
    ts.frac_lo = (uint32_t)(new_frac & 0xFFFFFFFFu);
}

// ─────────────────────────────────────────────────────────────────────────────
// Packet builders
// All values confirmed against send_it_repeat_buffer.py and app note Figure 4/5
// ─────────────────────────────────────────────────────────────────────────────

// Build the 7-word data packet header into pkt[0..27].
static void stamp_data_header(uint8_t* pkt,
                               uint32_t stream_id,
                               const Timestamp& ts,
                               uint8_t  pkt_count) {
    // Word 1: type=1, C=1, TSI=0, TSF-flags=0xE<<20, count[19:16], size[15:0]
    be32(pkt +  0, DATA_W0_BASE | ((uint32_t)(pkt_count & 0xF) << 16) | DATA_PKT_WORDS);
    // Word 2: Stream Identifier
    be32(pkt +  4, stream_id);
    // Word 3: 24-bit DIFI OUI (pad bits 31:24 = 0)
    be32(pkt +  8, DIFI_OUI & 0x00FFFFFFu);
    // Word 4: Information Class = 0x0000, Packet Class = 0x0000
    be32(pkt + 12, 0x00000000u);
    // Word 5: Integer Seconds Timestamp
    be32(pkt + 16, ts.int_sec);
    // Word 6: Fractional Seconds Timestamp (high 32 bits)
    be32(pkt + 20, ts.frac_hi);
    // Word 7: Fractional Seconds Timestamp (low 32 bits)
    be32(pkt + 24, ts.frac_lo);
    // Words 8+: Signal Data Payload follows (written separately)
}

// Build the full 27-word context packet into buf[0..107].
// sample_rate_hz: Hz as integer (round if non-integer)
// vector_size:    channels - 1  (0 = mono, 1 = stereo, etc.)
// item_bits:      sample bit width (16 for Normal, 32 for Precision)
static void build_context_packet(uint8_t* buf,
                                  uint32_t stream_id,
                                  const Timestamp& ts,
                                  uint8_t  pkt_count,
                                  int64_t  sample_rate_hz,
                                  int      vector_size,
                                  int      item_bits) {
    // Sample Rate: 64-bit fixed-point, integer Hz × 2^20
    uint64_t sr_fp = (uint64_t)llround((double)sample_rate_hz * (double)(1ull << 20));

    // Data Packet Payload Format word 26:
    //   bit 31 = 1 (processing-efficient packing)
    //   bits 11:6 = Item Packing Field Size = (item_bits - 1)
    //   bits 5:0  = Data Item Size          = (item_bits - 1)
    // Verified against Python: 0x800003CF for 16-bit (IPFS=15, DIS=15)
    int ipfs = item_bits - 1;
    uint32_t w26 = (1u << 31) | ((uint32_t)ipfs << 6) | (uint32_t)ipfs;

    // ── Word 1: context header ──────────────────────────────────────────────
    be32(buf +   0, CTX_W0_BASE | ((uint32_t)(pkt_count & 0xF) << 16) | CTX_WORDS);
    // Word 2: Stream Identifier
    be32(buf +   4, stream_id);
    // Word 3: 24-bit DIFI OUI
    be32(buf +   8, DIFI_OUI & 0x00FFFFFFu);
    // Word 4: Information Class = 0x0000, Packet Class = 0x0001
    be32(buf +  12, 0x00000001u);
    // Word 5: Integer Seconds Timestamp
    be32(buf +  16, ts.int_sec);
    // Word 6: Fractional Timestamp high
    be32(buf +  20, ts.frac_hi);
    // Word 7: Fractional Timestamp low
    be32(buf +  24, ts.frac_lo);
    // Word 8: Context Indicator Field (CIF0)
    be32(buf +  28, CIF0);
    // Word 9: Reference Point = 0
    be32(buf +  32, 0x00000000u);
    // Words 10-11: Bandwidth = 0
    be32(buf +  36, 0x00000000u);  be32(buf +  40, 0x00000000u);
    // Words 12-13: IF Reference Frequency = 0
    be32(buf +  44, 0x00000000u);  be32(buf +  48, 0x00000000u);
    // Words 14-15: RF Reference Frequency = 0
    be32(buf +  52, 0x00000000u);  be32(buf +  56, 0x00000000u);
    // Words 16-17: IF Band Offset = 0
    be32(buf +  60, 0x00000000u);  be32(buf +  64, 0x00000000u);
    // Word 18: Scaling = 0, Reference Level = 0
    be32(buf +  68, 0x00000000u);
    // Word 19: Gain 1 = 0, Gain 2 = 0
    be32(buf +  72, 0x00000000u);
    // Words 20-21: Sample Rate (64-bit fixed-point)
    be32(buf +  76, (uint32_t)(sr_fp >> 32));
    be32(buf +  80, (uint32_t)(sr_fp & 0xFFFFFFFFu));
    // Words 22-23: Timestamp Adjustment = 0
    be32(buf +  84, 0x00000000u);  be32(buf +  88, 0x00000000u);
    // Word 24: Timestamp Calibration Time = 0
    be32(buf +  92, 0x00000000u);
    // Word 25: State and Event Indicators = 0
    be32(buf +  96, 0x00000000u);
    // Word 26: Item Packing Field Size + Data Item Size
    be32(buf + 100, w26);
    // Word 27: Vector Size (channels - 1)
    be32(buf + 104, (uint32_t)vector_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// File loading
// ─────────────────────────────────────────────────────────────────────────────
// Raw mode (vita49=false):
//   File contains packed payload bytes with no headers.  Captured by
//   captureDataBuffer.py or capture_it without --vita49.  Payload is
//   little-endian 16-bit samples interleaved CH1[n], CH2[n], ...
//
// VITA-49.2 mode (vita49=true):
//   File contains full 1472-byte VITA-49.2 packets (28-byte header + 1444-byte
//   payload) as written by capture_it --vita49.  We strip the header from each
//   record so the transmit loop receives the same packed-payload layout.

static std::vector<uint8_t> load_payloads(const std::string& path, bool vita49) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s: %s\n", path.c_str(), strerror(errno)); exit(1); }
    auto sz = (size_t)f.tellg();
    if (sz == 0) { fprintf(stderr, "File is empty: %s\n", path.c_str()); exit(1); }
    f.seekg(0);

    if (vita49) {
        // ── VITA-49.2 mode: strip 28-byte header from every 1472-byte record ──
        size_t n_records = sz / DATA_PKT_BYTES;
        if (n_records == 0) {
            fprintf(stderr,
                "File too small for VITA-49.2 mode: need at least %d bytes "
                "for one complete packet (header + payload).\n", DATA_PKT_BYTES);
            exit(1);
        }
        size_t tail = sz % DATA_PKT_BYTES;
        if (tail != 0)
            fprintf(stderr,
                "Note: %zu trailing byte(s) ignored (not a whole VITA-49.2 record).\n",
                tail);

        // Read only the complete records
        size_t read_bytes = n_records * (size_t)DATA_PKT_BYTES;
        std::vector<uint8_t> raw(read_bytes);
        if (!f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)read_bytes)) {
            fprintf(stderr, "Read error: %s\n", path.c_str()); exit(1);
        }

        // Extract payloads, skipping the 28-byte VITA header in each record
        std::vector<uint8_t> buf(n_records * (size_t)PAYLOAD_BYTES);
        for (size_t i = 0; i < n_records; ++i) {
            memcpy(buf.data()  + i * (size_t)PAYLOAD_BYTES,
                   raw.data()  + i * (size_t)DATA_PKT_BYTES + DATA_HEADER_BYTES,
                   (size_t)PAYLOAD_BYTES);
        }
        fprintf(stderr,
            "Loaded %.2f MiB  (%zu VITA-49.2 packets, %d-byte headers stripped)\n",
            (double)sz / (1024.0 * 1024.0), n_records, DATA_HEADER_BYTES);
        return buf;

    } else {
        // ── Raw mode: packed payloads with no headers ─────────────────────────
        std::vector<uint8_t> buf(sz);
        if (!f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)sz)) {
            fprintf(stderr, "Read error: %s\n", path.c_str()); exit(1);
        }
        size_t aligned = (sz / (size_t)PAYLOAD_BYTES) * (size_t)PAYLOAD_BYTES;
        if (aligned == 0) {
            fprintf(stderr,
                "File too small: need at least %d bytes for one packet payload.\n",
                PAYLOAD_BYTES);
            exit(1);
        }
        if (aligned < sz)
            fprintf(stderr,
                "Note: trimmed %zu tail bytes to align to %d-byte payloads.\n",
                sz - aligned, PAYLOAD_BYTES);
        buf.resize(aligned);
        fprintf(stderr, "Loaded %.2f MiB  (%zu packets)\n",
                (double)sz / (1024.0 * 1024.0), aligned / (size_t)PAYLOAD_BYTES);
        return buf;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SO_TXTIME helper
// ─────────────────────────────────────────────────────────────────────────────
// Sends one UDP datagram with a hardware TX timestamp attached.
// The ETF qdisc holds the packet and fires it (via kernel or NIC hardware) at
// exactly txtime_tai_ns nanoseconds on the CLOCK_TAI clock.
// txtime_tai_ns must be at least (now_TAI + qdisc_delta) in the future.

static int sendmsg_txtime(int sock, const void* data, size_t len,
                           const struct sockaddr_in* dest_addr,
                           uint64_t txtime_tai_ns) {
    char ctrl[CMSG_SPACE(sizeof(uint64_t))];
    memset(ctrl, 0, sizeof(ctrl));

    struct iovec  iov = { const_cast<void*>(data), len };
    struct msghdr msg = {};
    msg.msg_name       = const_cast<struct sockaddr_in*>(dest_addr);
    msg.msg_namelen    = sizeof(*dest_addr);
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = ctrl;
    msg.msg_controllen = sizeof(ctrl);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_TXTIME;
    cmsg->cmsg_len   = CMSG_LEN(sizeof(uint64_t));
    memcpy(CMSG_DATA(cmsg), &txtime_tai_ns, sizeof(uint64_t));

    return (int)sendmsg(sock, &msg, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing helpers
// Uses clock_nanosleep(TIMER_ABSTIME) — self-correcting absolute time pacing.
// If the OS wakes us late, the next deadline is still at the right absolute
// time, so the average rate is always correct regardless of jitter.
// ─────────────────────────────────────────────────────────────────────────────
static inline void ts_add_ns(struct timespec& ts, int64_t ns) {
    ts.tv_nsec += (long)ns;
    while (ts.tv_nsec >= 1'000'000'000L) { ts.tv_nsec -= 1'000'000'000L; ts.tv_sec++; }
    while (ts.tv_nsec <  0L)             { ts.tv_nsec += 1'000'000'000L; ts.tv_sec--; }
}

static inline int64_t ts_diff_ns(const struct timespec& a, const struct timespec& b) {
    // returns a - b in nanoseconds
    return (int64_t)(a.tv_sec - b.tv_sec) * 1'000'000'000LL + (a.tv_nsec - b.tv_nsec);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    setup_realtime(a);

    // Lock all current and future memory pages into RAM.
    // Prevents page faults during the send loop that could cause multi-ms stalls
    // and large underflow bursts on the Moku side.
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        perror("mlockall (non-fatal — run as root for full effect)");
    else
        fprintf(stderr, "Memory locked (mlockall).\n");

    // Load payload data into RAM
    std::vector<uint8_t> payloads = load_payloads(a.file, a.vita49);
    size_t n_pkts = payloads.size() / PAYLOAD_BYTES;

    // Pre-fault every page of the payload buffer now, before the RT send loop
    // starts.  mlockall prevents future eviction; this touch ensures the pages
    // are actually resident and mapped before we need them.
    {
        volatile uint8_t sink = 0;
        for (size_t i = 0; i < payloads.size(); i += 4096)
            sink ^= payloads[i];
        (void)sink;
        fprintf(stderr, "Payload pages pre-faulted.\n");
    }

    // Pre-allocate one data packet buffer (we stamp and copy payload into it)
    std::vector<uint8_t> pkt_buf(DATA_PKT_BYTES);

    // Context packet buffer
    std::vector<uint8_t> ctx_buf(CTX_BYTES);

    // Derived timing parameters
    double   interval_s  = (double)SAMPLE_PAIRS_PER_PKT / a.fs;
    int64_t  interval_ns = (int64_t)llround(interval_s * 1e9);
    uint64_t tpp         = ticks_per_packet(a.fs);
    int64_t  sr_hz       = (int64_t)llround(a.fs);
    int      vec_size    = a.channels - 1;

    // Auto-scale max catch-up: 10 packet intervals, unless the user overrides.
    // Keeps overflow bursts at most 10 packets regardless of sample rate.
    // At 20.833 MSa/s: 173 µs.  At 44.643 MSa/s: 81 µs.  At 312.5 MSa/s: 12 µs.
    int64_t max_catch = (a.max_catchup_ms > 0.0)
                        ? (int64_t)llround(a.max_catchup_ms * 1e6)
                        : AUTO_CATCHUP_PKTS * interval_ns;

    // Adaptive sleep headroom for the hybrid sleep+busy-wait pacing.
    // We sleep until (target - headroom) then busy-wait to the exact target.
    // headroom = min(SLEEP_HEADROOM_NS, interval/3) — always sleep at least 2/3.
    int64_t headroom_ns = std::min(SLEEP_HEADROOM_NS, interval_ns / 3);
    headroom_ns = std::max(headroom_ns, (int64_t)100);  // never less than 100 ns

    bool needs_hwetf = (interval_ns < 5'000LL);  // < 5 µs needs hardware ETF

    fprintf(stderr,
            "Config:\n"
            "  File format     : %s\n"
            "  File packets    : %zu\n"
            "  Sample rate     : %.6f MSa/s\n"
            "  Packet interval : %.3f µs  (%lld ns)\n"
            "  Packet size     : %d bytes  (%d payload bytes, %d sample pairs)\n"
            "  Bit depth       : %d-bit  (Normal mode)\n"
            "  Channels        : %d  (vector_size=%d)\n"
            "  Loops           : %s\n"
            "  Max catch-up    : %.1f µs  (%lld pkts)\n"
            "  Sleep headroom  : %.1f µs  (busy-wait final stage)\n"
            "  Dest            : %s:%d\n",
            a.vita49 ? "VITA-49.2 (headers stripped on load)" : "raw payload (no headers)",
            n_pkts,
            a.fs / 1e6,
            interval_s * 1e6,
            (long long)interval_ns,
            DATA_PKT_BYTES, PAYLOAD_BYTES, SAMPLE_PAIRS_PER_PKT,
            a.bits,
            a.channels, vec_size,
            a.loops == 0 ? "infinite" : std::to_string(a.loops).c_str(),
            (double)max_catch / 1e3,
            (long long)(max_catch / interval_ns),
            (double)headroom_ns / 1e3,
            a.dest.c_str(), a.port);

    if (needs_hwetf && !a.txtime)
        fprintf(stderr,
            "WARNING: interval %.1f µs < 5 µs — hardware ETF offload strongly\n"
            "         recommended at this rate (--txtime with 'offload' qdisc).\n",
            interval_s * 1e6);

    // UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    if (a.socket_buf > 0)
        setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &a.socket_buf, sizeof(a.socket_buf));

    // SO_TXTIME: enable hardware/kernel TX scheduling via the ETF qdisc.
    // The NIC fires each packet at the exact CLOCK_TAI time attached to it,
    // decoupling packet timing from host CPU scheduling jitter entirely.
    if (a.txtime) {
        struct sock_txtime skt = {};
        skt.clockid = CLOCK_TAI;
        skt.flags   = 0;   // 0 = deadline mode off; drop silently if late
        if (setsockopt(sock, SOL_SOCKET, SO_TXTIME, &skt, sizeof(skt)) < 0) {
            perror("setsockopt SO_TXTIME");
            fprintf(stderr,
                "Hint: set up the ETF qdisc first:\n"
                "  sudo tc qdisc replace dev <iface> root etf "
                "clockid CLOCK_TAI delta 500000\n");
            return 1;
        }
        fprintf(stderr, "SO_TXTIME enabled  (CLOCK_TAI, delta=%lld µs)\n",
                (long long)a.txtime_delta_us);
    }

    struct sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port   = htons((uint16_t)a.port);
    if (inet_pton(AF_INET, a.dest.c_str(), &dest.sin_addr) != 1) {
        fprintf(stderr, "Bad IP address: %s\n", a.dest.c_str()); return 1;
    }

    // Initial timestamp: integer seconds = 0 (epoch = instrument deploy/reset),
    // fractional = 0.  The Moku only checks continuity between packets, not
    // the absolute epoch value.
    Timestamp ts{ 0, 0, 0 };
    uint8_t pkt_count = 0;

    // ── Send initial context packet ───────────────────────────────────────────
    build_context_packet(ctx_buf.data(), a.stream_id, ts, pkt_count,
                         sr_hz, vec_size, a.bits);
    sendto(sock, ctx_buf.data(), CTX_BYTES, 0,
           (struct sockaddr*)&dest, sizeof(dest));
    pkt_count = (pkt_count + 1) & 0xF;
    fprintf(stderr, "Context packet sent  stream_id=0x%08X  sample_rate=%.6f MSa/s\n",
            a.stream_id, a.fs / 1e6);

    // ── Main transmit loop ────────────────────────────────────────────────────
    // clock_nanosleep with TIMER_ABSTIME: the process sleeps until an absolute
    // clock time.  If the OS wakes us late, the NEXT target is still at the
    // correct absolute time → the long-run average rate is always exact.
    // No sleep headroom hacks needed.

    struct timespec send_target;
    clock_gettime(CLOCK_MONOTONIC, &send_target);

    int64_t  loop_count      = 0;
    uint64_t total_pkts_sent = 0;
    struct timespec stat_ts  = send_target;
    ts_add_ns(stat_ts, 5'000'000'000LL);

    fprintf(stderr, "Transmitting...  (Ctrl-C to stop)\n");

    int context_countdown = (a.context_every > 0) ? a.context_every : -1;

    // ── SO_TXTIME mode: initialise CLOCK_TAI baseline ────────────────────────
    // txtime_tai_ns tracks the CLOCK_TAI nanoseconds at which the CURRENT packet
    // should leave the NIC wire.  We submit each packet exactly txtime_delta_ns
    // ahead of that time so the ETF qdisc has time to queue it.
    // The NIC/kernel fires the packet at txtime_tai_ns regardless of any host
    // CPU jitter that happens after we call sendmsg().
    uint64_t txtime_tai_ns  = 0;
    int64_t  txtime_delta_ns = a.txtime_delta_us * 1'000LL;
    if (a.txtime) {
        struct timespec tai_now;
        clock_gettime(CLOCK_TAI, &tai_now);
        // Start 2× the delta in the future so the first packet is always valid
        txtime_tai_ns = (uint64_t)tai_now.tv_sec * 1'000'000'000ULL
                        + (uint64_t)tai_now.tv_nsec
                        + (uint64_t)(txtime_delta_ns * 2);
    }

    while (a.loops == 0 || loop_count < a.loops) {
        for (size_t i = 0; i < n_pkts; ++i) {

            // ── Optional periodic context re-send ─────────────────────────
            if (a.context_every > 0 && context_countdown <= 0) {
                build_context_packet(ctx_buf.data(), a.stream_id, ts, pkt_count,
                                     sr_hz, vec_size, a.bits);
                sendto(sock, ctx_buf.data(), CTX_BYTES, 0,
                       (struct sockaddr*)&dest, sizeof(dest));
                pkt_count = (pkt_count + 1) & 0xF;
                context_countdown = a.context_every;
            }
            if (a.context_every > 0) --context_countdown;

            // ── Stamp header + copy payload ────────────────────────────────
            stamp_data_header(pkt_buf.data(), a.stream_id, ts, pkt_count);
            memcpy(pkt_buf.data() + DATA_HEADER_BYTES,
                   payloads.data() + i * PAYLOAD_BYTES,
                   PAYLOAD_BYTES);

            if (!a.txtime) {
                // ══ Standard mode: hybrid sleep + busy-wait + sendto ══════
                //
                // Phase 1 — sleep until (send_target - headroom_ns).
                //   TIMER_ABSTIME is self-correcting: if a previous packet ran
                //   late, the next sleep is shorter, keeping long-run rate exact.
                //
                // Phase 2 — busy-wait the remaining headroom_ns by spinning on
                //   clock_gettime until we reach send_target.  This decouples
                //   the imprecise nanosleep wakeup (±1–5 µs) from the actual
                //   sendto call, achieving ±30 ns timing at any sample rate.
                //
                // This fixes the ±62% jitter at 44.643 MSa/s (8 µs interval)
                // that the sleep-only approach produced.

                struct timespec sleep_ts = send_target;
                ts_add_ns(sleep_ts, -headroom_ns);

                // Only call nanosleep if the sleep target is still in the future.
                struct timespec now_ts;
                clock_gettime(CLOCK_MONOTONIC, &now_ts);
                if (ts_diff_ns(now_ts, sleep_ts) < 0)  // now < sleep_ts
                    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &sleep_ts, nullptr);

                // Busy-wait to the exact send_target.
                do {
                    clock_gettime(CLOCK_MONOTONIC, &now_ts);
                } while (ts_diff_ns(now_ts, send_target) < 0);  // while now < target

                sendto(sock, pkt_buf.data(), DATA_PKT_BYTES, 0,
                       (struct sockaddr*)&dest, sizeof(dest));

                ts_add_ns(send_target, interval_ns);

                // Re-anchor if we've fallen behind by more than max_catch.
                // Using the now_ts already measured in the busy-wait above.
                int64_t behind = ts_diff_ns(now_ts, send_target);
                if (behind > max_catch) {
                    send_target = now_ts;
                    ts_add_ns(send_target, interval_ns);
                }

            } else {
                // ══ SO_TXTIME mode: submit ahead with hardware TX timestamp ═
                //
                // 1. Sleep until (txtime - delta): we want to arrive at the
                //    ETF qdisc at least delta_ns before the TX time.
                // 2. sendmsg() attaches txtime_tai_ns to the packet.
                // 3. The ETF qdisc / NIC hardware fires the packet at that
                //    exact CLOCK_TAI nanosecond — independent of any CPU jitter
                //    that occurs after the sendmsg() call returns.
                //
                // Timing diagram:
                //   ──[stamp+memcpy]──[sleep]──[sendmsg]──…──[NIC fires packet]
                //                                ^delta^        ^ txtime_tai_ns

                uint64_t submit_tai_ns = txtime_tai_ns - (uint64_t)txtime_delta_ns;
                struct timespec submit_ts {
                    (time_t)(submit_tai_ns / 1'000'000'000ULL),
                    (long)  (submit_tai_ns % 1'000'000'000ULL)
                };
                clock_nanosleep(CLOCK_TAI, TIMER_ABSTIME, &submit_ts, nullptr);

                // Sanity check: if txtime_tai_ns has already passed (we were
                // preempted too long), the ETF qdisc will drop the packet.
                // Re-anchor so subsequent packets stay valid.
                struct timespec tai_now;
                clock_gettime(CLOCK_TAI, &tai_now);
                uint64_t now_tai = (uint64_t)tai_now.tv_sec * 1'000'000'000ULL
                                   + (uint64_t)tai_now.tv_nsec;
                if (now_tai >= txtime_tai_ns) {
                    // We're late — skip ahead to keep subsequent packets valid.
                    // This mirrors the re-anchor in standard mode.
                    txtime_tai_ns = now_tai + (uint64_t)txtime_delta_ns
                                  + (uint64_t)interval_ns;
                }

                sendmsg_txtime(sock, pkt_buf.data(), DATA_PKT_BYTES,
                               &dest, txtime_tai_ns);
                txtime_tai_ns += (uint64_t)interval_ns;
            }

            // ── Advance DIFI timestamp and packet counter ──────────────────
            advance_ts(ts, tpp);
            pkt_count = (pkt_count + 1) & 0xF;
            ++total_pkts_sent;
        }

        ++loop_count;

        if (a.verbose) {
            struct timespec now_ts;
            clock_gettime(CLOCK_MONOTONIC, &now_ts);
            if (ts_diff_ns(now_ts, stat_ts) >= 0) {
                fprintf(stderr, "Loop %lld  |  total pkts: %llu\n",
                        (long long)loop_count, (unsigned long long)total_pkts_sent);
                stat_ts = now_ts;
                ts_add_ns(stat_ts, 5'000'000'000LL);
            }
        }
    }

    fprintf(stderr, "Done.  Loops: %lld  Total packets: %llu\n",
            (long long)loop_count, (unsigned long long)total_pkts_sent);
    close(sock);
    return 0;
}
