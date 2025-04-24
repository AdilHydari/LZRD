// =============== rangecoder.cpp ===============
#include <vector>
#include <cstdint>
#include <stdexcept> // For errors
#include <cmath> // For std::log2 (if needed for real price calc)
#include <limits>     // For numeric_limits
#include <algorithm>  // For std::reverse, std::min
#include <string>    // For std::to_string in error messages
#include <cassert>   // For existing assert calls

#include "rangecoder.h"

namespace { // Anonymous namespace for helper structs and functions

// Structure to store optimal path information
struct Opt {
    int price = std::numeric_limits<int>::max(); // total price to reach this position
    uint16_t len = 0;  // length of token ending here (1 for literal, >= MIN_MATCH_LEN for match)
    uint32_t dist = 0; // distance if match, 0 otherwise

    // Default constructor initializes price to max
    Opt() = default;

    // Constructor for convenience
    Opt(int p, uint16_t l, uint32_t d) : price(p), len(l), dist(d) {}
};

// Structure to represent a candidate match
struct MatchCandidate {
    uint32_t dist;
    uint32_t len;
};

// --- Price Estimation ---

// Helper: Estimate bits for a symbol given freq/total (scaled)
const int PRICE_BITS_SHIFT = 8; // Scale factor for integer representation (e.g., 8 means cost is in units of 1/256 bits)

int estimate_bits(uint16_t freq, uint16_t total) {
    if (freq == 0 || total == 0 || freq > total) {
         // Invalid input or zero probability, assign effectively infinite cost
         // Divide by 2 to prevent overflow when summing multiple max costs
         return std::numeric_limits<int>::max() / 2;
    }
    // Avoid log2(0) or log2(negative) which are undefined or complex.
    // Freq is guaranteed > 0 here. Total is also > 0.

    // Calculate bits: -log2(probability) = log2(total) - log2(freq)
    // Use double for intermediate log2 calculation for precision
    double log2_freq = std::log2(static_cast<double>(freq));
    double log2_total = std::log2(static_cast<double>(total));
    double bits = log2_total - log2_freq;

    // Scale and convert to integer, ensuring non-negative result. Add 0.5 for rounding.
    int scaled_bits = static_cast<int>(bits * (1 << PRICE_BITS_SHIFT) + 0.5);

    // Ensure minimum cost is slightly > 0 to distinguish from impossible paths
    // and avoid issues if scaling results in 0 for high-probability symbols.
    return std::max(1, scaled_bits);
}

// Helper: Estimate bits for a varuint (using its byte model)
int estimate_varuint_bits(uint32_t value, const rangecoder::ModelO0<256>& model) {
    int total_cost = 0;
    int max_cost_check = std::numeric_limits<int>::max() / 2; // Precompute limit

    do {
        uint8_t byte = value & 0x7F; // Get lower 7 bits
        value >>= 7;
        if (value > 0) {
            byte |= 0x80; // Set continuation bit
        }

        // Estimate cost of this byte using the model
        int byte_cost = estimate_bits(model.freq[byte], model.total);

        // Check for potential overflow before adding
        if (byte_cost >= max_cost_check || total_cost > max_cost_check - byte_cost) {
             return max_cost_check; // Return max cost if overflow would occur or byte cost is max
        }
        total_cost += byte_cost;

    } while (value > 0);

    return total_cost;
}

// Get price for encoding a literal
int get_literal_price(
    uint8_t literal,
    uint8_t context,
    const rangecoder::ModelO0<2>& M_flag,
    const rangecoder::ModelO1& M_lit)
{
    int max_cost_check = std::numeric_limits<int>::max() / 2;

    // Cost of is_match=0 flag
    int flag_cost = estimate_bits(M_flag.freq[0], M_flag.total);
    if (flag_cost >= max_cost_check) return max_cost_check;

    // Cost of the literal itself using the context model
    int literal_cost = estimate_bits(M_lit.freq[context][literal], M_lit.cum[context][256]);
     if (literal_cost >= max_cost_check) return max_cost_check;

    // Return total cost, checking for overflow
    if (flag_cost > max_cost_check - literal_cost) {
        return max_cost_check;
    }
    return flag_cost + literal_cost;
}

// Get price for encoding a match
int get_match_price(
    uint32_t dist,
    uint32_t len,
    const rangecoder::ModelO0<2>& M_flag,
    const rangecoder::ModelO0<256>& M_dist,
    const rangecoder::ModelO0<256>& M_len)
{
     int max_cost_check = std::numeric_limits<int>::max() / 2;

    // Cost of is_match=1 flag
    int flag_cost = estimate_bits(M_flag.freq[1], M_flag.total);
    if (flag_cost >= max_cost_check) return max_cost_check;

    // Cost of distance (dist - 1)
    uint32_t dist_val = dist - 1;
    int dist_cost = estimate_varuint_bits(dist_val, M_dist);
     if (dist_cost >= max_cost_check) return max_cost_check;

    // Cost of length (len - min_match_len)
    const size_t min_match_len = rangecoder::LZRC_MIN_MATCH_LEN;
    if (len < min_match_len) { // Should not happen if find_all_matches is correct, but safety check
         return max_cost_check;
    }
    uint32_t len_val = len - min_match_len;
    int len_cost = estimate_varuint_bits(len_val, M_len);
    if (len_cost >= max_cost_check) return max_cost_check;


    // Return total cost, checking for overflow carefully
    int total_cost = flag_cost;
    if (total_cost > max_cost_check - dist_cost) return max_cost_check;
    total_cost += dist_cost;
    if (total_cost > max_cost_check - len_cost) return max_cost_check;
    total_cost += len_cost;

    return total_cost;
}

// --- Match Finding ---
// Returns *all* valid matches found within the window.
std::vector<MatchCandidate> find_all_matches(
    const std::vector<uint8_t>& data, // Changed from raw to data for clarity
    size_t current_pos,               // Changed from global_pos + pos
    size_t data_size,                 // Changed from N
    size_t window_size,               // Changed from win_size
    size_t min_len,                   // Changed from min_match_len
    size_t max_len)                   // Changed from max_match_len
{
    std::vector<MatchCandidate> candidates;
    if (current_pos == 0) return candidates; // Cannot match at the beginning

    // Ensure max_len doesn't exceed remaining data
    max_len = std::min(max_len, data_size - current_pos);

    if (max_len < min_len) return candidates; // Not enough data left for even min match

    size_t search_start = (current_pos > window_size) ? (current_pos - window_size) : 0;

    // --- Find all matches meeting min_len requirement ---
    // Note: This can return many candidates. Real compressors often use heuristics
    //       (e.g., keep N best, limit based on length/distance) to prune this list.
    //       For now, we return all found matches.

    const uint8_t* data_ptr = data.data(); // Use pointer for potential minor speedup
    const uint8_t* current_ptr = data_ptr + current_pos;

    // Search backwards from current_pos - 1
    for (size_t match_start_idx = current_pos - 1; ; --match_start_idx) {
        uint32_t current_match_len = 0;
        const uint8_t* match_start_ptr = data_ptr + match_start_idx;

        // Compare bytes using pointers
        while (current_match_len < max_len &&
               match_start_ptr[current_match_len] == current_ptr[current_match_len])
        {
            current_match_len++;
        }

        // If a valid match is found, add it to the candidates list
        if (current_match_len >= min_len) {
             uint32_t match_dist = static_cast<uint32_t>(current_pos - match_start_idx);
             // Add this match candidate to the list
             candidates.push_back({match_dist, static_cast<uint16_t>(current_match_len)});
             // Optimization: If we find a match of maximum possible length *starting at this position*,
             // any shorter match starting at the same position is redundant for optimal parse
             // (since the longer one covers it and gives more options later).
             // However, finding *multiple* matches starting at *different* positions is the goal.
             // We don't break here, allowing shorter matches from further back to be found.
        }

        // Break condition needs to be checked *before* potential underflow if search_start is 0
        if (match_start_idx == search_start) break;
    }

    // Optional: Sort candidates? (e.g., by length descending, then dist ascending)
    // Might help subsequent processing, but not strictly necessary for correctness.
    // std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b){ ... });

    return candidates;
}

} // end anonymous namespace

namespace rangecoder {

//----------------------------------------
// RangeEncoder implementation
//----------------------------------------
RangeEncoder::RangeEncoder(std::vector<uint8_t>& out)
    : outBuf(out), low(0), range(0xFFFFFFFFu)
{}

void RangeEncoder::shift_low(std::vector<uint8_t>& buf,
                             uint32_t& low, uint32_t& range)
{
    // emit bytes while MSB of low and low+range match
    while (((low ^ (low + range)) & 0xFF000000u) == 0) {
        buf.push_back(uint8_t(low >> 24));
        low <<= 8;
        range <<= 8;
    }
    // refill if range too small
    if (range < RC_BOTTOM_VALUE) {
        range = (~low + 1) & (RC_BOTTOM_VALUE - 1);
        buf.push_back(uint8_t(low >> 24));
        low <<= 8;
        range <<= 8;
    }
}

void RangeEncoder::encode(uint32_t cumFreq,
                          uint32_t freq,
                          uint32_t totFreq)
{
    assert(freq > 0 && cumFreq + freq <= totFreq);
    // scale range by total frequency
    range /= totFreq;
    low   += range * cumFreq;
    range *= freq;
    shift_low(outBuf, low, range);
}

void RangeEncoder::flush()
{
    // output final 4 bytes
    for (int i = 0; i < 4; ++i) {
        outBuf.push_back(uint8_t(low >> 24));
        low <<= 8;
    }
}

//----------------------------------------
// RangeDecoder implementation
//----------------------------------------

RangeDecoder::RangeDecoder(const uint8_t* buf, size_t size)
    : inBuf(buf), inSize(size), inPos(0), low(0), code(0), range(0xFFFFFFFFu)
{
    for (int i = 0; i < 4; ++i) {
        code = (code << 8) | read_byte();
    }
}

inline uint8_t RangeDecoder::read_byte()
{
    if (inPos >= inSize) throw std::runtime_error("Unexpected EOF");
    return inBuf[inPos++];
}

uint32_t RangeDecoder::getFreq(uint32_t totFreq)
{
    assert(totFreq <= RC_BOTTOM_VALUE);
    range /= totFreq;
    uint32_t tmp = (code - low) / range;
    return tmp;
}

void RangeDecoder::decode(uint32_t cumFreq,
                          uint32_t freq,
                          uint32_t totFreq)
{
    assert(freq > 0 && cumFreq + freq <= totFreq);
    low   += range * cumFreq;
    range *= freq;
    shift_low();
}

inline void RangeDecoder::shift_low()
{
    while (((low ^ (low + range)) & 0xFF000000u) == 0) {
        code  = (code << 8) | read_byte();
        low  <<= 8;
        range <<= 8;
    }
    if (range < RC_BOTTOM_VALUE) {
        range = (~low + 1) & (RC_BOTTOM_VALUE - 1);
        code  = (code << 8) | read_byte();
        low  <<= 8;
        range <<= 8;
    }
}

} // namespace rangecoder

//----------------------------------------
// Top-level Order-1 Static Encode/Decode
//----------------------------------------

namespace { // Anonymous namespace for internal helpers

// Helper to write big-endian uint32_t
void be_write32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(uint8_t(v >> 24));
    buf.push_back(uint8_t(v >> 16));
    buf.push_back(uint8_t(v >> 8));
    buf.push_back(uint8_t(v));
}

// Helper to write big-endian uint16_t
void be_write16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(uint8_t(v >> 8));
    buf.push_back(uint8_t(v));
}

// Helper to read big-endian uint32_t
uint32_t be_read32(const uint8_t*& p, const uint8_t* end) {
    if (p + 4 > end) throw std::runtime_error("Insufficient data reading uint32_t");
    uint32_t v = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
                 (uint32_t(p[2]) << 8)  | uint32_t(p[3]);
    p += 4;
    return v;
}

// Helper to read big-endian uint16_t
uint16_t be_read16(const uint8_t*& p, const uint8_t* end) {
    if (p + 2 > end) throw std::runtime_error("Insufficient data reading uint16_t");
    uint16_t v = (uint16_t(p[0]) << 8) | uint16_t(p[1]);
    p += 2;
    return v;
}

} // anonymous namespace

namespace rangecoder { // Re-open namespace for public API functions

std::vector<uint8_t> encode_order1(const std::vector<uint8_t>& raw) {
    if (raw.empty()) return {};

    uint32_t N = raw.size();

    // ---------- build order-1 table ----------
    uint16_t freq[256][256] = {};        // zero-initialise
    uint8_t ctx = 0;                     // arbitrary start ctx
    for (uint8_t b : raw) {
        if (freq[ctx][b] < 0xFFFF) { // Prevent overflow for static counts
             ++freq[ctx][b];
        }
        ctx = b;
    }

    // Optional Laplace smoothing: start every cell at 1 instead of 0
    // for (int c=0; c<256; ++c) for (int s=0; s<256; ++s) if (freq[c][s]==0) freq[c][s]=1;

    // Compute cumulative counts and totals
    uint32_t cumFreq[256][257]; // Use size 257 for cumFreq[c][256] = totFreq[c]
    uint32_t totFreq[256];

    for (int c = 0; c < 256; ++c) {
        cumFreq[c][0] = 0;
        uint32_t current_total = 0; // Use uint32_t for intermediate sums
        for (int s = 0; s < 256; ++s) {
            cumFreq[c][s+1] = cumFreq[c][s] + freq[c][s];
            current_total += freq[c][s];
        }
        totFreq[c] = current_total;
        // Ensure totFreq > 0 for all contexts used.
        if (totFreq[c] == 0) {
            // If a context has zero total frequency, give it a uniform distribution
             for(int s=0; s<256; ++s) freq[c][s] = 1;
             cumFreq[c][0] = 0;
             for(int s=0; s<256; ++s) cumFreq[c][s+1] = cumFreq[c][s] + freq[c][s];
             totFreq[c] = 256;
        }
    }


    // ---------- Write Header and Encode ----------
    std::vector<uint8_t> out;
    // Reserve space estimate: N + header size + few extra
    // Avoids potential reallocations if data doesn't compress well.
    size_t reserve_size = 4 + 256 * 256 * 2 + raw.size() + (raw.size() / 10); // Header + 110% of raw size
    out.reserve(reserve_size);

    // Write header: N then freq table
    be_write32(out, N);
    for (int c = 0; c < 256; ++c) {
        for (int s = 0; s < 256; ++s) {
            be_write16(out, freq[c][s]);
        }
    }

    RangeEncoder enc(out); // Pass the vector to store output directly
    ctx = 0; // Reset context for encoding loop
    for (uint8_t sym : raw) {
        // Ensure the context has valid frequencies before encoding
        assert(totFreq[ctx] > 0);
        // Use cumFreq[c][s] for start, freq[c][s] for width
        enc.encode(cumFreq[ctx][sym], freq[ctx][sym], totFreq[ctx]);
        ctx = sym;                      // advance context
    }
    enc.flush(); // Finalize the encoding stream

    return out; // 'out' now holds the compressed data
}

std::vector<uint8_t> decode_order1(const std::vector<uint8_t>& comp) {
    if (comp.size() < 4 + 256 * 256 * 2) {
        throw std::runtime_error("Compressed data too small for header");
    }

    const uint8_t* p = comp.data();
    const uint8_t* end = p + comp.size();

    // --- Read Header ---
    uint32_t N = be_read32(p, end);
    if (N == 0) return {}; // Handle empty original data case

    uint16_t freq[256][256];
    for (int c = 0; c < 256; ++c) {
        for (int s = 0; s < 256; ++s) {
            freq[c][s] = be_read16(p, end);
        }
    }

    // --- Compute cumulative frequencies (must match encoder exactly) ---
    uint32_t cumFreq[256][257]; // Use size 257
    uint32_t totFreq[256];

    for (int c = 0; c < 256; ++c) {
        cumFreq[c][0] = 0;
        uint32_t current_total = 0; // Use uint32_t for intermediate sums
        for (int s = 0; s < 256; ++s) {
            cumFreq[c][s+1] = cumFreq[c][s] + freq[c][s];
            current_total += freq[c][s];
        }
        totFreq[c] = current_total;
        // Apply same logic as encoder for zero-frequency contexts
        if (totFreq[c] == 0) {
             for(int s=0; s<256; ++s) freq[c][s] = 1; // Assume encoder did this
             cumFreq[c][0] = 0;
             for(int s=0; s<256; ++s) cumFreq[c][s+1] = cumFreq[c][s] + freq[c][s];
             totFreq[c] = 256;
        }
    }

    // --- Decode ---
    std::vector<uint8_t> out;
    out.reserve(N);

    size_t headerSize = p - comp.data(); // Correct way to get header size read
    if (comp.size() <= headerSize) {
        if (N > 0) throw std::runtime_error("Missing payload data after header");
    }

    RangeDecoder dec(p, end - p); // Pass remaining buffer slice

    uint8_t ctx = 0; // Initial context
    for (uint32_t i = 0; i < N; ++i) {
         assert(totFreq[ctx] > 0); // Should be guaranteed by build logic

        uint32_t f = dec.getFreq(totFreq[ctx]);

        // Locate symbol inside ctx’s histogram using binary search on cumulative frequencies
        // Find smallest s such that f < cumFreq[ctx][s+1]
        uint8_t sym = 0;
        int low = 0, high = 255;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (f < cumFreq[ctx][mid+1]) { // Check if f is in the range for symbol `mid`
                 // Potential match, but check if it's also >= cumFreq[ctx][mid]
                 if (f >= cumFreq[ctx][mid]) {
                      sym = mid;
                      break;
                 } else {
                      high = mid - 1; // f is too low, search in lower half
                 }
            } else {
                 low = mid + 1; // f is too high, search in upper half
            }
        }
        // Fallback linear search if binary search fails (shouldn't happen with correct logic)
        // The prompt suggested linear search, let's stick to that for simplicity first.
        sym = 0;
        while (cumFreq[ctx][sym+1] <= f && sym < 255) {
             sym++;
        }

        dec.decode(cumFreq[ctx][sym], freq[ctx][sym], totFreq[ctx]);

        out.push_back(sym);
        ctx = sym;                         // update context
    }

    return out;
}

std::vector<uint8_t> encode_adaptive(const std::vector<uint8_t>& raw)
{
    ModelO1 M; 
    std::vector<uint8_t> out;
    // Reserve rough estimate - adaptive should be smaller than raw
    out.reserve(raw.size() * 9 / 10 + 16); // Heuristic: 90% + small buffer

    // --- Write minimal header: N (original size) ---
    uint32_t N = raw.size();
    be_write32(out, N);

    // --- Encode Payload ---
    RangeEncoder enc(out); // Encoder writes directly to comp
    uint8_t ctx = 0;
    for (uint8_t sym : raw) {
        enc.encode(M.cum[ctx][sym], M.freq[ctx][sym], M.cum[ctx][256]);

        // --- model update ---
        ++M.freq[ctx][sym];
        // Fast path: bump cumulative cells *after* 'sym'
        for (int s = sym + 1; s <= 256; ++s) ++M.cum[ctx][s];

        // --- rescale check ---
        if (M.cum[ctx][256] >= MAX_C) {
            // Rescale: halve counts but keep them >= 1
            for (int s = 0; s < 256; ++s)
                M.freq[ctx][s] = (M.freq[ctx][s] + 1) >> 1;
            M.rebuild(ctx); // Rebuild cumulative counts for this context
        }
        ctx = sym; // Update context for next symbol
    }
    enc.flush();
    return out;
}


std::vector<uint8_t> decode_adaptive(const std::vector<uint8_t>& comp)
{
    if (comp.size() < 4) { // Need at least 4 bytes for N
         throw std::runtime_error("Compressed data too small for size header");
    }

    ModelO1 M; // Initialized identically to encoder
    const uint8_t* p = comp.data();
    const uint8_t* end = p + comp.size();

    // --- Read minimal header: N ---
    uint32_t N = be_read32(p, end);
    if (N == 0) return {}; // Handle empty original data case

    std::vector<uint8_t> out;
    out.reserve(N); // Reserve exact space

    // --- Decode Payload ---
    // Check if there's any data left for the RangeDecoder payload
    size_t headerSize = p - comp.data();
    if (comp.size() <= headerSize) {
         if (N > 0) throw std::runtime_error("Missing payload data after size header");
    }

    RangeDecoder dec(p, end - p); // Pass remaining buffer slice
    uint8_t ctx = 0;

    for (uint32_t i = 0; i < N; ++i) { // Loop exactly N times
        uint32_t total_freq_for_ctx = M.cum[ctx][256];
        if (total_freq_for_ctx == 0) {
             // Should not happen with Laplace smoothing init and rescaling >= 1
             // If it *can* become zero after updates (unlikely with MAX_C limit), handle it.
             // For now, assume it's an error state.
             throw std::runtime_error("Decoder model reached zero total frequency");
        }

        uint32_t f = dec.getFreq(total_freq_for_ctx);

        // Linear search for symbol
        uint8_t sym = 0;
        // Find smallest s such that f < M.cum[ctx][s+1]
        while (sym < 255 && f >= M.cum[ctx][sym+1]) {
             ++sym;
        }

        dec.decode(M.cum[ctx][sym], M.freq[ctx][sym], total_freq_for_ctx);
        out.push_back(sym);

        // --- mirror encoder’s update ---
        ++M.freq[ctx][sym];
        for (int s = sym + 1; s <= 256; ++s) ++M.cum[ctx][s];

        // --- mirror encoder's rescale check ---
        if (M.cum[ctx][256] >= MAX_C) {
            for (int s = 0; s < 256; ++s)
                M.freq[ctx][s] = (M.freq[ctx][s] + 1) >> 1;
            M.rebuild(ctx);
        }
        ctx = sym; // Update context
    }

    // Optional: Check if decoder consumed the whole stream or if N was wrong?
    // Not strictly necessary if we trust N.

    return out;
}

namespace internal { // Implementation of internal helpers

// Encodes one byte using the range coder and a ModelO0<256>
void encode_byte(RangeEncoder& enc, ModelO0<256>& model, uint8_t byte_val) {
    enc.encode(model.cum[byte_val], model.freq[byte_val], model.total);
    model.update(byte_val); // Update the model
}

// Decodes one byte using the range coder and a ModelO0<256>
uint8_t decode_byte(RangeDecoder& dec, ModelO0<256>& model) {
    uint32_t freq = dec.getFreq(model.total);
    uint8_t symbol = 0;
    while (symbol < 255 && freq >= model.cum[symbol + 1]) {
        symbol++;
    }
    dec.decode(model.cum[symbol], model.freq[symbol], model.total);
    model.update(symbol);
    return symbol; // Return the decoded byte
}


// Writes a whole uint32_t value as LEB128 using the byte-level encoder
void write_varuint(RangeEncoder& enc, ModelO0<256>& model, uint32_t value) {
    do {
        uint8_t byte = value & 0x7F; // Get lower 7 bits
        value >>= 7;
        if (value != 0) { // More bytes to follow?
            byte |= 0x80; // Set continuation bit
        }
        encode_byte(enc, model, byte); // Encode the byte using the provided model
    } while (value != 0);
}

// Reads a whole uint32_t value as LEB128 using the byte-level decoder
uint32_t read_varuint(RangeDecoder& dec, ModelO0<256>& model) {
    uint32_t result = 0;
    int shift = 0;
    uint8_t byte;
    do {
        byte = decode_byte(dec, model); // Corrected call

        result |= static_cast<uint32_t>(byte & 0x7F) << shift;
        shift += 7;
        if (shift > 35) { // Protect against malformed input (more than 5 bytes)
            throw std::runtime_error("VarUInt decoding exceeded 5 bytes");
        }
    } while (byte & 0x80); // Continue if continuation bit is set
    return result;
}


} // namespace internal

// --- Implementations for encode_lzrc / decode_lzrc will go here --- 

std::vector<uint8_t> encode_lzrc(const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> comp;
    uint32_t N = raw.size();
    if (N == 0) return comp; // Return empty vector for empty input

    // Reserve space and write header later (after encoder finishes)
    comp.reserve(N / 2 + 4); // Initial guess + 4 bytes for header

    // --- Initialize Models & Encoder ---
    RangeEncoder enc(comp); // Encoder writes directly to comp
    ModelO0<2> M_flag;
    ModelO1 M_lit;
    ModelO0<256> M_dist;
    ModelO0<256> M_len;

    // Constants defined in rangecoder.h
    const size_t min_match_len = rangecoder::LZRC_MIN_MATCH_LEN; // QUALIFIED
    const size_t win_size = rangecoder::LZRC_WIN_SIZE;         // QUALIFIED
    const size_t max_match_len = rangecoder::LZRC_MAX_MATCH_LEN;   // QUALIFIED

    // --- Optimal Parsing Setup ---
    const size_t block_len = 4096;
    std::vector<Opt> opts(block_len + max_match_len + 1);
    std::vector<MatchCandidate> chosen_tokens; // To store results of trace-back

    size_t global_pos = 0; // Position in the overall 'raw' buffer

    while (global_pos < N) {
        size_t current_block_len = std::min(block_len, N - global_pos);
        const uint8_t* block_buf_start = raw.data() + global_pos; // Pointer to start of current block in raw

        // --- Reset DP state for the block ---
        for (size_t i = 0; i <= current_block_len + max_match_len; ++i) {
             opts[i] = Opt{}; // Use default constructor (sets price to max)
        }
        opts[0].price = 0; // Start node has zero cost

        // --- Forward Pass (Fill DP Table) ---
        for (size_t pos = 0; pos < current_block_len; ++pos) {
            if (opts[pos].price == std::numeric_limits<int>::max()) continue; // Skip unreachable

            int base_price = opts[pos].price;

            // --- Evaluate Literal ---
            uint8_t current_ctx = (global_pos + pos > 0) ? raw[global_pos + pos - 1] : 0; // Approx context
            uint8_t current_literal = block_buf_start[pos];

            int p_lit = base_price + get_literal_price(current_literal, current_ctx, M_flag, M_lit);
            if (p_lit < opts[pos + 1].price) {
                opts[pos + 1] = Opt(p_lit, 1, 0);
            }

            // --- Evaluate Matches ---
            std::vector<MatchCandidate> candidates = find_all_matches(
                raw, global_pos + pos, N, win_size, min_match_len, max_match_len
             );

            for (const auto& match : candidates) {
                if (pos + match.len < opts.size()) {
                     int p_match = base_price + get_match_price(match.dist, match.len, M_flag, M_dist, M_len);
                     if (p_match < opts[pos + match.len].price) {
                         opts[pos + match.len] = Opt(p_match, static_cast<uint16_t>(match.len), match.dist);
                     }
                }
            }
        } // End forward pass

        // --- Trace-back Pass ---
        chosen_tokens.clear();
        chosen_tokens.reserve(current_block_len);
        size_t current_trace_pos = current_block_len; // Start trace from end of logical block

        while (current_trace_pos > 0) {
            const Opt& current_opt = opts[current_trace_pos];
            if (current_opt.len == 0 || current_opt.price == std::numeric_limits<int>::max()) {
                 // Add context to error message
                 throw std::runtime_error("Optimal parse trace-back failed: Invalid state at block offset " + 
                                          std::to_string(current_trace_pos) + " (global pos " + 
                                          std::to_string(global_pos + current_trace_pos) + ")");
            }

            if (current_opt.len == 1) { // Literal
                chosen_tokens.push_back({0, 1});
            } else { // Match
                chosen_tokens.push_back({current_opt.dist, current_opt.len});
            }
            current_trace_pos -= current_opt.len;
        }
        std::reverse(chosen_tokens.begin(), chosen_tokens.end());

        // --- Encode Chosen Tokens ---
        size_t block_pos_encoded = 0;
        for (const auto& token : chosen_tokens) {
             uint8_t context_byte = (global_pos + block_pos_encoded > 0) ? raw[global_pos + block_pos_encoded - 1] : 0;

             if (token.dist == 0) { // Encode Literal
                  uint8_t literal = raw[global_pos + block_pos_encoded];
                  enc.encode(M_flag.cum[0], M_flag.freq[0], M_flag.total); M_flag.update(0);
                  enc.encode(M_lit.cum[context_byte][literal], M_lit.freq[context_byte][literal], M_lit.cum[context_byte][256]); M_lit.update(context_byte, literal);
                  block_pos_encoded += 1;
             } else { // Encode Match
                  uint32_t match_dist = token.dist;
                  uint32_t match_len = token.len;
                  enc.encode(M_flag.cum[1], M_flag.freq[1], M_flag.total); M_flag.update(1);
                  internal::write_varuint(enc, M_dist, match_dist - 1);
                  internal::write_varuint(enc, M_len, match_len - rangecoder::LZRC_MIN_MATCH_LEN); // QUALIFIED
                  block_pos_encoded += match_len;
             }
        }

        if (block_pos_encoded != current_block_len) {
             // Add more context to the error message
             throw std::runtime_error("Optimal parse encoding mismatch: Encoded " + std::to_string(block_pos_encoded) + 
                                      " bytes, expected " + std::to_string(current_block_len) + 
                                      " in block starting at global pos " + std::to_string(global_pos));
        }
        global_pos += current_block_len;
        // TODO: Handle lookahead carry-over
    } // End while (global_pos < N)

    enc.flush();

    // --- Prepend Header ---
    std::vector<uint8_t> final_comp;
    final_comp.reserve(comp.size() + 4);
    be_write32(final_comp, N);
    final_comp.insert(final_comp.end(), comp.begin(), comp.end());

    return final_comp;
}

std::vector<uint8_t> decode_lzrc(const std::vector<uint8_t>& comp) {
    if (comp.size() < 4) {
        throw std::runtime_error("Compressed data too small for header");
    }
    const uint8_t* p = comp.data();
    const uint8_t* end = p + comp.size();

    // --- Read Header: N ---
    uint32_t N = be_read32(p, end); 
    if (N == 0) return {}; 

    std::vector<uint8_t> raw;
    raw.reserve(N);

    // --- Initialize Decoder & Models ---
    RangeDecoder dec(p, end - p); 
    ModelO0<2> M_flag;
    ModelO1 M_lit;
    ModelO0<256> M_dist;
    ModelO0<256> M_len;

    uint8_t prev_byte = 0; 

    while (raw.size() < N) {
        // --- Decode Flag ---
        uint32_t flag_total = M_flag.total; 
        uint32_t flag_freq = dec.getFreq(flag_total);
        uint8_t is_match;
        if (flag_freq < M_flag.cum[1]) { 
            is_match = 0;
            dec.decode(M_flag.cum[0], M_flag.freq[0], flag_total);
        } else {
            is_match = 1;
            dec.decode(M_flag.cum[1], M_flag.freq[1], flag_total);
        }
        M_flag.update(is_match);

        // --- Decode based on flag ---
        if (is_match) {
            // Decode Match
            uint32_t dist_val = internal::read_varuint(dec, M_dist);
            uint32_t len_val = internal::read_varuint(dec, M_len);

            uint32_t match_dist = dist_val + 1;
            uint32_t match_len = len_val + rangecoder::LZRC_MIN_MATCH_LEN; // QUALIFIED

            if (match_dist > raw.size()) {
                 throw std::runtime_error("Invalid match distance");
            }
            size_t remaining_capacity = N - raw.size();
            if (match_len > remaining_capacity) { 
                 throw std::runtime_error("Match length exceeds decoded size limit N");
            }

            // Copy match bytes
            size_t match_start_pos = raw.size() - match_dist;
            for (uint32_t i = 0; i < match_len; ++i) {
                raw.push_back(raw[match_start_pos + i]);
            }

            // Update context based on last byte of the match
            if (!raw.empty()) { 
                 prev_byte = raw.back();
            }

        } else {
            // Decode Literal
            uint32_t lit_total = M_lit.cum[prev_byte][256]; 
            if (lit_total == 0) {
                 // This case implies context never seen before AND no Laplace smoothing
                 // With the current ModelO1 init (Laplace=1), total should always be >= 256 initially
                 // If it *can* become zero after updates (unlikely with MAX_C limit), handle it.
                 // For now, assume it's an error state.
                 throw std::runtime_error("Zero total frequency in literal model context");
            }
            uint32_t lit_freq = dec.getFreq(lit_total);

            uint8_t literal = 0;
            // Find symbol range containing lit_freq
            while (literal < 255 && lit_freq >= M_lit.cum[prev_byte][literal + 1]) {
                literal++;
            }

            dec.decode(M_lit.cum[prev_byte][literal], M_lit.freq[prev_byte][literal], lit_total);
            M_lit.update(prev_byte, literal);

            raw.push_back(literal);
            prev_byte = literal; 
        }
    }

     if (raw.size() != N) {
         // This check should ideally be redundant if the above loop condition and match length check work correctly
         throw std::runtime_error("Decoded size does not match header N after loop exit");
     }

    return raw;
}


} // namespace rangecoder
