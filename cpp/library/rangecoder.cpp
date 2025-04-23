// =============== rangecoder.cpp ===============
#include "rangecoder.h"
#include <cassert>
#include <stdexcept>
#include <vector>
#include <cstdint>

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
    ModelO1 M; // Initialized with Laplace smoothing
    std::vector<uint8_t> out;
    // Reserve rough estimate - adaptive should be smaller than raw
    out.reserve(raw.size() * 9 / 10 + 16); // Heuristic: 90% + small buffer

    // --- Write minimal header: N (original size) ---
    uint32_t N = raw.size();
    be_write32(out, N);

    // --- Encode Payload ---
    RangeEncoder enc(out); // Pass vector by reference
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
    if (raw.empty()) {
        std::vector<uint8_t> header(4, 0); // Header only for empty input (N=0)
        return header;
    }

    std::vector<uint8_t> comp;
    RangeEncoder enc(comp);

    // Write original size N
    uint32_t N = raw.size();
    be_write32(comp, N); // Prepend size

    // --- Initialize Models ---
    ModelO0<2> M_flag;       // Model for is_match flag (0=literal, 1=match)
    ModelO1 M_lit;          // Model for literals (context is previous byte)
    ModelO0<256> M_dist;    // Model for distance VarUInt bytes
    ModelO0<256> M_len;     // Model for length VarUInt bytes

    uint8_t prev_byte = 0;  // Context for the very first byte
    size_t pos = 0;         // Current position in raw data

    while (pos < N) {
        // --- Find Longest Match ---
        uint32_t best_match_len = 0;
        uint32_t best_match_dist = 0;

        // Use constants with correct names and no prefix
        size_t search_start = (pos > WIN_SIZE) ? (pos - WIN_SIZE) : 0; 
        size_t max_possible_len = N - pos;

        // Search backwards from pos-1 down to search_start
        // Use explicit check for pos > 0 before loop entry
        if (pos > 0) { 
            for (size_t match_start_idx = pos -1; ; --match_start_idx) {
            
                uint32_t current_match_len = 0;
                // Compare bytes starting from match_start_idx and pos
                while (current_match_len < max_possible_len &&
                    raw[match_start_idx + current_match_len] == raw[pos + current_match_len])
                {
                    current_match_len++;
                }

                // Check if this match is better (longer)
                if (current_match_len > best_match_len) {
                    best_match_len = current_match_len;
                    best_match_dist = static_cast<uint32_t>(pos - match_start_idx); 
                    // Optimization: If we found max possible length, no need to search further back
                    if (best_match_len == max_possible_len) break;
                }

                // Break condition needs to be checked *before* potential underflow
                if (match_start_idx == search_start) break; 
            }
        }


        // --- Encode ---
        if (best_match_len >= MIN_MATCH_LEN) { 
            // Encode Match Flag = 1 directly using M_flag
            enc.encode(M_flag.cum[1], M_flag.freq[1], M_flag.total); 
            M_flag.update(1);

            // Encode distance and length using VarUInt helpers
            internal::write_varuint(enc, M_dist, best_match_dist - 1);           
            internal::write_varuint(enc, M_len, best_match_len - MIN_MATCH_LEN); 

            // Update context for the *next* token based on the last byte of the match
            prev_byte = raw[pos + best_match_len - 1];
            pos += best_match_len; // Advance position

        } else {
            // Encode Literal
            enc.encode(M_flag.cum[0], M_flag.freq[0], M_flag.total); 
            M_flag.update(0);

            uint8_t literal = raw[pos];
            // Use prev_byte as context for the literal model
            enc.encode(M_lit.cum[prev_byte][literal], M_lit.freq[prev_byte][literal], M_lit.cum[prev_byte][256]);
            M_lit.update(prev_byte, literal); 

            prev_byte = literal; 
            pos++; 
        }
    }

    enc.flush(); 

    // Prepend the header size back (it was modified by RangeEncoder constructor)
    be_write32(comp, N);

    return comp;
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
            uint32_t match_len = len_val + MIN_MATCH_LEN; 

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
