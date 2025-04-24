// =============== rangecoder.h ===============
#ifndef RANGECODER_H
#define RANGECODER_H

#include <vector>
#include <cstdint>

namespace rangecoder {

//---------------------------------------------------------------------------
// Constants for range coder
//---------------------------------------------------------------------------
static constexpr uint32_t RC_TOP_BITS      = 24;
static constexpr uint32_t RC_BOTTOM_BITS   = 16;
static constexpr uint32_t RC_TOP_VALUE     = 1u << RC_TOP_BITS;
static constexpr uint32_t RC_BOTTOM_VALUE  = 1u << RC_BOTTOM_BITS;

//---------------------------------------------------------------------------
// Adaptive Order-1 Model Structures & Constants
//---------------------------------------------------------------------------
static constexpr uint16_t ONE   = 1;          // Laplace smoothing
static constexpr uint16_t MAX_C = 1u << 15;   // rescale threshold (<= RC_BOTTOM_VALUE / 2)

struct ModelO1 {
    uint16_t freq[256][256];      // counts >= 1
    uint16_t cum [256][257];      // cum[x][256] == total for context x

    ModelO1() {                   // ctor builds equally-likely table
        for (int c = 0; c < 256; ++c) {
            for (int s = 0; s < 256; ++s) freq[c][s] = ONE;
            rebuild(c);
        }
    }

    // Rebuilds cumulative frequencies for a given context
    inline void rebuild(int ctx) {
        uint16_t run = 0;
        for (int s = 0; s < 256; ++s)
            cum[ctx][s] = run, run += freq[ctx][s];
        cum[ctx][256] = run; // Total frequency stored at index 256
    }

    // Update method for O1 literals in LZRC
    inline void update(uint8_t ctx, uint8_t sym) {
        freq[ctx][sym]++;
        for (int s = sym + 1; s <= 256; ++s) cum[ctx][s]++;

        if (cum[ctx][256] >= rangecoder::MAX_C) {
            for (int s = 0; s < 256; ++s)
                freq[ctx][s] = (freq[ctx][s] + 1) >> 1;
            rebuild(ctx);
        }
    }
};

//---------------------------------------------------------------------------
// LZRC Constants
//---------------------------------------------------------------------------
const size_t LZRC_WIN_SIZE = 1 << 15; // 32 KiB window
const size_t LZRC_MIN_MATCH_LEN = 3;  // Minimum match length
const size_t LZRC_MAX_MATCH_LEN = 258; // Max length of a match

//---------------------------------------------------------------------------
// Simple Adaptive Order-0 Model
//---------------------------------------------------------------------------
template<size_t N>
struct ModelO0 {
    static_assert(N > 0 && N <= 257, "ModelO0 size must be between 1 and 257");
    uint16_t freq[N] = {}; // Frequencies, init later
    uint16_t cum[N+1] = {}; // Cumulative frequencies
    uint16_t total = 0;     // Total count for rescaling

    ModelO0() { // Init with Laplace smoothing
        for(size_t i = 0; i < N; ++i) freq[i] = ONE;
        rebuild();
    }

    void rebuild() {
        total = 0;
        for(size_t i = 0; i < N; ++i) {
            cum[i] = total;
            total += freq[i];
        }
        cum[N] = total;
    }

    void update(uint16_t sym) {
        if (sym >= N) return; // Safety check

        freq[sym]++;
        // Fast cumulative update
        for (size_t s = sym + 1; s <= N; ++s) cum[s]++;
        total++;

        if (total >= MAX_C) { // Use same rescale threshold for simplicity
            for(size_t i = 0; i < N; ++i) {
                freq[i] = (freq[i] + 1) >> 1; // Halve, keeping >= 1
            }
            rebuild();
        }
    }
};

//---------------------------------------------------------------------------
// RangeEncoder: scalar range encoder for 8-bit alphabet
//---------------------------------------------------------------------------
class RangeEncoder {
public:
    explicit RangeEncoder(std::vector<uint8_t>& out);
    void encode(uint32_t cumFreq, uint32_t freq, uint32_t totFreq);
    void flush();

private:
    static inline void shift_low(std::vector<uint8_t>& buf,
                                 uint32_t& low, uint32_t& range);

    std::vector<uint8_t>& outBuf;
    uint32_t low;
    uint32_t range;
};

//---------------------------------------------------------------------------
// RangeDecoder: scalar range decoder for 8-bit alphabet
//---------------------------------------------------------------------------
class RangeDecoder {
public:
    RangeDecoder(const uint8_t* buf, size_t size);
    uint32_t getFreq(uint32_t totFreq);
    void decode(uint32_t cumFreq, uint32_t freq, uint32_t totFreq);

private:
    inline uint8_t read_byte();
    inline void shift_low();

    const uint8_t* inBuf;
    size_t inSize;
    size_t inPos;
    uint32_t low;
    uint32_t code;
    uint32_t range;
};

// Top-level API functions
//---------------------------------------------------------------------------

// Static order-1 (previous version)
std::vector<uint8_t> encode_order1(const std::vector<uint8_t>& raw);
std::vector<uint8_t> decode_order1(const std::vector<uint8_t>& comp);

// Adaptive order-1 (NEW)
std::vector<uint8_t> encode_adaptive(const std::vector<uint8_t>& raw);
std::vector<uint8_t> decode_adaptive(const std::vector<uint8_t>& comp);

// LZ77 + Adaptive Range Coder (LZRC - NEW)
std::vector<uint8_t> encode_lzrc(const std::vector<uint8_t>& raw); 
std::vector<uint8_t> decode_lzrc(const std::vector<uint8_t>& comp);

// --- Helper function declarations (to be implemented in .cpp) ---
namespace internal {
    // LEB128 VarUInt encoding/decoding using a byte model
    // Note: These encode/decode the raw byte produced by LEB128 logic.
    void encode_byte(RangeEncoder& enc, ModelO0<256>& model, uint8_t byte_val);
    uint8_t decode_byte(RangeDecoder& dec, ModelO0<256>& model);

    // Simpler interface for whole value (implementation uses byte coder)
    void write_varuint(RangeEncoder& enc, ModelO0<256>& model, uint32_t value);
    uint32_t read_varuint(RangeDecoder& dec, ModelO0<256>& model);

} // namespace internal

} // namespace rangecoder

#endif // RANGECODER_H