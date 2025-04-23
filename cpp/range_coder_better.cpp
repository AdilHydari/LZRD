// scalar_range_xz.cpp
// Adapted from Subbotin's implementation to mirror XZ's range coder style
// - Centralized constants in range_common.h
// - 16-bit probability models
// - Inline, branchless normalization
// - Raw header write + separate range coder for payload

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <fstream>

// ------------------ range_common.h ------------------
static constexpr uint32_t RC_TOP_BITS    = 24;
static constexpr uint32_t RC_BOTTOM_BITS = 16;
static constexpr uint32_t RC_TOP_VALUE   = 1u << RC_TOP_BITS;
static constexpr uint32_t RC_BOTTOM_VALUE= 1u << RC_BOTTOM_BITS;

// ------------------ RangeEncoder ------------------

class RangeEncoder {
public:
    explicit RangeEncoder(std::vector<uint8_t>& out)
        : outBuf(out), low(0), range(0xFFFFFFFFu) {}

    // Emit any full bytes from (low, range)
    static inline void shift_low(std::vector<uint8_t>& outBuf, uint32_t& low, uint32_t& range) {
        // branchless: while top bits of low and low+range are equal
        while (((low ^ (low + range)) & 0xFF000000u) == 0) {
            outBuf.push_back(uint8_t(low >> 24));
            low <<= 8;
            range <<= 8;
        }
        // if range falls below bottom, refill
        if (range < RC_BOTTOM_VALUE) {
            range = (~low + 1) & (RC_BOTTOM_VALUE - 1);
            outBuf.push_back(uint8_t(low >> 24));
            low <<= 8;
            range <<= 8;
        }
    }

    // cumFreq and freq must be 16-bit, totFreq <= 1<<16
    inline void encode(uint32_t cumFreq, uint32_t freq, uint32_t totFreq) {
        assert(freq && cumFreq + freq <= totFreq);
        // division by totFreq
        range /= totFreq;
        low += range * cumFreq;
        range *= freq;
        shift_low(outBuf, low, range);
    }

    void flush() {
        // push final 4 bytes
        for (int i = 0; i < 4; ++i) {
            outBuf.push_back(uint8_t(low >> 24));
            low <<= 8;
        }
    }

private:
    std::vector<uint8_t>& outBuf;
    uint32_t low;
    uint32_t range;
};

// ------------------ RangeDecoder ------------------

class RangeDecoder {
public:
    RangeDecoder(const uint8_t* buf, size_t size)
        : inBuf(buf), inSize(size), inPos(0), low(0), code(0), range(0xFFFFFFFFu) {
        // load first 4 bytes
        for (int i = 0; i < 4; ++i)
            code = (code << 8) | read_byte();
    }

    // Returns freq index tmp = (code-low)/(range/totFreq)
    inline uint32_t getFreq(uint32_t totFreq) {
        assert(totFreq <= RC_BOTTOM_VALUE);
        range /= totFreq;
        uint32_t tmp = (code - low) / range;
        return tmp;
    }

    inline void decode(uint32_t cumFreq, uint32_t freq, uint32_t totFreq) {
        assert(freq && cumFreq + freq <= totFreq);
        low += range * cumFreq;
        range *= freq;
        shift_low();
    }

private:
    const uint8_t* inBuf;
    size_t inSize;
    size_t inPos;
    uint32_t low, code, range;

    inline uint8_t read_byte() {
        if (inPos >= inSize) throw std::runtime_error("Unexpected EOF");
        return inBuf[inPos++];
    }

    inline void shift_low() {
        while (((low ^ (low + range)) & 0xFF000000u) == 0) {
            code = (code << 8) | read_byte();
            low <<= 8;
            range <<= 8;
        }
        if (range < RC_BOTTOM_VALUE) {
            range = (~low + 1) & (RC_BOTTOM_VALUE - 1);
            code = (code << 8) | read_byte();
            low <<= 8;
            range <<= 8;
        }
    }
};

// ------------------ Usage Note ------------------
// Write your file header (length, freq[256]) in raw bytes, then:
//   RangeEncoder enc(out);
//   for each symbol: enc.encode(cumFreq[s], freq[s], totFreq);
//   enc.flush();
// Decoder does reverse: raw-read header, then RangeDecoder dec(buf, size);

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <input> <compressed> <decompressed>\n", argv[0]);
        return 1;
    }
    const char* inputPath = argv[1];
    const char* compPath  = argv[2];
    const char* decompPath= argv[3];

    // Read input file
    std::vector<uint8_t> input;
    {
        std::ifstream ifs(inputPath, std::ios::binary);
        if (!ifs) { std::perror("Input open"); return 1; }
        input.assign(std::istreambuf_iterator<char>(ifs), {});
    }

    // Build freq and cumFreq
    std::vector<uint16_t> freq(256, 0);
    for (auto b : input) freq[b]++;
    uint32_t totFreq = 0;
    for (auto f : freq) totFreq += f;
    std::vector<uint32_t> cumFreq(256, 0);
    for (int i = 1; i < 256; ++i) cumFreq[i] = cumFreq[i-1] + freq[i-1];

    // Open compressed output and write raw header
    std::vector<uint8_t> comp;
    {
        std::ofstream ofs(compPath, std::ios::binary);
        if (!ofs) { std::perror("Comp open"); return 1; }
        // Write length
        uint32_t N = input.size();
        for (int shift = 24; shift >= 0; shift -= 8)
            ofs.put(char((N >> shift) & 0xFF));
        // Write freq table
        for (auto f : freq) {
            ofs.put(char((f >> 8) & 0xFF));
            ofs.put(char(f & 0xFF));
        }
        ofs.close();
    }
    // Encode payload
    {
        // Pre-reserve file
        comp.reserve(8 + input.size());
        RangeEncoder enc(comp);
        for (auto b : input) {
            enc.encode(cumFreq[b], freq[b], totFreq);
        }
        enc.flush();
        // Append to compressed file
        std::ofstream ofs(compPath, std::ios::binary | std::ios::app);
        ofs.write((char*)comp.data(), comp.size());
    }

    // Read back compressed
    std::vector<uint8_t> fullComp;
    {
        std::ifstream ifs(compPath, std::ios::binary);
        fullComp.assign(std::istreambuf_iterator<char>(ifs), {});
    }

    // Decode
    std::vector<uint8_t> output;
    {
        const uint8_t* ptr = fullComp.data();
        size_t pos = 0;
        uint32_t N = 0;
        for (int i = 0; i < 4; ++i)
            N = (N << 8) | ptr[pos++];
        std::vector<uint16_t> freqD(256);
        for (int i = 0; i < 256; ++i) {
            uint16_t hi = ptr[pos++];
            uint16_t lo = ptr[pos++];
            freqD[i] = (hi << 8) | lo;
        }
        std::vector<uint32_t> cumFreqD(256);
        for (int i = 1; i < 256; ++i)
            cumFreqD[i] = cumFreqD[i-1] + freqD[i-1];
        uint32_t totFreqD = cumFreqD[255] + freqD[255];

        RangeDecoder dec(ptr + pos, fullComp.size() - pos);
        output.reserve(N);
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t sym = dec.getFreq(totFreqD);
            // find symbol via cumFreqD
            uint8_t b = 0;
            while (!(sym >= cumFreqD[b] && sym < cumFreqD[b] + freqD[b])) ++b;
            dec.decode(cumFreqD[b], freqD[b], totFreqD);
            output.push_back(b);
        }
    }

    // Write decompressed
    {
        std::ofstream ofs(decompPath, std::ios::binary);
        ofs.write((char*)output.data(), output.size());
    }

    // Verify
    if (output == input) {
        std::puts("OK: decompressed matches input");
        return 0;
    } else {
        std::puts("ERROR: mismatch");
        return 1;
    }
}
