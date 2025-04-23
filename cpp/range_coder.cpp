#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
// #include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// --------------------------------------------------
// Subbotin's carryless Range Encoder/Decoder
// (Refactored as separate classes)
// --------------------------------------------------
constexpr uint32_t TOP = 1 << 24;
constexpr uint32_t BOT = 1 << 16;

// ==================================================
// RangeEncoder
// ==================================================
class RangeEncoder {
private:
  std::vector<uint8_t> &outVec; // we store the compressed data here
  uint32_t low;
  uint32_t range;

  // Write one byte to outVec
  void outByte(uint8_t c) { outVec.push_back(c); }

public:
  RangeEncoder(std::vector<uint8_t> &out)
      : outVec(out), low(0), range(0xFFFFFFFF) {}

  // Encode a symbol:
  //  cumFreq = sum of frequencies for symbols < thisSymbol
  //  freq    = frequency for thisSymbol
  //  totFreq = sum of all symbol frequencies
  void encode(uint32_t cumFreq, uint32_t freq, uint32_t totFreq) {
    // Safety checks
    assert(cumFreq + freq <= totFreq);
    assert(freq > 0);
    assert(totFreq <= BOT);

    range /= totFreq;
    low += range * cumFreq;
    range *= freq;

    // Renormalize
    while ((low ^ (low + range)) < TOP ||
           (range < BOT && ((range = -low & (BOT - 1)), true))) {
      outByte(static_cast<uint8_t>(low >> 24));
      range <<= 8;
      low <<= 8;
    }
  }

  // Flush final bytes after all symbols are encoded
  void endEncode() {
    for (int i = 0; i < 4; i++) {
      outVec.push_back(static_cast<uint8_t>(low >> 24));
      low <<= 8;
    }
  }
};

// ==================================================
// RangeDecoder
// ==================================================
class RangeDecoder {
private:
  const uint8_t *inBuf;
  uint32_t inSize;
  uint32_t inPos; // current read index in inBuf

  uint32_t low;
  uint32_t code;
  uint32_t range;

  // Read one byte
  uint8_t inByte() {
    if (inPos >= inSize) {
      throw std::runtime_error("RangeDecoder out of data (EOF)");
    }
    return inBuf[inPos++];
  }

public:
  RangeDecoder(const uint8_t *buf, uint32_t size)
      : inBuf(buf), inSize(size), inPos(0), low(0), code(0), range(0xFFFFFFFF) {
    // Load initial 4 bytes into 'code'
    for (int i = 0; i < 4; i++) {
      code = (code << 8) | inByte();
    }
  }

  // Returns the subrange index where the code is pointing
  uint32_t getFreq(uint32_t totFreq) {
    range /= totFreq;
    uint32_t tmp = (code - low) / range;
    if (tmp >= totFreq) {
      throw std::runtime_error("Input data corrupt (getFreq out of range)");
    }
    return tmp;
  }

  // Once you know the symbol => update low, range
  void decode(uint32_t cumFreq, uint32_t freq, uint32_t totFreq) {
    // Checks
    assert(cumFreq + freq <= totFreq);
    assert(freq > 0);
    assert(totFreq <= BOT);

    low += range * cumFreq;
    range *= freq;

    // Renormalize
    while ((low ^ (low + range)) < TOP ||
           (range < BOT && ((range = -low & (BOT - 1)), true))) {
      code = (code << 8) | inByte();
      range <<= 8;
      low <<= 8;
    }
  }
};

// ==================================================
// Helper: read entire file into a std::vector<uint8_t>
// ==================================================
static std::vector<uint8_t> readFile(const std::string &filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  // Read file contents into a buffer
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
  return data;
}

// ==================================================
// Helper: write a std::vector<uint8_t> to a file
// ==================================================
static void writeFile(const std::string &filename,
                      const std::vector<uint8_t> &data) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  ofs.write(reinterpret_cast<const char *>(data.data()), data.size());
}

// ==================================================
// Build freq/cumFreq for [0..255], totFreq must be <= 65536
// ==================================================
static void buildFreqTables(const std::vector<uint8_t> &data,
                            uint32_t freq[256], uint32_t cumFreq[256],
                            uint32_t &totFreq) {
  std::memset(freq, 0, 256 * sizeof(uint32_t));
  for (auto b : data) {
    freq[b]++;
  }
  totFreq = 0;
  for (int i = 0; i < 256; i++) {
    totFreq += freq[i];
  }
  // Build cumulative
  cumFreq[0] = 0;
  for (int i = 1; i < 256; i++) {
    cumFreq[i] = cumFreq[i - 1] + freq[i - 1];
  }
}

// ==================================================
// Main: usage:
//   ./rangecoder input.bin compressed.bin decompressed.bin
//
// Steps:
//  1) Read input file into memory
//  2) Build freq table
//  3) Encode (including writing freq[] as "symbols")
//  4) Write compressed to compressed.bin
//  5) Read compressed.bin
//  6) Decode
//  7) Write decompressed.bin
//  8) Compare to check correctness
// ==================================================
int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " input.bin compressed.bin decompressed.bin\n";
    return 1;
  }
  std::string inFile = argv[1];
  std::string compressedFile = argv[2];
  std::string outFile = argv[3];

  // 1) Read input file
  std::vector<uint8_t> data;
  try {
    data = readFile(inFile);
  } catch (const std::exception &ex) {
    std::cerr << "Error reading input file: " << ex.what() << "\n";
    return 1;
  }
  uint32_t N = (uint32_t)data.size();
  std::cout << "Read " << N << " bytes from " << inFile << "\n";

  // 2) Build frequency table
  static uint32_t freq[256];
  static uint32_t cumFreq[256];
  uint32_t totFreq = 0;
  buildFreqTables(data, freq, cumFreq, totFreq);

  if (totFreq > 65536) {
    std::cerr << "Error: total frequency " << totFreq
              << " exceeds 65536, which Subbotin's code can't handle.\n";
    return 1;
  }

  // 3) Encode
  std::vector<uint8_t> compressed;
  {
    RangeEncoder encoder(compressed);

    // Step A: store N (the data length) as 4 symbols in [0..255]
    // (like a simplistic approach from the example)
    for (int shift = 24; shift >= 0; shift -= 8) {
      uint32_t symbol = (N >> shift) & 0xFF;
      // TOTFREQ=256, freq=1, cumFreq=symbol
      encoder.encode(symbol, 1, 256);
    }

    // Step B: store freq[] as 256 * 2 symbols (hi, lo)
    for (int i = 0; i < 256; i++) {
      uint32_t f = freq[i];
      uint32_t hi = (f >> 8) & 0xFF;
      uint32_t lo = (f) & 0xFF;
      encoder.encode(hi, 1, 256);
      encoder.encode(lo, 1, 256);
    }

    // Step C: encode the actual data
    for (auto b : data) {
      uint32_t f = freq[b];
      uint32_t c = cumFreq[b];
      encoder.encode(c, f, totFreq);
    }

    encoder.endEncode();
  }
  std::cout << "Compressed size = " << compressed.size() << " bytes\n";

  // 4) Write compressed to file
  try {
    writeFile(compressedFile, compressed);
    std::cout << "Wrote compressed data to " << compressedFile << "\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error writing compressed file: " << ex.what() << "\n";
    return 1;
  }

  // 5) Read compressed.bin
  std::vector<uint8_t> compData;
  try {
    compData = readFile(compressedFile);
    std::cout << "Read " << compData.size() << " bytes from " << compressedFile
              << "\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error reading compressed file: " << ex.what() << "\n";
    return 1;
  }

  // 6) Decode
  std::vector<uint8_t> decoded;
  {
    RangeDecoder decoder(compData.data(), (uint32_t)compData.size());

    // A: read length N
    uint32_t readN = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      uint32_t sym = decoder.getFreq(256);
      decoder.decode(sym, 1, 256);
      readN |= (sym << shift);
    }

    // B: read freq[] by decoding 256*2 bytes
    static uint32_t freqD[256];
    std::memset(freqD, 0, sizeof(freqD));
    for (int i = 0; i < 256; i++) {
      uint32_t hi = decoder.getFreq(256);
      decoder.decode(hi, 1, 256);
      uint32_t lo = decoder.getFreq(256);
      decoder.decode(lo, 1, 256);
      freqD[i] = (hi << 8) | lo;
    }

    // sum frequencies
    uint32_t totFreqD = 0;
    for (int i = 0; i < 256; i++) {
      totFreqD += freqD[i];
    }
    if (totFreqD == 0) {
      // fallback if empty
      totFreqD = 1;
      freqD[0] = 1;
    }

    // build cumFreqD
    static uint32_t cumFreqD[256];
    cumFreqD[0] = 0;
    for (int i = 1; i < 256; i++) {
      cumFreqD[i] = cumFreqD[i - 1] + freqD[i - 1];
    }

    // decode readN symbols
    decoded.reserve(readN);
    for (uint32_t count = 0; count < readN; count++) {
      uint32_t f = decoder.getFreq(totFreqD);

      // linear search (could do binary for speed):
      uint8_t s = 0;
      for (int i = 0; i < 256; i++) {
        uint32_t cLo = cumFreqD[i];
        uint32_t cHi = cLo + freqD[i];
        if (f >= cLo && f < cHi) {
          s = static_cast<uint8_t>(i);
          decoder.decode(cLo, freqD[i], totFreqD);
          break;
        }
      }
      decoded.push_back(s);
    }
  }
  std::cout << "Decoded size = " << decoded.size() << " bytes\n";

  // 7) Write decompressed.bin
  try {
    writeFile(outFile, decoded);
    std::cout << "Wrote decompressed data to " << outFile << "\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error writing output file: " << ex.what() << "\n";
    return 1;
  }

  // 8) Compare to check correctness
  if (decoded.size() == data.size() &&
      std::memcmp(decoded.data(), data.data(), data.size()) == 0) {
    std::cout << "Decoded data matches original!\n";
  } else {
    std::cout << "Decoded data DOES NOT MATCH original.\n";
  }

  return 0;
}
