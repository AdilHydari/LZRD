// compress_floats.cpp
// Generates a circle of float points, delta-encodes, compresses with rangecoder,
// writes compressed file, then reads back, decompresses, reconstructs floats,
// undo delta, verifies integrity, and writes decompressed floats to binary.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "rangecoder.h"

static constexpr float PI = 3.14159265358979323846f;

// Define a tolerance for float comparison
static constexpr float FLOAT_EPSILON = 1e-6f;

struct point {
    float x, y;
    bool operator==(const point& o) const {
        return x == o.x && y == o.y;
    }
};

// Generate n points on a circle centered at (x0,y0) with radius r
std::vector<point> generate_circle(float x0, float y0, float r, int n) {
    std::vector<point> pts;
    pts.reserve(n);
    float dt = 2*PI / n;
    float theta = 0;
    for (int i = 0; i < n; ++i, theta += dt) {
        pts.push_back({
            x0 + r * std::cos(theta),
            y0 + r * std::sin(theta)
        });
    }
    return pts;
}

// Write raw bytes to file
void writeFile(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open " + path);
    ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

// Read entire file into bytes
std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open " + path);
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(ifs), {}
    );
}

// Write vector<point> to binary file (float x,y pairs)
void writePoints(const std::string& path, const std::vector<point>& pts) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open " + path);
    ofs.write(reinterpret_cast<const char*>(pts.data()), pts.size()*sizeof(point));
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_points> <compressed.bin> <decompressed.bin>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::string compPath = argv[2];
    std::string decompPath = argv[3];

    // 1) Generate circle and delta-encode
    auto points = generate_circle(0.0f, 0.0f, 1.0f, n);
    std::cout << "DEBUG: Generated " << n << " points. First few:\n";
    for (int i=0; i < std::min(n, 5); ++i) {
        std::cout << "  [" << i << "] x=" << points[i].x << ", y=" << points[i].y << "\n";
    }

    std::vector<point> delta = points;
    for (int i = n-1; i > 0; --i) {
        delta[i].x -= delta[i-1].x;
        delta[i].y -= delta[i-1].y;
    }
    std::cout << "DEBUG: Delta-encoded points. First few:\n";
    for (int i=0; i < std::min(n, 5); ++i) {
        std::cout << "  [" << i << "] dx=" << delta[i].x << ", dy=" << delta[i].y << "\n";
    }

    // 2) Flatten to bytes (assuming little-endian host)
    std::vector<uint8_t> raw;
    raw.reserve(delta.size() * sizeof(point));
    for (auto& p : delta) {
        // Directly insert the bytes of the point struct
        const uint8_t* point_bytes = reinterpret_cast<const uint8_t*>(&p);
        raw.insert(raw.end(), point_bytes, point_bytes + sizeof(point));
    }
    std::cout << "DEBUG: Flattened to " << raw.size() << " raw bytes (little-endian). First few:\n  ";
    for (size_t i = 0; i < std::min(raw.size(), size_t{16}); ++i) {
        std::printf("%02X ", raw[i]);
    }
    std::cout << "\n";

    // --- Compress ---
    printf("Compressing using LZRC...\n");
    std::vector<uint8_t> comp = rangecoder::encode_lzrc(raw); // FIX: Use encode_lzrc
    printf("  Raw size: %zu bytes\n", raw.size());
    printf("  Compressed size: %zu bytes\n", comp.size());
    double ratio = (double)comp.size() / raw.size();
    writeFile(compPath, comp);
    std::cout << "DEBUG: Wrote compressed data to " << compPath << "\n";
    std::cout << "Compressed " << raw.size() << " bytes into "
              << comp.size() << " bytes\n";

    // 4) Read & decompress
    auto compRead = readFile(compPath);
    std::cout << "DEBUG: Read " << compRead.size() << " bytes from " << compPath << "\n";
    // --- Decompress ---
    printf("Decompressing...\n");
    std::vector<uint8_t> rawDec = rangecoder::decode_lzrc(compRead); // FIX: Use decode_lzrc
    printf("  Decompressed size: %zu bytes\n", rawDec.size());

    // --- Verify ---
    if (rawDec.size() != raw.size()) {
        std::cerr << "Decompressed size mismatch\n";
        return 1;
    }
    std::cout << "DEBUG: Decompressed to " << rawDec.size() << " raw bytes. First few:\n  ";
    for (size_t i = 0; i < std::min(rawDec.size(), size_t{16}); ++i) {
        std::printf("%02X ", rawDec[i]);
    }
    std::cout << "\n";

    // *** ADDED: Full raw byte comparison ***
    if (raw == rawDec) {
        std::cout << "DEBUG: Full raw byte comparison PASSED.\n";
    } else {
        std::cerr << "ERROR: Full raw byte comparison FAILED.\n";
        // Optional: Find first mismatching byte
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] != rawDec[i]) {
                std::cerr << "  First raw byte mismatch at index " << i
                          << ": Original=0x" << std::hex << (int)raw[i]
                          << ", Decompressed=0x" << (int)rawDec[i] << std::dec << "\n";
                break;
            }
        }
        // It might still be useful to continue to see the float recon error
        // return 1; 
    }

    // 5) Reconstruct points from bytes (assuming little-endian host)
    std::vector<point> recon;
    recon.reserve(n);
    for (size_t i = 0; i < rawDec.size(); i += sizeof(point)) {
        point p;
        // Directly copy bytes back into point struct
        std::memcpy(&p, rawDec.data() + i, sizeof(point));
        recon.push_back(p);
    }
    std::cout << "DEBUG: Reconstructed delta points (little-endian). First few:\n";
    for (int i=0; i < std::min(n, 5); ++i) {
        std::cout << "  [" << i << "] dx=" << recon[i].x << ", dy=" << recon[i].y << "\n";
    }

    // undo delta
    for (int i = 1; i < n; ++i) {
        recon[i].x += recon[i-1].x;
        recon[i].y += recon[i-1].y;
    }
    std::cout << "DEBUG: Final reconstructed points (after undoing delta). First few:\n";
    for (int i=0; i < std::min(n, 5); ++i) {
        std::cout << "  [" << i << "] x=" << recon[i].x << ", y=" << recon[i].y << "\n";
    }

    // 6) Verify and write decompressed points
    bool mismatch_found = false;
    if (recon.size() != points.size()) {
        mismatch_found = true;
        std::cerr << "ERROR: Size mismatch after reconstruction! Original=" << points.size()
                  << ", Recon=" << recon.size() << "\n";
    } else {
        for (size_t i = 0; i < points.size(); ++i) {
            bool x_diff = std::abs(recon[i].x - points[i].x) > FLOAT_EPSILON;
            bool y_diff = std::abs(recon[i].y - points[i].y) > FLOAT_EPSILON;
            if (x_diff || y_diff) {
                 if (!mismatch_found) { // Print header only once
                     std::cerr << "ERROR: data mismatch (using epsilon=" << FLOAT_EPSILON << ")\n";
                 }
                 mismatch_found = true;
                 std::cerr << "  Mismatch at index " << i << ":\n";
                 std::cerr << "    Original: x=" << points[i].x << ", y=" << points[i].y << "\n";
                 std::cerr << "    Recon:    x=" << recon[i].x << ", y=" << recon[i].y << "\n";
                 // Optionally break after first mismatch or limit output
                 if (i > 10) { // Limit detailed output
                      std::cerr << "    (Further mismatches omitted...)\n";
                      break;
                 }
            }
        }
    }

    if (!mismatch_found) {
        std::cout << "OK: decompressed matches original (within epsilon=" << FLOAT_EPSILON << ")\n";
        writePoints(decompPath, recon);
        return 0;
    } else {
        return 1;
    }
}
