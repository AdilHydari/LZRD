// compress_floats_lzma.cpp
// Generates circle data, delta-encodes, flattens, and compresses using LZMA.
//
// Requires liblzma-dev: sudo apt-get install liblzma-dev
// Compile with: g++ -std=c++17 -O2 -o compress_floats_lzma compress_floats_lzma.cpp -llzma -Wall -Wextra

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// LZMA header
#include <lzma.h>

// --- Code duplicated from compress_floats.cpp (Consider refactoring to common utils) ---

static constexpr float PI = 3.14159265358979323846f;

struct Point {
    float x, y;
};

// Generates N points roughly on a unit circle
std::vector<Point> generate_circle(int N) {
    std::vector<Point> points;
    points.reserve(N);
    for (int i = 0; i < N; ++i) {
        float angle = 2.0f * PI * static_cast<float>(i) / static_cast<float>(N);
        points.push_back({std::cos(angle), std::sin(angle)});
    }
    return points;
}

// Simple delta encoding: p[i] = p[i] - p[i-1]
std::vector<Point> delta_encode(const std::vector<Point>& points) {
    if (points.empty()) return {};
    std::vector<Point> deltas = points; // Start with a copy
    for (size_t i = points.size() - 1; i > 0; --i) {
        deltas[i].x -= points[i-1].x;
        deltas[i].y -= points[i-1].y;
    }
    return deltas;
}

// Flatten points to little-endian byte stream
std::vector<uint8_t> flatten_points_le(const std::vector<Point>& points) {
    std::vector<uint8_t> raw;
    raw.resize(points.size() * sizeof(Point)); // 8 bytes per point
    for (size_t i = 0; i < points.size(); ++i) {
        // Assuming little-endian architecture or handling is needed otherwise
        std::memcpy(raw.data() + i * sizeof(Point), &points[i], sizeof(Point));
    }
    return raw;
}

// Helper to write a byte vector to a file
void writeFile(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    outFile.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// --- End of duplicated code ---


// Function to compress data using LZMA easy encoder
std::vector<uint8_t> lzma_compress(const std::vector<uint8_t>& input_data, uint32_t preset = LZMA_PRESET_DEFAULT) {
    if (input_data.empty()) {
        return {};
    }

    // Estimate buffer size - LZMA can slightly expand incompressible data
    size_t output_buffer_size = input_data.size() + input_data.size() / 3 + 128;
    std::vector<uint8_t> output_buffer(output_buffer_size);

    size_t output_pos = 0; // Stores the actual size of compressed data

    // Initialize the encoder
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_easy_encoder(&strm, preset, LZMA_CHECK_CRC64);
    if (ret != LZMA_OK) {
        throw std::runtime_error("lzma_easy_encoder initialization failed: " + std::to_string(ret));
    }

    // Set input and output buffers
    strm.next_in = input_data.data();
    strm.avail_in = input_data.size();
    strm.next_out = output_buffer.data();
    strm.avail_out = output_buffer.size();

    // Perform compression
    ret = lzma_code(&strm, LZMA_FINISH);

    if (ret != LZMA_STREAM_END) {
         // Check for errors or insufficient buffer space
        if (ret == LZMA_MEMLIMIT_ERROR) {
             throw std::runtime_error("LZMA memory usage limit exceeded");
        } else if (ret == LZMA_MEM_ERROR) {
             throw std::runtime_error("LZMA memory allocation failed");
        } else if (ret == LZMA_DATA_ERROR) {
             throw std::runtime_error("LZMA found data error");
        } else if (strm.avail_out == 0 && ret != LZMA_OK) {
             // This could happen if output_buffer_size was too small, but easy encoder should handle it.
             // However, better safe than sorry.
             throw std::runtime_error("LZMA compression ran out of output buffer space.");
        } else {
             throw std::runtime_error("LZMA compression failed with unexpected code: " + std::to_string(ret));
        }
    }

    output_pos = strm.total_out;

    // Clean up the encoder
    lzma_end(&strm);

    // Resize output vector to actual compressed size
    output_buffer.resize(output_pos);
    return output_buffer;
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_points> <output_lzma_file>\n";
        return 1;
    }

    int numPoints = 0;
    std::string compPath;
    try {
        numPoints = std::stoi(argv[1]);
        compPath = argv[2];
        if (numPoints <= 0) {
            throw std::invalid_argument("Number of points must be positive.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    try {
        // 1) Generate points
        std::vector<Point> points = generate_circle(numPoints);
        std::cout << "DEBUG: Generated " << points.size() << " points.\n";

        // 2) Delta-encode
        std::vector<Point> deltas = delta_encode(points);
        std::cout << "DEBUG: Delta-encoded points.\n";

        // 3) Flatten to bytes (little-endian)
        std::vector<uint8_t> raw = flatten_points_le(deltas);
        std::cout << "DEBUG: Flattened to " << raw.size() << " raw bytes.\n";

        // 4) Compress using LZMA
        std::cout << "DEBUG: Compressing with LZMA...\n";
        std::vector<uint8_t> comp = lzma_compress(raw);
        std::cout << "Compressed " << raw.size() << " bytes into " << comp.size() << " bytes (LZMA)\n";


        // 5) Write compressed data
        writeFile(compPath, comp);
        std::cout << "DEBUG: Wrote LZMA compressed data to " << compPath << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
