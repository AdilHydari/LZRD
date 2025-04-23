use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::time::Instant;
use std::{env, process, cmp};

/// The file‐reader/writer wrapper, analogous to the DFile class.
struct DFile<RW: Read + Write> {
    inner: RW,
}

impl<RW: Read + Write> DFile<RW> {
    fn read_symbol(&mut self) -> Option<u8> {
        let mut buf = [0u8; 1];
        match self.inner.read(&mut buf) {
            Ok(n) if n == 1 => Some(buf[0]),
            _ => None, // EOF or error
        }
    }

    fn write_symbol(&mut self, c: u8) -> std::io::Result<()> {
        let buf = [c];
        self.inner.write_all(&buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

const TOP: u32 = 1 << 24; // 0x0100_0000

/// RangeCoder for 32-bit arithmetic coding, storing `low` in a 64-bit integer.
struct RangeCoder<W: Write, R: Read> {
    code: u32,     // Decoding register
    range: u32,    // Current range
    ff_num: u32,   // Number of 0xFF overflows
    cache: u8,     // Last output byte awaiting potential carry
    low: u64,      // 64-bit "low" accumulator
    writer: Option<BufWriter<W>>, // For encoding
    reader: Option<BufReader<R>>, // For decoding
}

impl<W: Write, R: Read> RangeCoder<W, R> {
    fn new() -> Self {
        RangeCoder {
            code: 0,
            range: std::u32::MAX, // 0xFFFF_FFFF
            ff_num: 0,
            cache: 0,
            low: 0,
            writer: None,
            reader: None,
        }
    }

    //--------------------------------------------------
    // Encoding
    //--------------------------------------------------
    fn start_encode(&mut self, out: W) {
        self.low = 0;
        self.ff_num = 0;
        self.cache = 0;
        self.range = std::u32::MAX;
        self.writer = Some(BufWriter::new(out));
    }

    // fn shift_low(&mut self) -> std::io::Result<()> {
    //     // We treat the top byte of low as a "carry" check.
    //     let top_byte = (self.low >> 24) as u8; // top 8 bits
    //     if top_byte != 0xFF {
    //         // No 0xFF overflow => flush the cache + carry
    //         if let Some(ref mut w) = self.writer {
    //             w.write_all(&[(self.cache as u8).wrapping_add(top_byte)])?;
    //             let carry_byte = 0xFFu8.wrapping_add(top_byte);
    //             while self.ff_num > 0 {
    //                 w.write_all(&[carry_byte])?;
    //                 self.ff_num -= 1;
    //             }
    //         }
    //         self.cache = (self.low >> 16) as u8; // Or simply (self.low >> 24) & 0xFF
    //     } else {
    //         // If top byte == 0xFF, we can't finalize carry yet
    //         self.ff_num += 1;
    //     }
    //     // Shift self.low by 8 bits (equivalent to low<<=8 in the C++ code).
    //     // We only keep the lower 32 bits in `low` (in the C++ code, it's `low = (low & 0xFFFFFF)<<8`).
    //     self.low = (self.low & 0xFF_FFFF) << 8;
    //     Ok(())
    // }
    // fn shift_low(&mut self) -> std::io::Result<()> {
    //     if (self.low >> 24) != 0xFF as u64 {
    //         std::io::stdout().write_all(&[self.cache + (self.low >> 32) as u8])?;
    //         while self.ff_num > 0 {
    //             std::io::stdout().write_all(&[0xFF + (self.low >> 32) as u8])?;
    //             self.ff_num -= 1; 
    //         }
    //         self.cache = (self.low >> 24) as u8;
    //     } else {
    //         self.ff_num += 1;
    //     }
    //     self.low = (self.low & 0xFFFFFF as u64) << 8;
    //     Ok(())
    // }

    fn shift_low(&mut self) -> std::io::Result<()> {
        // Compute carry: the high 32 bits (normally 0 or 1)
        let carry = (self.low >> 32) as u8;
        // Get the top byte (bits 24..31)
        let top = (self.low >> 24) as u8;
        if top != 0xFF {
            if let Some(ref mut w) = self.writer {
                // Output Cache plus the computed carry.
                let out_byte = self.cache.wrapping_add(carry);
                w.write_all(&[out_byte])?;
                // Flush all pending overflow bytes:
                while self.ff_num > 0 {
                    let c = 0xFFu8.wrapping_add(carry);
                    w.write_all(&[c])?;
                    self.ff_num -= 1;
                }
            }
            self.cache = (self.low >> 24) as u8;
        } else {
            // Underflow: increment pending overflow counter.
            self.ff_num += 1;
        }
        // Reset low: cast low to u32 (truncating to lower 32 bits) then shift left 8 bits.
        self.low = ((self.low as u32) as u64) << 8;
        Ok(())
    }
    fn encode(&mut self, cum_freq: u32, freq: u32, tot_freq: u32) -> std::io::Result<()> {
        self.range /= tot_freq;
        self.low += (cum_freq as u64) * (self.range as u64);
        self.range *= freq;
        while self.range < TOP {
            self.shift_low()?;
            self.range <<= 8;
        }
        Ok(())
    }

    fn finish_encode(&mut self) -> std::io::Result<()> {
        self.low += 1;
        for _ in 0..5 {
            self.shift_low()?;
        }
        // flush
        if let Some(ref mut w) = self.writer {
            w.flush()?;
        }
        Ok(())
    }

    //--------------------------------------------------
    // Decoding
    //--------------------------------------------------
    fn start_decode(&mut self, inp: R) -> std::io::Result<()> {
        self.code = 0;
        self.range = std::u32::MAX;
        self.reader = Some(BufReader::new(inp));
        // Read 5 bytes to initialize code
        let mut tmp = [0u8; 5];
        if let Some(ref mut rd) = self.reader {
            rd.read_exact(&mut tmp)?;
        }
        for &b in &tmp {
            self.code = (self.code << 8) | (b as u32);
        }
        Ok(())
    }

    fn get_freq(&mut self, tot_freq: u32) -> u32 {
        self.range /= tot_freq;
        self.code / self.range
    }

    fn decode_update(&mut self, cum_freq: u32, freq: u32, tot_freq: u32) -> std::io::Result<()> {
        self.code = self.code.wrapping_sub(cum_freq * self.range);
        self.range *= freq;
        while self.range < TOP {
            // read 1 byte from input
            let next_byte = {
                let mut tmp = [0u8; 1];
                if let Some(ref mut rd) = self.reader {
                    match rd.read_exact(&mut tmp) {
                        Ok(_) => tmp[0],
                        Err(_) => 0, // fallback
                    }
                } else {
                    0
                }
            };
            self.code = (self.code << 8) | (next_byte as u32);
            self.range <<= 8;
        }
        Ok(())
    }
}

// -------------------------------------------------------------
// Context Model
// -------------------------------------------------------------
#[derive(Copy, Clone)]
struct ContextModel {
    esc: i32,
    tot_fr: i32,
    count: [i32; 256],
}

impl ContextModel {
    fn new() -> Self {
        ContextModel {
            esc: 0,
            tot_fr: 0,
            count: [0; 256],
        }
    }
}

const MAX_TOTFR: i32 = 0x3FFF;
static mut STACK: [*mut ContextModel; 2] = [std::ptr::null_mut(), std::ptr::null_mut()];
static mut SP: usize = 0;

// Global "cm" array + context[0], just like the original
static mut CM: [ContextModel; 257] = [ContextModel {
    esc: 0,
    tot_fr: 0,
    count: [0; 256],
}; 257];
static mut CONTEXT: [usize; 1] = [0];

// Very close translation of the original model initialization
unsafe fn init_model() {
    for j in 0..256 {
        CM[256].count[j] = 1;
    }
    CM[256].tot_fr = 256;
    CM[256].esc = 1;
    CONTEXT[0] = 0;
    SP = 0;
}

unsafe fn encode_sym(
    ac: &mut RangeCoder<std::fs::File, std::fs::File>,
    cm_ptr: *mut ContextModel,
    c: i32,
) -> std::io::Result<bool> {
    STACK[SP] = cm_ptr;
    SP += 1;
    let cm_ref = &mut *cm_ptr;
    if cm_ref.count[c as usize] != 0 {
        let mut cum_freq_under = 0;
        for i in 0..c {
            cum_freq_under += cm_ref.count[i as usize];
        }
        ac.encode(
            cum_freq_under as u32,
            cm_ref.count[c as usize] as u32,
            (cm_ref.tot_fr + cm_ref.esc) as u32,
        )?;
        Ok(true)
    } else {
        if cm_ref.esc != 0 {
            ac.encode(
                cm_ref.tot_fr as u32,
                cm_ref.esc as u32,
                (cm_ref.tot_fr + cm_ref.esc) as u32,
            )?;
        }
        Ok(false)
    }
}

unsafe fn decode_sym(
    ac: &mut RangeCoder<std::fs::File, std::fs::File>,
    cm_ptr: *mut ContextModel,
    c: &mut i32,
) -> std::io::Result<bool> {
    STACK[SP] = cm_ptr;
    SP += 1;
    let cm_ref = &mut *cm_ptr;
    if cm_ref.esc == 0 {
        return Ok(false);
    }
    let cum_freq = ac.get_freq((cm_ref.tot_fr + cm_ref.esc) as u32);
    if cum_freq < cm_ref.tot_fr as u32 {
        // find symbol
        let mut cum_freq_under = 0;
        let mut i = 0;
        loop {
            let tmp = cum_freq_under + cm_ref.count[i as usize];
            if tmp as u32 > cum_freq {
                break;
            }
            cum_freq_under = tmp;
            i += 1;
        }
        ac.decode_update(
            cum_freq_under as u32,
            cm_ref.count[i as usize] as u32,
            (cm_ref.tot_fr + cm_ref.esc) as u32,
        )?;
        *c = i;
        Ok(true)
    } else {
        // it is an escape
        ac.decode_update(
            cm_ref.tot_fr as u32,
            cm_ref.esc as u32,
            (cm_ref.tot_fr + cm_ref.esc) as u32,
        )?;
        Ok(false)
    }
}

unsafe fn rescale(cm_ref: &mut ContextModel) {
    cm_ref.tot_fr = 0;
    for i in 0..256 {
        cm_ref.count[i] -= cm_ref.count[i] >> 1; // half
        cm_ref.tot_fr += cm_ref.count[i];
    }
}

unsafe fn update_model(c: i32) {
    while SP > 0 {
        SP -= 1;
        let cm_ptr = STACK[SP];
        let cm_ref = &mut *cm_ptr;
        if cm_ref.tot_fr >= MAX_TOTFR {
            rescale(cm_ref);
        }
        cm_ref.tot_fr += 1;
        if cm_ref.count[c as usize] == 0 {
            cm_ref.esc += 1;
        }
        cm_ref.count[c as usize] += 1;
    }
}

// -------------------------------------------------------------
// encode_file / decode_file
// -------------------------------------------------------------
unsafe fn encode_file(
    ac: &mut RangeCoder<std::fs::File, std::fs::File>,
    mut data: DFile<std::fs::File>,
    mut comp: DFile<std::fs::File>,
) -> std::io::Result<()> {
    init_model();
    ac.start_encode(comp.inner);

    loop {
        let c_opt = data.read_symbol();
        match c_opt {
            Some(cbyte) => {
                let c = cbyte as i32;
                let success = encode_sym(ac, &mut CM[CONTEXT[0]] as *mut ContextModel, c)?;
                if !success {
                    encode_sym(ac, &mut CM[256] as *mut ContextModel, c)?;
                }
                update_model(c);
                CONTEXT[0] = c as usize;
            }
            None => {
                // EOF => write 2 escapes
                let cm_ref = &mut CM[CONTEXT[0]];
                ac.encode(
                    cm_ref.tot_fr as u32,
                    cm_ref.esc as u32,
                    (cm_ref.tot_fr + cm_ref.esc) as u32,
                )?;
                let cm256 = &mut CM[256];
                ac.encode(
                    cm256.tot_fr as u32,
                    cm256.esc as u32,
                    (cm256.tot_fr + cm256.esc) as u32,
                )?;
                break;
            }
        }
    }
    ac.finish_encode()
}

unsafe fn decode_file(
    ac: &mut RangeCoder<std::fs::File, std::fs::File>,
    mut data: DFile<std::fs::File>,
    mut comp: DFile<std::fs::File>,
) -> std::io::Result<()> {
    init_model();
    ac.start_decode(comp.inner)?;

    loop {
        let mut c: i32 = 0;
        let success = decode_sym(ac, &mut CM[CONTEXT[0]] as *mut ContextModel, &mut c)?;
        if !success {
            let success2 = decode_sym(ac, &mut CM[256] as *mut ContextModel, &mut c)?;
            if !success2 {
                break; // no symbol found => stop
            }
        }
        update_model(c);
        CONTEXT[0] = c as usize;
        data.write_symbol(c as u8)?;
    }
    data.flush()
}

// -------------------------------------------------------------
// Helpers
// -------------------------------------------------------------
fn get_file_size(path: &str) -> i64 {
    match std::fs::metadata(path) {
        Ok(meta) => meta.len() as i64,
        Err(_) => -1,
    }
}

// -------------------------------------------------------------
// main
// -------------------------------------------------------------
fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Context compressor (Rust port of M.Smirnov's code).");
        eprintln!("Usage for compression  : c <infile> <outfile>");
        eprintln!("Usage for decompression: d <infile> <outfile>");
        process::exit(1);
    }

    let mode = &args[1];
    let infile = &args[2];
    let outfile = &args[3];

    let data_size = get_file_size(infile);
    if mode.starts_with('c') {
        let start = Instant::now();
        let inf = File::open(infile)?;
        let outf = File::create(outfile)?;
        let data_dfile = DFile { inner: inf };
        let comp_dfile = DFile { inner: outf };

        let mut ac = RangeCoder::new();
        unsafe {
            encode_file(&mut ac, data_dfile, comp_dfile)?;
        }

        let result_size = get_file_size(outfile);
        println!(
            "Original size       {:10} bytes\nEncoded size        {:10} bytes",
            data_size, result_size
        );
        let elapsed = start.elapsed().as_secs_f64();
        println!("Encoding time:      {:.3} sec", elapsed);
        if data_size > 0 {
            let ratio = (result_size as f64) / (data_size as f64);
            println!("Compression ratio:  {:.3}", ratio);
        }
    } else if mode.starts_with('d') {
        println!("Compressed file size {:10} bytes", data_size);
        let start = Instant::now();
        let inf = File::open(infile)?;
        let outf = File::create(outfile)?;
        let comp_dfile = DFile { inner: inf };
        let data_dfile = DFile { inner: outf };

        let mut ac = RangeCoder::new();
        unsafe {
            decode_file(&mut ac, data_dfile, comp_dfile)?;
        }

        let elapsed = start.elapsed().as_secs_f64();
        println!("Decoding time:      {:.3} sec", elapsed);
    } else {
        eprintln!("Invalid command (must be 'c' or 'd').");
    }

    Ok(())
}
