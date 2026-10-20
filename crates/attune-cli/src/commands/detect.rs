//! `attune detect` — hardware inventory (moved out of `main.rs`).

use anyhow::Result;
use attune_hal::HardwareInfo;

/// Human-readable bytes: pick KB/MB/GB based on magnitude.
fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else if bytes > 0 {
        format!("{} B", bytes)
    } else {
        "unknown".to_string()
    }
}

/// Render the cache row; falls back to "unknown" if CPUID didn't expose the leaf.
fn render_cache(c: &attune_hal::cache::CacheInfo) -> String {
    let any = c.l1d_bytes + c.l1i_bytes + c.l2_bytes + c.l3_bytes;
    if any == 0 {
        return "unknown".to_string();
    }
    let line = if c.line_size_bytes > 0 {
        format!(" | line {} B", c.line_size_bytes)
    } else {
        String::new()
    };
    format!(
        "L1d {} | L1i {} | L2 {} | L3 {}{line}",
        human_bytes(c.l1d_bytes),
        human_bytes(c.l1i_bytes),
        human_bytes(c.l2_bytes),
        human_bytes(c.l3_bytes),
    )
}

fn print_pretty(hw: &HardwareInfo) {
    println!(
        "CPU:        {} ({vendor}, {p}P, {l}L)",
        hw.cpu.brand,
        vendor = hw.cpu.vendor,
        p = hw.cpu.physical_cores,
        l = hw.cpu.logical_cores,
    );
    println!("SIMD:       {}", hw.cpu.simd);
    println!("Cache:      {}", render_cache(&hw.cache));
    println!(
        "NUMA:       {} node{} ({})",
        hw.numa.nodes,
        if hw.numa.nodes == 1 { "" } else { "s" },
        hw.numa.note,
    );
    println!(
        "GPU:        {} ({})",
        if hw.gpu.count == 0 {
            "none detected".to_string()
        } else {
            hw.gpu.count.to_string()
        },
        hw.gpu.note,
    );
    println!(
        "RAM:        {:.1} GB total / {:.1} GB available",
        hw.memory.total_gb, hw.memory.available_gb,
    );
    println!("OS:         {}", hw.os);
}

pub fn run(json: bool) -> Result<()> {
    let hw = HardwareInfo::detect()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&hw)?);
    } else {
        print_pretty(&hw);
    }
    Ok(())
}
