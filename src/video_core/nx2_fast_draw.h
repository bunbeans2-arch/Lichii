// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

/// A draw that follows a draw of the same pipeline with only light state changes in between takes
/// the delta path. It keeps the previous draw's pipeline, image views, samplers and texture buffer
/// bindings, and resolves and binds every buffer (index, vertex, uniform, storage, texture) again
/// as a full resolve does, so guest memory written without a notification is streamed as in a
/// full resolve.
/// Light changes are draw parameters, index buffer and vertex stream registers, constant buffer
/// binds and inline constant buffer data (collected in g_delta: the rasterizer resolves in full
/// when they reach the pipeline's texture handles at their GPU addresses or through another
/// mapping of their device memory). Anything else that could change what a resolve produces bumps
/// a serial: a 3D register that changes, any 3D method with side effects (uploads, reports,
/// clears, barriers, ...), constant buffer data reaching the texture or sampler headers (the same
/// way), any other engine, deferred calls on the split's back (except the ones marked
/// GpuSplit::FastDrawNeutral), GPU::TickWork requests, texture cache writes, unmaps and
/// deletions, shader removals, fence releases, HLE macros and frames. The rasterizer takes the
/// delta path only while the serial, the GPU memory map, the command buffer, the render pass and
/// the bound pipeline are the ones the previous draw left.
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the pipeline, views,
/// samplers or kept buffer bindings differ.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    DeltaOverflow,
    Map,
    AliasWrite,   ///< TextureCache::MarkModification of an image with aliases (the v6 guard)
    FenceRelease, ///< FenceManager released fences (Fence is the puller's acquires only)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "acquire",       "indirect reset", "frame",           "texture header data",
    "delta overflow", "memory map",    "alias write",     "fence release",
};

/// EDEN_NX2_FAST_DRAW=0 turns fast (delta) draws off (kill switch, A/B). Read once at start.
[[nodiscard]] inline bool EnabledAtStart() {
    const char* const value = std::getenv("EDEN_NX2_FAST_DRAW");
    return value == nullptr || value[0] != '0';
}

inline std::atomic<bool> g_enabled{EnabledAtStart()};

/// Any thread: whether fast draws run (always, unless a test turned them off). The rasterizer
/// latches it per draw (on the thread that draws) and starts with a full resolve when it turns on.
[[nodiscard]] inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Tests only.
inline void SetEnabledForTest(bool on) {
    g_enabled.store(on, std::memory_order_relaxed);
}

/// EDEN_NX2_FAST_DRAW_VERIFY=1, read once.
[[nodiscard]] inline bool VerifyEnabled() {
    static const bool verify = [] {
        const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// v7 switches, read once at start. All off is v6.
struct Switches {
    bool stats;    ///< EDEN_NX2_FD_STATS=1   log the fast draw lines without EDEN_NX2_GPU_PROFILE
    bool census;   ///< EDEN_NX2_FD_CENSUS=1  break census (turns shortcut off)
    bool alias;    ///< EDEN_NX2_FD_ALIAS=1   alias writes rely on the alias tick check
    bool query;    ///< EDEN_NX2_FD_QUERY=1   ZPass reports and counter resets are neutral
    bool cond;     ///< EDEN_NX2_FD_COND=1    conditional rendering is neutral
    bool barrier;  ///< EDEN_NX2_FD_BARRIER=1 fragment and tiled cache barriers are neutral
    bool shortcut; ///< EDEN_NX2_FD_SHORT=1   no break or note work once the chain is broken
    bool vertex_id; ///< EDEN_NX2_FD_VERTEXID=1 vertex_id_base (0x446) writes are neutral
};
[[nodiscard]] inline Switches ReadSwitches() {
    const auto on = [](const char* name) {
        const char* const value = std::getenv(name);
        return value != nullptr && value[0] == '1';
    };
    return {on("EDEN_NX2_FD_STATS"), on("EDEN_NX2_FD_CENSUS"),  on("EDEN_NX2_FD_ALIAS"),
            on("EDEN_NX2_FD_QUERY"), on("EDEN_NX2_FD_COND"),    on("EDEN_NX2_FD_BARRIER"),
            on("EDEN_NX2_FD_SHORT"), on("EDEN_NX2_FD_VERTEXID")};
}
inline Switches g_switches = ReadSwitches();
[[nodiscard]] inline const Switches& Sw() {
    return g_switches;
}

/// Census only: registers the work a delta draw skips reads (descriptor tables, headers,
/// samplers), plus the vertex input and render target state. Method index = byte offset / 4
/// (maxwell_3d.h ASSERT_REG_POSITION; maxwell_3d.cpp static_asserts the ones named here).
[[nodiscard]] constexpr bool IsHardRegister(u32 m) {
    return m == 0x87                                                // invalidate_shader_cache
           || (m >= 0xE0 && m <= 0x1D1)                             // transform feedback .. enabled
           || (m >= 0x200 && m <= 0x27F)                            // rt[8]
           || m == 0x369 || m == 0x3C4                              // no-wfi invalidate, select_texture_headers
           || (m >= 0x3F8 && m <= 0x3FC)                            // zeta
           || m == 0x43D                                            // unbind_all_constant_buffers
           || (m >= 0x453 && m <= 0x456)                            // mesh control / shaders
           || (m >= 0x458 && m <= 0x477)                            // vertex_attrib_format
           || (m >= 0x483 && m <= 0x484) || m == 0x487              // sampler/header invalidate, rt_control
           || (m >= 0x48A && m <= 0x48D)                            // zeta_size, sampler_binding
           || (m >= 0x4CC && m <= 0x4CE) || (m >= 0x509 && m <= 0x50A) // cache invalidates
           || m == 0x54E                                            // zeta_enable
           || (m >= 0x557 && m <= 0x559) || (m >= 0x55D && m <= 0x55F) // tex_sampler, tex_header
           || m == 0x574                                            // anti_alias_samples_mode
           || (m >= 0x620 && m <= 0x63F)                            // vertex_stream_instances
           || (m >= 0x700 && m < 0x780 && (m % 4 == 0 || m % 4 == 3)) // stream stride/enable, frequency
           || m == 0x982 || (m >= 0xA00 && m <= 0xA7F);             // bindless slot, stream_out_layout
}

/// EDEN_NX2_FD_CENSUS: classes of the breaks between two draws (one bit each).
enum CensusClass : u32 {
    CensusQuery = 1U << 0,
    CensusCond = 1U << 1,
    CensusBarrier = 1U << 2,
    CensusAlias = 1U << 3,
    CensusFenceRelease = 1U << 4,
    CensusHandleData = 1U << 5,
    CensusSoftRegister = 1U << 6,
    CensusHardRegister = 1U << 7,
    CensusMethod = 1U << 8,
    CensusMacro = 1U << 9,
    CensusCall = 1U << 10,
    CensusAcquire = 1U << 11,
    CensusInvalidation = 1U << 12,
    CensusOtherEngine = 1U << 13,
    CensusOther = 1U << 14,
    CensusVertexIdBase = 1U << 15,
};
inline constexpr u32 CensusBits = 16;
inline constexpr std::array<const char*, CensusBits> census_names{
    "query",       "cond",   "barrier", "alias",       "fence release",
    "handle data", "soft register", "hard register", "method", "macro",
    "call",        "acquire", "invalidation", "other engine", "other",
    "vertex id base",
};

[[nodiscard]] constexpr u32 CensusClassOf(Reason reason, u32 m) {
    switch (reason) {
    case Reason::AliasWrite:
        return CensusAlias;
    case Reason::FenceRelease:
        return CensusFenceRelease;
    case Reason::DeltaOverflow:
    case Reason::TextureHeaders:
        return CensusHandleData;
    case Reason::Call:
        return CensusCall;
    case Reason::Fence:
        return CensusAcquire;
    case Reason::Invalidation:
        return CensusInvalidation;
    case Reason::OtherEngine:
        return CensusOtherEngine;
    case Reason::Register:
    case Reason::Method:
        break;
    default:
        return CensusOther;
    }
    // report_semaphore, clear_report_value, zpass_pixel_count_enable
    if ((m >= 0x6C0 && m <= 0x6C3) || m == 0x54C || m == 0x545) {
        return CensusQuery;
    }
    // render_enable, render_enable_override
    if ((m >= 0x554 && m <= 0x556) || m == 0x651) {
        return CensusCond;
    }
    // fragment_barrier, tiled_cache_barrier
    if (m == 0x378 || m == 0x3DF) {
        return CensusBarrier;
    }
    if (m == 0x446) { // vertex_id_base
        return CensusVertexIdBase;
    }
    if (reason == Reason::Method) {
        return m >= 0xE00 ? CensusMacro : CensusMethod;
    }
    return IsHardRegister(m) ? CensusHardRegister : CensusSoftRegister;
}

inline std::atomic<u64> g_serial{1};
/// The last break's reason and, for Register and Method, its 3D method (diagnostics).
inline std::atomic<u32> g_last_break{};
/// EDEN_NX2_FD_CENSUS: classes of the breaks since the drawing thread last read it.
inline std::atomic<u32> g_census{};
/// The thread that draws, and the serial its last draw latched (thread-local: only that thread
/// skips work, EDEN_NX2_FD_SHORT).
inline thread_local bool t_drawing{};
inline thread_local u64 t_draw_snapshot{};

/// EDEN_NX2_FD_SHORT: on the drawing thread, the serial moved since its last draw latched it, so
/// the next draw resolves in full whatever happens: more breaks and notes change no comparison.
/// The candidate draw always holds the serial its own PrepareDraw latched (NoteResolvedDraw),
/// and TakeReusableDraw clears it, so a moved serial means the next take returns StateChanged.
[[nodiscard]] inline bool AlreadyBroken() {
    return t_drawing && Sw().shortcut && !Sw().census &&
           g_serial.load(std::memory_order_relaxed) != t_draw_snapshot;
}

inline void Break(Reason reason, u32 method = 0) {
    if (!Enabled() || AlreadyBroken()) {
        return;
    }
    g_serial.fetch_add(1, std::memory_order_relaxed);
    g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    if (Sw().census) {
        g_census.fetch_or(CensusClassOf(reason, method), std::memory_order_relaxed);
    }
}

/// PrepareDraw, once the caches are locked and the draw's serial is read.
inline void LatchDraw(u64 serial) {
    t_drawing = true;
    t_draw_snapshot = serial;
}

inline u64 Serial() {
    return g_serial.load(std::memory_order_relaxed);
}

/// The device memory a GPU range maps to. A write through another GPU address that maps the
/// same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless that range of memory was translated last
    /// and memory's mappings haven't changed since.
    void Update(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const;

private:
    const Tegra::MemoryManager* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    std::vector<std::pair<DAddr, DAddr>> spans; ///< Sorted, disjoint [begin, end) pairs
};

/// Light changes since the last draw that can reach texture handles. Written and read on the
/// thread that runs the 3D engine and the rasterizer (the split's back).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    /// Constant buffer data written, by GPU address (unmapped addresses included)...
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    /// ...and by the device address it landed on.
    std::array<DAddr, MaxWrites> device_begin{};
    std::array<DAddr, MaxWrites> device_end{};
    u32 num_device_writes{};

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    /// Whether data written reaches [address, address + size), whose device memory is spans: at
    /// one of its GPU addresses, or through any GPU address that maps its device memory.
    [[nodiscard]] bool Reaches(GPUVAddr address, u64 size, const DeviceSpans& spans) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        for (u32 write = 0; write < num_device_writes; ++write) {
            if (spans.Intersects(device_begin[write], device_end[write])) {
                return true;
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// Appends [begin, end) to a delta's range list, or extends its last range when that ends at
/// begin. False when the list is full.
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return true;
    }
    if (count == Delta::MaxWrites) {
        return false;
    }
    begins[count] = begin;
    ends[count] = end;
    ++count;
    return true;
}

inline void NoteBind(size_t stage, u32 slot) {
    if (Enabled() && !AlreadyBroken()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// Inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (!Enabled() || AlreadyBroken()) {
        return;
    }
    Delta& delta = g_delta;
    if (!AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end)) {
        Break(Reason::DeltaOverflow);
    }
}

/// The device addresses [begin, end) that inline constant buffer data landed on.
inline void NoteDeviceData(DAddr begin, DAddr end) {
    if (!Enabled() || AlreadyBroken()) {
        return;
    }
    Delta& delta = g_delta;
    if (!AppendRange(delta.device_begin, delta.device_end, delta.num_device_writes, begin, end)) {
        Break(Reason::DeltaOverflow);
    }
}

} // namespace VideoCommon::Nx2FastDraw
