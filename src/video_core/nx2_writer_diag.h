// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_WRITER_DIAG=1: why semaphore acquires wait for queued compute and copy work.
//
// GpuSplit::NoteWriter records no address, so an acquire whose word nothing queued writes still
// waits for the last compute launch or DMA copy (Puller, ReadReadyPosition == WriterPosition). These
// counters tell a false wait (the word already met the acquire before it) from a real one (queued
// work released it), and which engine set the writer position. Compute releases (QMD release slots
// and the report semaphore) are deferred to fences, so a wait that ends unmet at the address of a
// recent compute release counts as a real dependency on compute. Diagnostic only: nothing here
// changes what the emulator does.
namespace VideoCommon::Nx2WriterDiag {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_WRITER_DIAG");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

/// Device addresses of the latest compute releases. Written by the back, read by the front.
struct ComputeReleases {
    std::mutex mutex;
    std::array<u64, 256> addresses{};
    size_t next{};
    u64 total{};
};

inline ComputeReleases& Releases() {
    static ComputeReleases releases;
    return releases;
}

inline void NoteComputeRelease(u64 device_address) {
    auto& releases = Releases();
    std::scoped_lock lock{releases.mutex};
    releases.addresses[releases.next] = (device_address & ~u64{3}) | 1;
    releases.next = (releases.next + 1) % releases.addresses.size();
    ++releases.total;
}

inline bool IsRecentComputeRelease(u64 device_address) {
    auto& releases = Releases();
    const u64 key = (device_address & ~u64{3}) | 1;
    std::scoped_lock lock{releases.mutex};
    for (const u64 address : releases.addresses) {
        if (address == key) {
            return true;
        }
    }
    return false;
}

inline u64 ComputeReleaseCount() {
    auto& releases = Releases();
    std::scoped_lock lock{releases.mutex};
    return releases.total;
}

enum class WriterEngine : size_t { Compute, Dma, Other, Count };
enum class WaitOutcome : size_t {
    MetBefore,   ///< Met before and after the wait: the wait was not needed this time
    MetBeforeLost, ///< Met before, not after: queued work changed the word (a real dependency)
    MetAfter,    ///< Queued work released it: a real dependency
    Unmet,       ///< Still unmet after the wait (fence-deferred release, or another channel/CPU)
    Count
};

/// Acquire waits on the writer position. Front thread only (every channel's puller runs there).
struct AcquireWriterWaits {
    std::array<std::array<u64, static_cast<size_t>(WaitOutcome::Count)>,
               static_cast<size_t>(WriterEngine::Count)>
        counts{};
    std::array<u64, static_cast<size_t>(WriterEngine::Count)> wait_ns{};
    u64 unmet_at_compute_release{};
    std::unordered_map<u32, u64> sources; ///< writer source (engine << 12 | method) -> waits
    u64 total{};
};

inline AcquireWriterWaits& Waits() {
    static AcquireWriterWaits waits;
    return waits;
}

inline void NoteAcquireWriterWait(WriterEngine engine, u32 source, WaitOutcome outcome, u64 wait_ns,
                                  bool at_compute_release) {
    auto& waits = Waits();
    const auto e = static_cast<size_t>(engine);
    ++waits.counts[e][static_cast<size_t>(outcome)];
    waits.wait_ns[e] += wait_ns;
    if (outcome == WaitOutcome::Unmet && at_compute_release) {
        ++waits.unmet_at_compute_release;
    }
    if (waits.sources.size() < 64 || waits.sources.contains(source)) {
        ++waits.sources[source];
    }
    if ((++waits.total & 0xfff) != 0) {
        return;
    }
    static constexpr std::array<const char*, 3> names{"compute", "DMA", "other"};
    for (size_t i = 0; i < names.size(); ++i) {
        const auto& c = waits.counts[i];
        LOG_INFO(HW_GPU,
                 "Acquire writer waits ({}): met before {}, met before then lost {}, met after {}, "
                 "unmet {}, wait {:.1f} ms",
                 names[i], c[0], c[1], c[2], c[3], static_cast<double>(waits.wait_ns[i]) / 1e6);
    }
    LOG_INFO(HW_GPU, "Acquire writer waits: total {}, unmet at a recent compute release {}, compute "
                     "releases {}",
             waits.total, waits.unmet_at_compute_release, ComputeReleaseCount());
    for (const auto& [writer, count] : waits.sources) {
        LOG_INFO(HW_GPU, "Acquire writer wait source {:#x}: {}", writer, count);
    }
}

} // namespace VideoCommon::Nx2WriterDiag
