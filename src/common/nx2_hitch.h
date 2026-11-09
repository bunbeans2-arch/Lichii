// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

// LICHII_HITCH=1: one "Hitch:" log line for every frame over 50 ms, with what the GPU thread did
// during it (after tomk134/lichii-custom's recorder, extended with texture cache work).
namespace Common::Nx2Hitch {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("LICHII_HITCH");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

inline std::atomic<u64> g_images_created{};
inline std::atomic<u64> g_images_deleted{};
inline std::atomic<u64> g_gc_deletes{};
inline std::atomic<u64> g_upload_bytes{};
inline std::atomic<u64> g_download_waits{};
inline std::atomic<u64> g_astc_decodes{};
inline std::atomic<u64> g_astc_submits{};
inline std::atomic<u64> g_pipelines{};
inline std::atomic<u64> g_pipeline_us{};

inline void Add(std::atomic<u64>& counter, u64 value = 1) {
    if (Enabled()) {
        counter.fetch_add(value, std::memory_order_relaxed);
    }
}

/// Called once per guest present with the time since the previous one.
inline void NoteFrame(long long frame_us) {
    if (!Enabled()) {
        return;
    }
    struct Snapshot {
        u64 created, deleted, gc, upload, waits, decodes, submits, pipelines, pipeline_us;
    };
    const auto take = [] {
        return Snapshot{g_images_created.load(std::memory_order_relaxed),
                        g_images_deleted.load(std::memory_order_relaxed),
                        g_gc_deletes.load(std::memory_order_relaxed),
                        g_upload_bytes.load(std::memory_order_relaxed),
                        g_download_waits.load(std::memory_order_relaxed),
                        g_astc_decodes.load(std::memory_order_relaxed),
                        g_astc_submits.load(std::memory_order_relaxed),
                        g_pipelines.load(std::memory_order_relaxed),
                        g_pipeline_us.load(std::memory_order_relaxed)};
    };
    static Snapshot last{};
    const Snapshot now = take();
    if (frame_us > 50'000) {
        LOG_INFO(Service,
                 "Hitch: {:.1f} ms frame; images created {} deleted {} (collector {}), uploads "
                 "{:.1f} MiB, {} download waits, ASTC {} decodes in {} submits, {} pipelines "
                 "built on the GPU thread in {:.1f} ms",
                 static_cast<double>(frame_us) / 1000.0, now.created - last.created,
                 now.deleted - last.deleted, now.gc - last.gc,
                 static_cast<double>(now.upload - last.upload) / 1048576.0, now.waits - last.waits,
                 now.decodes - last.decodes, now.submits - last.submits,
                 now.pipelines - last.pipelines,
                 static_cast<double>(now.pipeline_us - last.pipeline_us) / 1000.0);
    }
    last = now;
}

} // namespace Common::Nx2Hitch
