// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"

namespace VideoCommon {

/// EDEN_NX2_TEX_PAGES=1 (off by default): a draw or dispatch translates each guest page it reads
/// texture handles from once (GuestPageCache), instead of once per handle word. From #717
/// (stephenjayakar), handle reads only; #717's TIC/TSC page reuse belongs to its binding memo.
[[nodiscard]] inline bool TexPagesEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_TEX_PAGES");
        return value != nullptr && value[0] != 0 && value[0] != '0';
    }();
    return enabled;
}

/// The host pointers of the guest pages one draw or dispatch reads its texture handles from, each
/// page translated once instead of once per read. The reads then all see the mappings as they were
/// at the draw's first read: the guest can't remap memory a pending draw reads without racing the
/// GPU itself, and a host pointer points into guest RAM, which is never freed. A cache lives for
/// one draw or dispatch.
struct GuestPageCache {
    static constexpr size_t NUM_ENTRIES = 8;
    static constexpr GPUVAddr PAGE_BYTES = GPUVAddr{1} << 12; ///< Core::Memory::YUZU_PAGESIZE
    static constexpr GPUVAddr NO_PAGE = ~GPUVAddr{0};          ///< never a page base

    std::array<GPUVAddr, NUM_ENTRIES> pages{NO_PAGE, NO_PAGE, NO_PAGE, NO_PAGE,
                                            NO_PAGE, NO_PAGE, NO_PAGE, NO_PAGE};
    std::array<const u8*, NUM_ENTRIES> hosts{};
    u32 next{};

    /// MemoryManager::Read<u32>(addr), from the translated page when it can. A page that is not
    /// mapped, or that the syncpoint shim overlaps (Read<u32> answers those words from the host1x
    /// syncpoints, not from memory), is remembered as nullptr and read the slow way, so every value
    /// equals what Read<u32> returns.
    template <class MemoryManager>
    [[nodiscard]] u32 ReadWord(const MemoryManager& gpu_memory, GPUVAddr addr) noexcept {
        const GPUVAddr offset = addr & (PAGE_BYTES - 1);
        if (offset + sizeof(u32) <= PAGE_BYTES) {
            const GPUVAddr page = addr - offset;
            const u8* host = nullptr;
            bool found = false;
            for (size_t index = 0; index < NUM_ENTRIES; ++index) {
                if (pages[index] == page) {
                    host = hosts[index];
                    found = true;
                    break;
                }
            }
            if (!found) {
                host = gpu_memory.GetPointer(page);
                if (host != nullptr && gpu_memory.OverlapsSyncpointShim(page, PAGE_BYTES)) {
                    host = nullptr;
                }
                const size_t slot = next++ % NUM_ENTRIES;
                pages[slot] = page;
                hosts[slot] = host;
            }
            if (host != nullptr) {
                u32 value;
                std::memcpy(&value, host + offset, sizeof(value));
                return value;
            }
        }
        return gpu_memory.template Read<u32>(addr);
    }
};

} // namespace VideoCommon
