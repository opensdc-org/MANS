#pragma once

#ifdef ENABLE_TIMING

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <initializer_list>

namespace mans {

struct TimingStats {
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::infinity();
    double max_ms = 0.0;
    std::uint64_t count = 0;
};

class TimingCollector {
public:
    static TimingCollector& instance() {
        static TimingCollector inst;
        return inst;
    }

    void begin_run() {
        runs_.emplace_back();
        current_ = &runs_.back().stats;
    }

    void end_run() {
        current_ = nullptr;
    }

    void add(const char* name, double ms) {
        if (!current_) {
            begin_run();
        }
        auto& stats = (*current_)[name];
        stats.total_ms += ms;
        stats.count += 1;
        stats.min_ms = std::min(stats.min_ms, ms);
        stats.max_ms = std::max(stats.max_ms, ms);
    }

    void start(const char* name) {
        auto& active = active_timers();
        const auto now = std::chrono::steady_clock::now();
        auto it = active.find(name);
        if (it != active.end()) {
            std::cerr << "Timing start called twice without stop: " << name << "\n";
            it->second = now;
            return;
        }
        active.emplace(name, now);
    }

    void stop(const char* name) {
        auto& active = active_timers();
        auto it = active.find(name);
        if (it == active.end()) {
            std::cerr << "Timing stop without start: " << name << "\n";
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(end - it->second).count();
        active.erase(it);
        add(name, ms);
    }

    void dump_csv(const std::string& path) const {
        warn_unfinished_timers();
        std::ofstream out(path);
        if (!out) {
            return;
        }
        out << std::fixed << std::setprecision(6);
        std::vector<std::string> keys;
        keys.reserve(32);
        std::unordered_set<std::string> seen;
        seen.reserve(64);

        for (const auto& run : runs_) {
            for (const auto& kv : run.stats) {
                if (seen.insert(kv.first).second) {
                    keys.push_back(kv.first);
                }
            }
        }
        std::sort(keys.begin(), keys.end());
        const std::vector<std::string> priority_keys = {
            "adm/compress_total",
            "adm/decompress_total"
        };
        for (auto it = priority_keys.rbegin(); it != priority_keys.rend(); ++it) {
            const auto key_it = std::find(keys.begin(), keys.end(), *it);
            if (key_it != keys.end()) {
                const std::string key = *key_it;
                keys.erase(key_it);
                keys.insert(keys.begin(), key);
            }
        }

        out << "run_index";
        for (const auto& key : keys) {
            out << "," << key << "_total_ms"
                << "," << key << "_count"
                << "," << key << "_avg_ms";
        }
        out << "\n";

        for (std::size_t i = 0; i < runs_.size(); ++i) {
            const auto& stats = runs_[i].stats;

            out << (i + 1);
            for (const auto& key : keys) {
                const auto s = get_stats(stats, key.c_str());
                out << "," << s.total_ms
                    << "," << s.count
                    << "," << avg_ms(s.total_ms, s.count);
            }
            out << "\n";
        }
    }

    const std::unordered_map<std::string, TimingStats>* last_run_stats() const {
        if (runs_.empty()) {
            return nullptr;
        }
        return &runs_.back().stats;
    }

    TimingStats last_run_stat(const char* name) const {
        const auto* stats = last_run_stats();
        if (!stats) {
            return TimingStats{};
        }
        return get_stats(*stats, name);
    }

    double last_run_sum_ms(std::initializer_list<const char*> names) const {
        const auto* stats = last_run_stats();
        if (!stats) {
            return 0.0;
        }
        double total = 0.0;
        for (const auto* name : names) {
            total += get_stats(*stats, name).total_ms;
        }
        return total;
    }

    double last_run_sum_prefix_ms(const char* prefix) const {
        const auto* stats = last_run_stats();
        if (!stats) {
            return 0.0;
        }
        return sum_prefix(*stats, prefix).total_ms;
    }

    void reset() {
        runs_.clear();
        current_ = nullptr;
    }

private:
    struct RunData {
        std::unordered_map<std::string, TimingStats> stats;
    };

    TimingCollector() = default;

    static TimingStats get_stats(const std::unordered_map<std::string, TimingStats>& stats,
                                 const char* name) {
        auto it = stats.find(name);
        if (it == stats.end()) {
            return TimingStats{};
        }
        return it->second;
    }

    struct SumStats {
        double total_ms = 0.0;
        std::uint64_t count = 0;
    };

    static SumStats sum_prefix(const std::unordered_map<std::string, TimingStats>& stats,
                               const char* prefix) {
        const std::string prefix_str(prefix);
        SumStats total{};
        for (const auto& kv : stats) {
            if (kv.first.rfind(prefix_str, 0) == 0) {
                total.total_ms += kv.second.total_ms;
                total.count += kv.second.count;
            }
        }
        return total;
    }

    static double sum_all_ms(const std::unordered_map<std::string, TimingStats>& stats) {
        double total = 0.0;
        for (const auto& kv : stats) {
            total += kv.second.total_ms;
        }
        return total;
    }

    static double avg_ms(double total_ms, std::uint64_t count) {
        return count ? (total_ms / static_cast<double>(count)) : 0.0;
    }

    static std::unordered_map<std::string, std::chrono::steady_clock::time_point>&
    active_timers() {
        thread_local std::unordered_map<std::string, std::chrono::steady_clock::time_point> active;
        return active;
    }

    static void warn_unfinished_timers() {
        auto& active = active_timers();
        if (active.empty()) {
            return;
        }
        for (const auto& kv : active) {
            std::cerr << "Timing stop missing for: " << kv.first << "\n";
        }
    }

    std::vector<RunData> runs_;
    std::unordered_map<std::string, TimingStats>* current_ = nullptr;
};

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name)
        : name_(name),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(end - start_).count();
        TimingCollector::instance().add(name_, ms);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};

class RunScope {
public:
    RunScope() { TimingCollector::instance().begin_run(); }
    ~RunScope() { TimingCollector::instance().end_run(); }

    RunScope(const RunScope&) = delete;
    RunScope& operator=(const RunScope&) = delete;
};

} // namespace mans

#define MANS_TIMING_SCOPE(name) mans::ScopedTimer _mans_timer_##__LINE__(name)
#define MANS_TIMING_START(name) mans::TimingCollector::instance().start(name)
#define MANS_TIMING_STOP(name) mans::TimingCollector::instance().stop(name)
#define MANS_TIMING_DUMP(path) mans::TimingCollector::instance().dump_csv(path)
#define MANS_TIMING_RUN_SCOPE() mans::RunScope _mans_run_##__LINE__{}
#define MANS_TIMING_RESET() mans::TimingCollector::instance().reset()

#else

#define MANS_TIMING_SCOPE(name) do {} while (0)
#define MANS_TIMING_START(name) do {} while (0)
#define MANS_TIMING_STOP(name) do {} while (0)
#define MANS_TIMING_DUMP(path) do {} while (0)
#define MANS_TIMING_RUN_SCOPE() do {} while (0)
#define MANS_TIMING_RESET() do {} while (0)

#endif
