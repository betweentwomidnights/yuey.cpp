#include "yue2/generation.h"

#include <cassert>
#include <stdexcept>
#include <string>

namespace {

bool rejected(const yue2::PlanningHeader & header) {
    try {
        (void)yue2::make_planning_abc_prefix(header);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    assert(yue2::normalize_abc_key("C# minor") == "C#m");
    assert(yue2::normalize_abc_key(" c#:MINOR ") == "C#m");
    assert(yue2::normalize_abc_key("Db major") == "Db");
    assert(yue2::normalize_abc_key("Am") == "Am");
    assert(yue2::normalize_abc_key("F#maj") == "F#");

    const auto prefix = yue2::make_planning_abc_prefix({95, 4, 4, "C# minor"});
    assert(prefix ==
        "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=95\n"
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        "K:C#m\n% intro\n");
    assert(prefix.back() == '\n');

    assert(rejected({0, 4, 4, "C"}));
    assert(rejected({401, 4, 4, "C"}));
    assert(rejected({95, 0, 4, "C"}));
    assert(rejected({95, 4, 3, "C"}));
    assert(rejected({95, 4, 4, "H minor"}));
    assert(rejected({95, 4, 4, "C#m\nV: injected"}));
    return 0;
}
