// SPDX-License-Identifier: MIT
#include "server/prompts.h"

#include "server/json.h"

namespace yue2::server {
namespace {

// Genre and subgenre, mood and energy, the band, then how it moves. Spread
// deliberately: eras, regions and ensemble sizes, so consecutive rolls of the
// dice do not all land in the same decade of the same city. Dance music gets
// the largest share because most of the people rolling are producers.
const std::vector<std::string> kInstrumental = {
    // dance and edm
    "progressive house, euphoric and wide, supersaw chords, plucked arpeggio, rolling bass, big build and drop",
    "melodic techno, dark and hypnotic, pulsing arpeggio, driving bassline, crisp hats, slow filter sweeps",
    "tech house, groovy and minimal, punchy kick, rolling bassline, shuffled hats, percussive stabs",
    "big room house, festival-sized and triumphant, huge kick, lead synth riff, snare roll build, massive drop",
    "bass house, swaggering and heavy, growling bassline, punchy kick, shuffled hats, organ stabs",
    "electro house, gritty and bouncy, distorted bass stabs, four-on-the-floor kick, filter sweeps",
    "future house, sleek and bouncy, metallic pluck bass, crisp claps, bright chords",
    "garage house, soulful and bumping, organ bassline, swung hats, piano chords",
    "minimal house, crisp and understated, clicky percussion, deep bass, sparse chords",
    "tropical house, breezy and warm, marimba melody, soft kick, plucked guitar, airy pads",
    "afro house, deep and rolling, hand percussion, warm chords, hypnotic bassline, shakers",
    "amapiano, smooth and bouncy, log drum bass, shakers, jazzy piano chords, soft pads",
    "nu-disco, glossy and funky, slap bass, filtered strings, four-on-the-floor, Rhodes stabs",
    "french house, sun-soaked and pumping, filtered disco loop, sidechained chords, punchy kick",
    "techno, pounding and industrial, distorted kick, clanging percussion, acid line, relentless groove",
    "trance, uplifting and soaring, supersaw lead, rolling offbeat bass, gated pads, long breakdown",
    "hardstyle, pounding and anthemic, distorted kick, screeching lead, euphoric chords",
    "eurodance, euphoric and nostalgic, bright piano riff, pumping bass, four-on-the-floor, trance pads",
    "happy hardcore, giddy and relentless, piano stabs, hoover synths, pounding kick",
    "future bass, bright and bittersweet, wide supersaw chords, sidechained pads, pitched plucks, snappy snare",
    "melodic dubstep, emotional and huge, lush pads, soaring lead, half-time drums, growling bass",
    "dubstep, menacing and heavy, wobble bass, half-time drums, metallic growls, sparse pads",
    "riddim dubstep, stomping and minimal, repetitive wobble bass, half-time drums, sparse hits",
    "neurofunk drum and bass, dark and technical, reese bass, tight breakbeat, metallic stabs",
    "jump up drum and bass, bouncy and rowdy, wobbly bassline, punchy two-step drums, playful synth hook",
    "jungle, raw and rolling, chopped breakbeats, deep sub bass, dub stabs, warm pads",
    "breakcore, chaotic and euphoric, chopped breakbeats, glitchy edits, rave stabs, distorted bass",
    "jersey club, bouncy and playful, triplet kicks, chopped samples, bright synth stabs",
    "moombahton, sweaty and bouncy, reggaeton drums, big synth stabs, heavy bass",
    "phonk, dark and aggressive, cowbell riff, distorted 808, crunchy drums",
    "glitch hop, funky and crunchy, chopped breaks, wobbling bass, glitchy edits, brass stabs",
    "chillwave, hazy and warm, washed-out synths, soft drum machine, dreamy chords",
    // electronic
    "dark minimal techno, hypnotic and airless, analog bass, closed hats, one-bar acid figure",
    "dub techno, submerged and patient, tape-delayed chord stabs, deep bass, brushed hats",
    "deep house, warm and unhurried, Rhodes chords, filtered bass, shuffled hats",
    "electro, brittle and mechanical, 808 toms, synth lead, sharp claps",
    "ambient, weightless and slow, warm pads, soft bass, no percussion",
    "IDM, fidgety and bright, chopped breaks, bell synth, stuttering edits",
    "liquid drum and bass, spacious, sub bass, jazzy Rhodes, rolling break",
    "breakbeat, euphoric and rough, chopped funk drums, rave stabs, diving bass",
    "synthwave, neon and nocturnal, gated drums, analog lead, arpeggiated bassline",
    "trip hop, smoky and heavy-lidded, dusty drum loop, upright bass, muted trumpet",
    "psytrance, relentless, rolling bassline, acid arpeggio, tight kick",
    "footwork, frantic, chopped soul sample, skittering hats, heavy 808",
    "lo-fi house, tape-saturated, slightly detuned keys, loose drum machine",
    "acid house, rubbery and insistent, 303 bassline, 909 drums, minimal chords",
    "uk garage, swung and glossy, clipped organ stabs, sub bass, shuffling hats",
    "vaporwave, sun-bleached and slow, lounge keys, soft rim clicks, warm bass",
    // hip hop and beats
    "boom bap, dusty and head-nodding, chopped soul sample, upright bass, crisp snare",
    "lo-fi beat, rainy and nostalgic, Rhodes loop, brushed drums, soft bass",
    "trap instrumental, cavernous, 808 slides, triplet hats, sparse minor piano",
    "drill instrumental, menacing, sliding 808, dark string ostinato, skeletal hats",
    "cinematic hip hop, widescreen and aggressive, orchestral stabs, heavy kick, taiko accents",
    "tribal hip hop, aggressive and ritual, hand percussion, low brass, eastern strings",
    "jazz rap instrumental, loose and warm, walking upright bass, muted trumpet, brushed kit",
    // rock and metal
    "post-rock, slow-burning to enormous, clean tremolo guitar, swelling toms, wide cymbals",
    "math rock, angular and bright, interlocking clean guitars, syncopated snare, odd meter",
    "shoegaze instrumental, blurred and sweet, layered fuzz guitar, glacial bends, buried drums",
    "doom metal, tectonic, downtuned riff, crawling bass, cymbal wash",
    "surf rock, reverb-soaked and sunny, twangy guitar, tremolo picking, snappy snare",
    "krautrock, motorik and unbothered, steady eighth-note kit, one-chord organ, fuzz bass",
    "desert rock, sunbaked and loping, fuzzy riff, loose backbeat, tambourine",
    "progressive rock instrumental, restless, organ and guitar trading, shifting time signatures",
    "sludge, tar-thick and slow, detuned riff, feedback, punishing snare",
    "garage rock instrumental, scrappy and urgent, overdriven guitar, rattling drums",
    "psychedelic rock instrumental, kaleidoscopic, phased guitar, hammond organ, loose kit",
    // jazz
    "modal jazz, cool and exploratory, upright bass, brushed drums, muted trumpet, piano comping",
    "spiritual jazz, searching and ecstatic, tenor saxophone, piano, loose ride cymbal",
    "jazz trio, intimate and conversational, piano, upright bass, brushes",
    "bossa nova, sunlit and lilting, nylon guitar, soft snare rim, warm flute",
    "free jazz, fractured and combustible, squalling saxophone, clattering kit, restless bass",
    "cool jazz, unhurried and elegant, flugelhorn, vibraphone, walking bass",
    "jazz funk, greasy and tight, clavinet, slap bass, horn stabs",
    "big band swing, brassy and kinetic, full horn section, walking bass, ride-driven drums",
    // folk and traditional
    "warm fingerpicked acoustic guitar, intimate and unhurried, close-mic'd, room tone",
    "celtic instrumental, wind-scoured, fiddle, tin whistle, hand drum, droning guitar",
    "west african highlife, bright and rolling, clean interlocking guitars, congas, horn lines",
    "flamenco, taut and dramatic, nylon guitar, handclaps, cajon",
    "appalachian old-time, rough-hewn, clawhammer banjo, fiddle, foot stomp",
    "delta blues instrumental, cracked and lonesome, slide guitar, stomping foot, harmonica",
    "klezmer, wheeling and bittersweet, clarinet lead, accordion, upright bass",
    "afro-cuban, bright and driving, montuno piano, congas, horn section",
    // classical and chamber
    "string quartet, restrained and aching, close harmony, long bows",
    "minimalist piano, patient and repetitive, slowly shifting figures, soft pedal",
    "baroque chamber, poised and ornamented, harpsichord, cello, recorder",
    "romantic solo piano, stormy, wide dynamics, dense left hand",
    "brass quintet, ceremonial and bright, interlocking fanfare figures",
    "solo cello, unaccompanied and grave, long phrases, resonant low register",
    "wind ensemble, pastoral, oboe lead, clarinet bed, distant horn",
    // cinematic
    "cinematic score, foreboding and immense, low strings, pulsing timpani, brass swells",
    "horror score, creeping, high sustained strings, low piano, slow timpani",
    "western score, dusty and wide, twangy guitar, lonesome trumpet, sparse percussion",
    "noir score, rain-slicked, muted trumpet, vibraphone, brushed kit",
    "space ambient, weightless, slow pads, distant bell tones, deep drone",
    "documentary underscore, curious and lightly propulsive, marimba, pizzicato strings, soft kit",
    "library music, brisk and businesslike, fuzz organ, tight kit, walking bass",
};

// Same structure, plus a lead vocal type where the guidance puts it, between
// mood and instrumentation -- but only where the voice is the point. Without
// the instrumental toggle yuey sings whatever the prompt says or not, so a
// prompt that names no singer still gets one and leaves the choice to the
// model; the test holds at least a quarter of the bucket to that.
const std::vector<std::string> kVocal = {
    // dance and edm
    "progressive house, euphoric and anthemic, soaring female topline, supersaw chords, big build and drop",
    "future bass, bittersweet and bright, chopped vocal hooks, wide supersaw chords, snappy snare",
    "melodic dubstep, cathartic and huge, emotional female lead, lush pads, half-time drums, growling bass",
    "trance anthem, uplifting and soaring, ethereal female voice, rolling offbeat bass, long breakdown",
    "deep house, late-night and sultry, soulful male vocal, warm chords, rolling bass, shuffled hats",
    "dance-pop, glossy and euphoric, big chorus, pulsing synth bass, four-on-the-floor, bright plucks",
    "electro house, rowdy and bouncy, shouted vocal hook, distorted bass stabs, pounding kick",
    "drum and bass, rushing and euphoric, soulful female vocal, rolling breaks, deep sub bass",
    "uk garage, late-night and flirty, swung beat, chopped vocal hook, organ stabs, sub bass",
    "eurodance, nostalgic and euphoric, belted female chorus, rapped male verses, bright piano riff, pumping bass",
    "nu-disco, glossy and flirtatious, falsetto hooks, slap bass, filtered strings, four-on-the-floor",
    "tropical house, breezy and sunlit, relaxed male vocal, marimba, soft kick, airy pads",
    "afro house, spiritual and rolling, chanted vocals, hand percussion, deep bassline, warm chords",
    "amapiano, smooth and late-night, log drum bass, jazzy piano, shakers, soft pads",
    "techno, dark and hypnotic, spoken vocal fragments, driving kick, acid line",
    "big room house, festival-sized, crowd-chant hook, huge kick, lead synth riff, massive drop",
    "french house, sun-soaked and filtered, vocoder hook, disco guitar, pumping chords",
    "phonk, dark and hazy, chopped vocal samples, cowbell riff, distorted 808",
    "future garage, nocturnal and yearning, pitched vocal chops, swung drums, deep sub, hazy pads",
    "synthpop, cool and wistful, reverb-drenched synths, gated drums, driving bassline",
    // the genre carries it
    "indie rock, restless and bright, jangly guitars, driving bass, crashing cymbals",
    "80s pop, glossy and romantic, gated drums, chorused guitar, lush synth pads",
    "pop punk, bratty and energetic, buzzsaw guitars, pounding drums, gang-shouted chorus",
    "latin pop, sunny and flirtatious, reggaeton beat, nylon guitar, bright synths",
    "k-pop, polished and dynamic, punchy synth bass, switch-up drums, bright synth hooks",
    "country pop, bright and heartfelt, acoustic strumming, pedal steel, stomping kit",
    "emo, raw and confessional, twinkly guitars, loud-quiet dynamics, crashing drums",
    "r&b, smooth and late-night, warm synth bass, trap hats, glassy keys",
    "folk rock, earthy and uplifting, twelve-string guitar, harmonica, steady kit",
    "funk, tight and greasy, slap bass, choppy guitar, horn stabs, clavinet",
    "piano ballad, tender and intimate, solo piano, soft strings",
    "arena rock, anthemic and huge, crunchy guitars, pounding drums, big chorus",
    // voices
    "soul ballad, aching and unhurried, warm female lead, Rhodes, upright bass, brushed drums",
    "gospel, jubilant and full-throated, massed choir, hammond organ, tambourine, clapping",
    "indie folk, hushed and close, breathy male voice, fingerpicked guitar, brushed snare",
    "alt-country, weather-beaten, raspy male baritone, pedal steel, telecaster, shuffling kit",
    "neo soul, languid and intricate, smoky female lead, Rhodes, fretless bass, loose kit",
    "britpop, swaggering and bright, jangling guitars, driving bass, tambourine",
    "dream pop, blurred and lovelorn, airy female voice, chorused guitar, soft drum machine",
    "post-punk, taut and urgent, deadpan male voice, chorus-heavy bass, sharp guitar stabs",
    "motown-style soul, snappy and joyful, female lead with backing trio, punchy bass, tambourine",
    "torch song, smoky and theatrical, low female voice, brushed kit, muted trumpet, piano",
    "power pop, sugar-rush, chiming guitars, handclaps, punchy kit",
    "grunge, ragged and heavy, howling male voice, fuzzed guitar, loose hard-hit drums",
    "riot grrrl punk, snarling and fast, shouted female vocal, buzzsaw guitar, pounding kit",
    "shoegaze, submerged and sweet, half-buried female voice, layered fuzz, washing cymbals",
    "bluegrass, high lonesome, tight vocal harmony, banjo, fiddle, upright bass",
    "sea shanty, hearty and rhythmic, unison male chorus, stomping, concertina",
    "fado, mournful and proud, quavering female voice, portuguese guitar, classical guitar",
    "qawwali-inspired, rising and ecstatic, call-and-response male voices, harmonium, tabla, clapping",
    "afrobeats, buoyant and sleek, smooth male lead, syncopated guitar, log drum, airy synth",
    "highlife, sunny and gliding, warm male voice, interlocking guitars, horn section",
    "reggae, laid-back and sun-warmed, weathered male voice, skanking guitar, deep bass, rimshot",
    "dancehall, punchy, rapid male toasting, sparse digital riddim, heavy sub",
    "samba, rolling and bright, layered female voices, cavaquinho, surdo, agogo bells",
    "cumbia, loping and warm, male lead, accordion, guiro, electric bass",
    "city pop, glossy and wistful, clear female voice, slap bass, glassy Rhodes, saxophone fills",
    "j-pop-influenced, bright and busy, high female voice, dense synths, tight kit",
    "chanson, wry and intimate, spoken-sung male voice, accordion, upright bass",
    "bolero, swooning, rich female voice, nylon guitar, strings, brushed kit",
    "trip hop, narcotic and bruised, breathy female voice, dusty break, sub bass, muted keys",
    "r&b slow jam, velvet and patient, layered male harmonies, warm synth bass, soft hats",
    "hyperpop, gleeful and overdriven, pitched-up voice, clipped synths, blown-out kick",
    "rap over a soul loop, conversational and dense, male lead, upright bass, dusty drums",
    "rap over cinematic strings, urgent, male lead, tense ostinato, sliding 808",
    "conscious rap, measured and warm, male lead, jazz keys, brushed kit, walking bass",
    "folk protest song, plainspoken and insistent, weathered male voice, acoustic guitar, harmonica",
    "murder ballad, grim and narrative, low male voice, banjo, fiddle, sparse kick",
    "spiritual, unaccompanied and grave, massed voices in close harmony, foot stomp",
    "doo-wop, lovestruck and courtly, tenor lead with backing group, brushed kit, walking bass",
    "psychedelic rock, kaleidoscopic, phased guitar, hammond organ, swirling tape delay",
    "prog rock, grandiose, soaring male voice, mellotron, complex kit, shifting meters",
    "symphonic metal, operatic and vast, soprano lead, double kick, full orchestra",
    "black metal, frostbitten and distant, shrieked vocal, tremolo guitar, blast beats",
    "stoner metal, heavy-lidded, fuzzed riff, swinging kit, rumbling bass",
    "electro-pop, cool and precise, analog bass, gated drums, glassy arpeggios",
    "house with a diva vocal, euphoric, belted female lead, piano stabs, four-on-the-floor",
    "garage two-step, skittish and yearning, chopped female vocal, swung sub bass",
    "bedroom pop, lo-fi and shy, doubled soft voice, detuned keys, brushed drum machine",
    "americana duet, road-worn, male and female voices trading lines, pedal steel, brushed kit",
};

std::string json_array(const std::vector<std::string> & items) {
    std::string body = "[";
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (index) body.push_back(',');
        body += json::quote(items[index]);
    }
    body.push_back(']');
    return body;
}

} // namespace

const DicePrompts & dice_prompts() {
    static const DicePrompts pool{kInstrumental, kVocal};
    return pool;
}

std::string dice_prompts_json() {
    const auto & pool = dice_prompts();
    return "{\"dice\":{\"instrumental\":" + json_array(pool.instrumental) +
        ",\"vocal\":" + json_array(pool.vocal) + "}}";
}

} // namespace yue2::server
