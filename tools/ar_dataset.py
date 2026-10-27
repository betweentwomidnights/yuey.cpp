"""Pure dataset planning, caption and score normalization for AR training."""
from fractions import Fraction
import math
import re


def caption_metadata(text):
    normalized=text.replace('\u266d','b').replace('\u266f','#')
    bpm=re.search(r'\b(\d+(?:\.\d+)?)\s*bpm\b',normalized,re.I)
    key=re.search(r'\b([A-G](?:#|b)?)\s+(major|minor)\b',normalized,re.I)
    if not bpm or not key:raise ValueError('Caption needs explicit tempo and key')
    tempo=float(bpm[1]);tonic=key[1][0].upper()+key[1][1:]
    if not 20<=tempo<=400:raise ValueError('Unsupported caption tempo')
    style=', '.join(part.strip() for part in normalized.split(',') if part.strip() and not re.fullmatch(r'\d+(?:\.\d+)?\s*bpm|[A-G](?:#|b)?\s+(?:major|minor)',part.strip(),re.I))
    return dict(bpm=tempo,key=tonic+('m' if key[2].lower()=='minor' else ''),style=style)


def window_starts(total_frames,frames,bpm):
    """Cover a recording with real, bar-aligned cores; end cores may overlap."""
    if frames<=0 or total_frames<frames:raise ValueError('Recording shorter than training window')
    bar_frames=240/bpm*25
    starts={0}
    for at in range(frames,total_frames,frames):
        candidate=round(math.floor(min(at,total_frames-frames)/bar_frames)*bar_frames)
        starts.add(max(0,min(total_frames-frames,candidate)))
    return sorted(starts)


def abc_pitch(pitch):
    if not 0<=pitch<=127:raise ValueError('MIDI pitch out of range')
    names=['=C','^C','=D','^D','=E','=F','^F','=G','^G','=A','^A','=B']
    octave=pitch//12-1;name=names[pitch%12]
    if octave>=5:return name.lower()+"'"*(octave-5)
    return name+','*max(0,4-octave)


def chord_name(text):
    if not text or text=='N':return ''
    if ':' not in text:return text
    root,kind=text.split(':',1)
    kind,separator,inversion=kind.partition('/')
    suffix={'maj':'','min':'m','dim':'dim','aug':'aug','7':'7','maj7':'maj7','min7':'m7','dim7':'dim7','hdim7':'m7b5','sus4':'sus4','sus2':'sus2','maj6':'6','min6':'m6','sus4(b7)':'7sus4','minmaj7':'m(maj7)'}
    if kind not in suffix:return ''
    result=root+suffix[kind]
    intervals={'2':2,'b3':3,'3':4,'5':7,'b7':10,'7':11}
    if separator and inversion in intervals:
        pitch={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[root[0]]+({'#':1,'b':-1}.get(root[1:],0))
        names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        result+='/'+names[(pitch+intervals[inversion])%12]
    return result


def duration_suffix(ticks):
    value=Fraction(ticks,3)  # L:1/16, 12 grid ticks per quarter.
    if value==1:return ''
    return str(value.numerator) if value.denominator==1 else f'{value.numerator}/{value.denominator}'


def score_from_events(events,seconds,bpm,key):
    """Re-engrave absolute times/pitches, rather than replacing Q/K headers.

    A 12-tick quarter grid admits both sixteenths and eighth-note triplets.
    Accidentals are explicit, including naturals, so key cannot reinterpret
    absolute note pitches. Both native melody voices are retained for the
    separately validated instrumental transfer step.
    """
    if not math.isfinite(seconds) or seconds<=0 or not 20<=bpm<=400:raise ValueError('Invalid score timing')
    if not re.fullmatch(r'[A-G][#b]?m?',key):raise ValueError('Unsupported key')
    numerator=denominator=4
    for event in events:
        if event.get('meter_numerator',0)>0 and event.get('meter_denominator',0)>0:
            numerator=event['meter_numerator'];denominator=event['meter_denominator'];break
    if denominator not in [2,4,8,16] or not 1<=numerator<=12:raise ValueError('Unsupported meter')
    bar_ticks=Fraction(numerator*48,denominator)
    if bar_ticks.denominator!=1:raise ValueError('Unsupported meter grid')
    bar_ticks=int(bar_ticks);scale=bpm/60*12
    limit=math.floor(seconds*scale+1e-6);total=math.ceil(max(1,limit)/bar_ticks)*bar_ticks
    chords={};voices={0:[],1:[]};rounding=[];invalid=0;meter_changes=set()
    for event in events:
        start=float(event['time_seconds'])
        if not math.isfinite(start):raise ValueError('Nonfinite onset')
        at=max(0,round(start*scale))
        if event.get('meter_numerator',0):meter_changes.add((event['meter_numerator'],event['meter_denominator']))
        chord=chord_name(event.get('chord',''))
        if at<limit and chord:chords[at]=chord
        for note in event['notes']:
            end=float(note['end_time_seconds']);pitch=int(note['pitch']);track=int(note['track'])
            if not math.isfinite(end) or end<start or not 0<=pitch<=127 or track not in voices:
                raise ValueError('Invalid note event')
            if start>=seconds or end<=0:continue
            finish=min(limit,max(at+1,round(min(end,seconds)*scale)))
            if at>=finish:invalid+=1;continue
            rounding.extend([abs(at/scale-max(0,start)),abs(finish/scale-min(end,seconds))])
            voices[track].append(dict(start=at,end=finish,pitch=pitch))
    if len(meter_changes)>1:raise ValueError('Changing meter needs explicit review')
    lines=['X:1','T:','M:'+str(numerator)+'/'+str(denominator),'L:1/16','Q:1/4='+str(round(bpm)),
           'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"','V: Ins clef=treble name="Ins Melody" snm="Inst."','K:'+key]
    retained=duplicates=0
    for voice,name in [(0,'Vocal'),(1,'Ins')]:
        notes=sorted(voices[voice],key=lambda n:n['start']);mono=[]
        for note in notes:
            if mono and mono[-1]['start']==note['start']:duplicates+=1;continue
            mono.append(dict(note))
        for a,b in zip(mono,mono[1:]):a['end']=min(a['end'],b['start'])
        retained+=len(mono);cells=[None]*total
        for i,note in enumerate(mono):
            for at in range(note['start'],note['end']):cells[at]=(note['pitch'],i)
        bars=[]
        for begin in range(0,total,bar_ticks):
            end=begin+bar_ticks;at=begin;body=[]
            while at<end:
                cell=cells[at];finish=at+1
                while finish<end and cells[finish]==cell and (voice!=0 or finish not in chords):finish+=1
                annotation=('"'+chords[at]+'"') if voice==0 and at in chords else ''
                pitch='z' if cell is None else abc_pitch(cell[0])
                tie='-' if cell is not None and finish<total and cells[finish]==cell else ''
                body.append(annotation+pitch+duration_suffix(finish-at)+tie);at=finish
            bars.append(''.join(body)+'|')
        for index in range(0,len(bars),4):lines.extend(['V: '+name,''.join(bars[index:index+4])])
    stats=dict(input_notes=sum(len(v) for v in voices.values()),retained_monophonic_notes=retained,duplicate_onsets_dropped=duplicates,
               clipped_notes=invalid,chord_positions=len(chords),max_rounding_seconds=max(rounding,default=0),
               max_expected_grid_error_seconds=1/scale,score_seconds=total/scale,audio_seconds=seconds,
               bpm=bpm,key=key,meter=f'{numerator}/{denominator}',grid='12 ticks/quarter; explicit pitches; real absolute timestamps',
               manual_note_accuracy_verified=False)
    return '\n'.join(lines)+'\n',stats
