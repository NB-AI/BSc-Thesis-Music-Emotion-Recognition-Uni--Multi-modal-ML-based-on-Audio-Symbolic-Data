Coutinho et al., Psychoacoustic cues to emotion in speech prosody and music

- Loudness
    - Dynamic loudness [eGeMAPS is implemented as Zwicker and Fastl; Paper as CHalupper and Fastl 2002] -> Loudness_sma3_...
- Duration
    - Tempo (only music stimuli) -> rhythm.bpm
- Pitch [contour = variation!]
    - Melody contour (only music stimuli) -> lowlevel.pitch_salience.mean, lowlevel.pitch_salience.stdev
    - Spectral flux [eGeMAPS functionals] -> OS: pcm_fftMag_spectralFlux_sma_...
- Timbre
    - Sharpness (Zwicker and Fastl) -> OS: pcm_fftMag_psySharpness_sma_...
    - Sharpness (Aures)
    - Power spectrum centroid -> lowlevel.spectral_centroid.mean, lowlevel.spectral_centroid.stdev
- Roughness
    - Psychoacoustical roughness
    - Auditory dissonance (Hutchinson and Knopoff) -> lowlevel.dissonance.mean, lowlevel.dissonance.stdev
    - Auditory dissonance (Sethares)


Panda et al., Novel Audio Features for Music Emotion Recognition
https://github.com/renatopanda/TAFFC2018/tree/master/data

- FFT spectrum - average power spectrum (median)
- FFT spectrum - skewness (median) -> pcm_fftMag_spectralSkewness_...
- FFT Spectrum - Spectral 2nd moment (median)
- fluctuation (std) -> Fluctuation (std);MIR Toolbox 1.6.1;
- linear spectral pairs (std)
- mfcc1 (std) -> mfcc1_sma3_stddevNorm
- MFCC1 mean ['mfcc1_somesmoothing'] -> mfcc1_sma3_amean
- musical layers (mean)
- rolloff (mean) -> lowlevel.spectral_rolloff.mean, lowlevel.spectral_rolloff.stdev
- roughness (std)
- spectral entropy (std) -> pcm_fftMag_spectralEntropy_sma_...
- spectral skewness (max)
- spectral skewness (std) -> lowlevel.spectral_skewness.mean, lowlevel.spectral_skewness.stdev
- transitions ML0->ML1 per sec (voice)
- transitions ML1 -> ML0 per sec
- tremolo notes in cents (mean)


From Survey:

[5] R. Panda, R. Malheiro, and R.P. Paiva, “Novel Audio Features
for Music Emotion Recognition,” IEEE T. Affect. Comput., 2018
[101] L. Yang, K. Z. Rajab, and E. Chew, “AVA: An interactive system
for visual and quantitative analyses of vibrato and portamento
performance styles,” Proc. 17th Int. Soc. for Music Inf. Retrieval
Conf. (ISMIR 2016), 2016
[103] M. Mauch and M. Levy, “Structural change on multiple time
scales as a correlate of musical complexity,” Proc. 12th Int. Soc.
Music Inf. Retr. Conf. ISMIR 2011, pp. 489–494, 2011.
[104] G. Shibata, R. Nishikimi, E. Nakamura, and K. Yoshii, “Statistical
Music Structure Analysis Based on a Homogeneity-, Repetitiveness-,
and Regularity-Aware Hierarchical Hidden Semi-Markov
Model,” in Proc. of the 20th Int. Soc. Music Inf. Retr.Conf., 2019.

MELODY FEATURES
- Pitch   
    - Pitch: Marsyas, MIR TB, PsySound3, Essentia
    - Virtual Pitch Features: PsySound3
    - Pitch Salience: MIR TB, Essentia
    - Predominant Melody F0: Essentia
    - Pitch Content: Marsyas (unconf.)
- Pitch variation
    - MIDI Note Number stats: [5]
- Pitch range
    - Register Distribution: [5]
- Melodic intervals
    - n.a.
- Melodic direction and contour
    - Note Smoothness stats: [5]
- Melodic movement
    - Ratios of Pitch and Trans.: [5]

HARMONY FEATURES
- Harmonic perception (harmonic intervals)
    - Inharmonicity: MIR TB, Essentia
    - Chromagram: Marsyas, MIR TB, Essentia
    - Chord Sequence: Essentia
- Harmony (tonality)
     - Tuning Frequency: Essentia
    - Key Strength: MIR TB, Essentia
    - Key and Key Clarity: MIR TB, Essentia
    - Tonal Centroid Vector: MIR TB
    - HCDF: PsySound3
    - Sharpness: PsySound3
- Harmony (mode) 
    - Modality: MIR TB, Essentia

RHYTHM FEATURES
- Tempo
    - Beat Spectrum: MIR TB
    - Beat Location: Marsyas, Essentia
    - Onset Time: MIR TB, Essentia
    - Event Density: MIR TB
    - Average Duration of Events: MIR TB
    - Tempo: Marsyas, MIR TB, Essentia
    - PLP Novelty Curves: Essentia
    - HWPS: Marsyas
- Tempo and Note Values
    - Metrical Structure: MIR TB
    - Metrical Centroid and Strength: MIR TB
    - Note Duration statistics: [5]
    - Note Duration Distribution: [5]
    - Ratios of Note Duration Transitions: [5]
- Rhythm Types
    - Rhythmic Fluctuation: MIR TB
    - Tempo Change: MIR TB
    - Pulse / Rhythmic Clarity: MIR TB, Essentia
    - Rests n.a. n.a.

DYNAMICS FEATURES
- Dynamic levels (forte, piano, etc.)
    - RMS Energy: Marsyas, MIR TB, Essentia
    - Low Energy Rate: Marsyas, MIR TB
    - Sound Level: PsySound3
    - Instantaneous Level, Freq. and Phase: PsySound3
    - Loudness: PsySound3, Essentia
    - Timbral Width: PsySound3
    - Volume: PsySound3
    - Sound Balance: MIR TB, Essentia
    - Note Intensity statistics: [5]
    - Note Intensity Distribution: [5]
- Accents and changes in dynamic levels
    - Ratios of Note Intensity Transitions: [5]
    - Crescendo and Decrescendo metrics [5]

TONE COLOR (TIMBRE) FEATURES
- Amplitude envelope
    - Attack/Decay Time: MIR TB, Essentia
    - Attack/Decay Slope: MIR TB
    - Attack/Decay Leap: MIR TB
    - Zero Crossing Rate: Marsyas, MIR TB, Essentia
- Spectral envelope (no. harmonics)
    - Spectral Flatness: Marsyas, MIR TB, Essentia
    - Spectral Crest Factor: Marsyas
    - Irregularity: MIR TB
    - Tristimulus: Essentia
    - Odd-to-even harmonic energy ratio: Essentia
- Spectral characteristics (e.g., spectral centroid)
    - Spectral Centroid: Marsyas, MIR TB, PsySound3, Essentia
    - Spectral Spread: MIR TB, PsySound3, Essentia
    - Spectral Skewness: MIR TB, PsySound3, Essentia
    - Spectral Kurtosis: MIR TB, PsySound3, Essentia
    - Spectral Entropy: MIR TB, Essentia
    - Spectral Flux: Marsyas, MIR TB, Essentia
    - Spectral Rolloff: Marsyas, MIR TB, Essentia
    - High-frequency Energy: MIR TB, Essentia
    - Cepstrum (Real/Complex): PsySound3
    - Energy in Mel/Bark/ERB Bands: MIR TB, PsySound3, Essentia
    - MFCCs: Marsyas, MIR TB, Essentia
    - LPCCs: Marsyas, Essentia
    - Linear Spectral Pairs: Marsyas
    - Spectral Contrast: Essentia
    - Roughness: MIR TB, PsySound3, Essentia
    - Spectral and Tonal Dissonance: PsySound3

EXPRESSIVITY FEATURES
 - Articulation
    - Average Silence Ratio: MIR TB
    - Articulation metrics: [5]
- Ornamentation
    - Glissando metrics: [5]
    - Portamento metrics: [101]
- Vibrato
    - Vibrato metrics: [5, 101]
- Tremolo
    - Tremolo metrics: [5]

TEXTURE FEATURES
- Number of layers and density
    - Musical Layers statistics: [5]
    - Musical Layers Distribution: [5]
    - Ratio of Musical Layers Transitions: [5]
- Texture type
    - n.a.: n.a.

FORM FEATURES
- Form Complexity
    - Structural Change: [103]
- Organization Levels
    - Similarity Matrix: MIR TB
    - Novelty Curve: MIR TB
- Song Elements
    - Higher-Level Form Analysis: [104-106]

VOCAL FEATURES
- All Features from the Vocals Channel: [5]
- Voiced and Unvoiced statistics: [5]
- Creaky Voice statistics: [5]

HIGH-LEVEL FEATURES
- Emotion: MIR Toolbox, Essentia
- Classification-based Feat. (genre, etc.): Essentia
- Danceability: Essentia
- Dynamic Complexity: Essentia