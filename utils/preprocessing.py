from scipy import signal
import librosa
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import os
from collections import OrderedDict
from itertools import chain


def import_raw_audio(filename, indir, sr=None):
    import warnings
    import librosa.util.exceptions
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa.core.audio')
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa.util.decorators')
      
    t, sr = librosa.load(indir + filename, sr=sr, mono=True)
    duration = t.shape[0] / sr  # in seconds
    mu_t = t.mean()
    min_t = t.min()
    max_t = t.max()
    # tnorm = (t - mu_t )
    # tnorm = tnorm / (max_t-mu_t)
    f_token = np.array([filename[:-4]]).reshape(1, -1)
    tokens = np.array([sr, duration, mu_t, max_t, min_t]).reshape(1, -1)
    audio_df = pd.DataFrame(
        data=np.hstack((f_token, tokens)),
        columns=[
            "AUDIO_FILE",
            "SAMPLING_RATE",
            "DURATION",
            "MEAN_SIG",
            "MAX_SIG",
            "MIN_SIG",
        ],
    )
    audio_df["SAMPLING_RATE"] = (
        audio_df["SAMPLING_RATE"].astype(float).astype(int)
    )  # weird conversion from string to int
    for i in ["DURATION", "MEAN_SIG", "MAX_SIG", "MIN_SIG"]:
        audio_df[i] = audio_df[i].astype(float)

    return audio_df, t, sr


def zero_padding(t, sr, target_duration):
    """do zero-padding to get audio files all of the same duration;
    this will allow us to have spectrograms all of the same size"""
    target_len = target_duration * sr
    if t.shape[0] > target_len:
        t = t[0:target_len]
    elif t.shape[0] < target_len:
        n_pads = target_len - t.shape[0]
        t = np.append(t, np.repeat(0, n_pads))
    else:
        pass
    return t


def calc_stft_power_spectrum(stft, sr, n_fft):
    amplitudes = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr, n_fft)
    psx = amplitudes.mean(axis=-1)
    return frequencies, np.sqrt(psx)


def calc_power_spectrum_welch(t, sr, n_fft):
    f, psx = signal.welch(
        t, sr, window="hann", 
        nfft=n_fft, 
        noverlap=0, 
        axis=-1, 
        scaling="spectrum"
    )
    return f, np.sqrt(psx)


def calc_spectral_features(
    t, sr, n_fft=512, win_length=None, win_overlap=0.0, n_mfcc=None, rec_width=0
):
    """ 
    Calculate spectrograms:
    -) Short-time Fourier transform (STFT) for the power spectrum
    -) Mel-frequency cepstral coefficients (MFCC)

    win_overlap: float, [0.0, 1.0] ; if 0.0, windows will be NOT overlapping
    """

    if win_length is None:
        win_length = n_fft

    if n_mfcc is None:
        n_mfcc = n_fft

    assert (win_overlap >= 0) & (
        win_overlap < 1.0
    ), "Invalid value of win_overlap {} - it must be in range [0.0, 1.0) ".format(
        win_overlap
    )
    hop_length = int(win_length * (1.0 - win_overlap))

    stft = librosa.stft(y=t, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mfcc = librosa.feature.mfcc(y=t, sr=sr, n_mfcc=n_mfcc, dct_type=2)

    return stft, mfcc


def stack_rows_with_pad(list_of_arrays):
    f1 = lambda x: x.shape[1]
    max_dim = max(list(map(f1, list_of_arrays)))
    padded_arrays = [
        np.append(m, np.full([m.shape[0], max_dim - m.shape[1]], np.nan), axis=1)
        for m in list_of_arrays
    ]
    return np.concatenate(padded_arrays, axis=0)


def calc_spectral_properties_welch(t, sr, n_fft, time_window_ms, freq_bins):
    """
        Computes a whole bunch of spectral properties, after the reference (see section III.A)
        https://myresearchspace.uws.ac.uk/ws/files/10993506/2018_12_15_Monge_Alvarez_et_al_Cough.pdf

        It splits the audio signal in smaller chunks. For each chunk computes the Power Spectrum Density
        using the Welch method. It then averages the power for user-defined frequency bands.
        At that point, we have many subsegments of the audio, k, and many average PSD, j
        The spectral properties are calculated averaging and summing over k.
        Output is a dictionary with various spectral properties, each one replicated j times
        (as many as the frequency bands).
    """

    # sanity checks
    assert (
        len(freq_bins) > 1
    ), "Error, input freq_bins must be a list with the boundaries of the frequency bins"

    # define how many ms is long each sample of the audio signal
    # and how many values go in each subsegment

    n_samples_tot = len(t)
    if time_window_ms is None:
        time_window_ms = 1000 * n_samples_tot / sr
    chunk_length = min(
        n_samples_tot, round(time_window_ms * sr / 1000)
    )  
    # how many audio samples fit in time_window_ms
    n_chunks = int(np.ceil(n_samples_tot / chunk_length))
    n_freq_bins = len(freq_bins) - 1
    out_all_freq = np.empty((n_freq_bins, 0), float)
    out_all_psx = np.empty((n_freq_bins, 0), float)

    for k in range(0, n_chunks, 1):
        tmin = k * chunk_length
        tmax = min((k + 1) * chunk_length, n_samples_tot)
        tmp_segment = t[tmin:tmax]
        freqs_welch, psx_welch = calc_power_spectrum_welch(tmp_segment, sr, n_fft)

        chunk_freq = np.empty((1, 0), float)
        chunk_psx = np.empty((1, 0), float)

        for j in range(0, n_freq_bins, 1):
            freqmin = freq_bins[j]
            freqmax = freq_bins[j + 1]
            freq_mask = (freqs_welch >= freqmin) & (freqs_welch < freqmax)
            selfreqs = freqs_welch[freq_mask]
            selpsx = psx_welch[freq_mask]
            # print("{} {} |||  {} {} {}".format(k,j,chunk_freq.shape,selfreqs.shape, selfreqs.reshape(1,-1).shape,selpsx.shape))
            if j == 0:
                chunk_freq = selfreqs.reshape(1, -1)
                chunk_psx = selpsx.reshape(1, -1)
            else:
                chunk_freq = stack_rows_with_pad([chunk_freq, selfreqs.reshape(1, -1)])
                chunk_psx = stack_rows_with_pad([chunk_psx, selpsx.reshape(1, -1)])

        out_all_freq = np.append(out_all_freq, chunk_freq, axis=1)
        out_all_psx = np.append(out_all_psx, chunk_psx, axis=1)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        t, frame_length=chunk_length, hop_length=chunk_length + 1
    )

    # spectral centroid
    psx_sum = np.nansum(out_all_psx, axis=1)
    spec_centroid = (np.nansum(out_all_freq * out_all_psx, axis=1) / psx_sum).reshape(
        -1, 1
    )

    # spectral bandwidth
    spec_bw = (
        np.nansum(((out_all_freq - spec_centroid) ** 2) * out_all_psx, axis=1) / psx_sum
    )

    # spectral crest factor
    # C = 1.0 / (np.nanmax(out_all_freq) - np.nanmin(out_all_freq) +1)
    # spec_crest = (np.nanmax(out_all_psx)/(C*psx_sum) ).reshape(-1,1)
    psx_25 = np.nanquantile(out_all_psx, 0.25, axis=1)
    psx_50 = np.nanquantile(out_all_psx, 0.50, axis=1)
    psx_75 = np.nanquantile(out_all_psx, 0.75, axis=1)
    psx_max = np.nanmax(out_all_psx, axis=1)
    # print("MAX: {} ; P25: {} ; P50: {} ; P75: {}".format(psx_max, psx_25, psx_50, psx_75))
    spec_crest = (psx_max - psx_50) / (psx_75 - psx_25)

    # spectral standard deviation
    spec_sd = np.nanstd(out_all_psx, axis=1)

    # spectral skewness
    n_entries = np.array(
        [len(row[~np.isnan(row)]) for row in out_all_psx]
    )  # .reshape(-1,)
    skew_factors = [e * np.sqrt(e - 1) / (e - 2) for e in n_entries]
    spec_mean = np.nanmean(out_all_psx, axis=1).reshape(-1, 1)
    spec_skew = (
        skew_factors * np.nansum((out_all_psx - spec_mean) ** 3, axis=1) / spec_sd**3
    )

    return (
        zcr,
        spec_centroid.reshape(1, -1),
        spec_bw.reshape(1, -1),
        spec_crest.reshape(1, -1),
        spec_mean.reshape(1, -1),
        spec_sd.reshape(1, -1),
        spec_skew.reshape(1, -1),
    )


def prepare_data(
    input_data,
    audio_datadir,
    sr,
    target_duration,
    n_fft,
    n_mfcc,
    fft_window_size,
    psd_freq_bins,
    mfcc_feature_names,
    psd_feature_names,
    max_audio_samples=None,
    print_every_n=10,
    tmp_metadata = None
):
    """
    Prepares a dataframe with a collection of properties and sound features
    that can be readily used later in a ML classification process

    input_data: pandas data.frame; an extract of the metadata file present in the original dataset

    audio_datadir: string; the path to the diretory where the audio files are stored

    sr: int; sampling rate

    target_duration: int; final length of audio sample, in seconds. All audio files will be formatted
                    to this duration; longer audios will be cut; shorter audios will be padded with zeros

    n_fft: int; number of frequency bins to be considered in the Fast Fourier Transform

    n_mfcc: int; number of Mel-freuqencies to be used when computing the MFCC

    max_audio_samples: int; maximum number of audio files to be processed. If None, all available UUIDs
                    will be processed; otherwise, only the first max_audio_sample UUID will be considered


    Output: The output of this loop is a big pandas dataframe with as many rows as audio files
            and as many columns as a series of audio features.
            The column list includes also the audio UUID and the sample label (the "status" column in the metadata file).
    """

    my_n_fft = 512
    
    # get the full list of uuid to be processed
    all_uuids = input_data["uuid"].values
    if max_audio_samples is not None:
        all_uuids[0:max_audio_samples]

    # empty pandas df where to store all features for all UUIDs
    all_data = pd.DataFrame()

    # init  timer and df containig some metadata of the audio files
    skipped_uuids = []
    audio_metadata = pd.DataFrame()
    t_start = timer()
    
    # Espectrogramas 
    spectrograms_data = {} 

    for idx, uuid in enumerate(all_uuids):

        tmp_audiofilename = uuid + ".webm"
        if not os.path.exists(audio_datadir + tmp_audiofilename):
            # try to look for a .ogg file
            tmp_audiofilename = uuid + ".ogg"
            if not os.path.exists(audio_datadir + tmp_audiofilename):
                print(
                    "WARNING! Could not find audio file for UUID: {}  . Skipping.".format(
                        uuid
                    )
                )
                continue

        if idx % print_every_n == 0:
            print()
            #print("Processing file #{}: {}".format(idx, tmp_audiofilename))

        try:
            tmp_df, tmp_audio, sr = import_raw_audio(
                tmp_audiofilename, indir=audio_datadir, sr=sr
            )
        except FileNotFoundError as e_fnf:
            print("Could not find audio file {}.\n\n\n".format(tmp_audiofilename))
            skipped_uuids = skipped_uuids + [uuid]
            continue  # move to next file
        except Exception as e:
            print("Some other exception occurred")
            raise e  # rethrow exception

        tmp_audio = zero_padding(tmp_audio, sr=sr, target_duration=target_duration)
        tmp_df["UUID"] = uuid
        audio_metadata = pd.concat([audio_metadata, tmp_df], ignore_index=True)

        stft, mfcc = calc_spectral_features(
            tmp_audio,
            sr,
            n_fft=n_fft,
            n_mfcc=n_mfcc,
            win_length=fft_window_size,
            win_overlap=0.0,
        )
        print(f"STFT DIMENSIONS:{stft.shape}")
        # Guardar espectrogramas
        spectrograms_data[uuid] = {
            'stft': {
                'magnitude': np.abs(stft),
                'phase': np.angle(stft).tolist(),
                'frequencies': librosa.fft_frequencies(sr=sr, n_fft=n_fft).tolist(),
                'times': librosa.times_like(stft, sr=sr).tolist()
            },
            'mfcc': {
                'coefficients': mfcc.tolist(),
                'n_mfcc': n_mfcc,
                'sr': sr
            },
            'metadata': {
                'uuid': uuid,
                'filename': tmp_audiofilename,
                'duration': target_duration,
                'sr': sr,
                'n_fft': n_fft
            }
        }

        # extract mean and std dev for each mel-frequency in the mfcc
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_sd = np.std(mfcc, axis=1)
        mfcc_features = np.append(mfcc_mean, mfcc_sd, axis=0)
        mfcc_feat_dict = {
            name: val for name, val in zip(mfcc_feature_names, mfcc_features)
        }

        # Power Spectrum Density based short-term features
        zcr, sc, sb, scf, ssmean, ssd, ssk = calc_spectral_properties_welch(
            tmp_audio, sr, my_n_fft, None, psd_freq_bins
        )
        # consider only every second bin to reduce features; following original article freq bins
        psd_features = np.array(
            [
                (x0, x1, x2, x3, x4, x5)
                for i, (x0, x1, x2, x3, x4, x5) in enumerate(
                    zip(*sc, *sb, *scf, *ssmean, *ssd, *ssk)
                )
                if i % 2 == 0
            ]
        ).transpose()

        # now extract each element of the PSD feature (correspondignto a unique combination of spectral feature and freq bin)
        n_freq_bins = psd_features.shape[1]
        psd_features = psd_features.ravel()
        psd_feature_names_expanded = [
            ["{f}_{b:02}".format(f=f, b=b) for b in range(0, n_freq_bins, 1)]
            for f in psd_feature_names
        ]
        psd_feature_names_expanded = list(
            chain.from_iterable(psd_feature_names_expanded)
        )
        assert (
            len(zcr) == 1
        ), "Zero-Crossing Rate vector has length different from 1: {}".format(len(zcr))
        assert len(psd_feature_names_expanded) == len(
            psd_features
        ), "Mismatch between number of spectral features ({nf}) and vector with their names ({nn})".format(
            nf=len(psd_features), nn=len(psd_feature_names_expanded)
        )
        psd_feat_dict = {
            name: val for name, val in zip(psd_feature_names_expanded, psd_features)
        }
        psd_feat_dict["ZCR"] = zcr[0, 0]

        # store all features in a pandas dataframe
        tmp_df = input_data.loc[
            tmp_metadata["uuid"] == uuid,
            [
                "uuid",
                "audio_class",
                "cough_detected",
                "SNR",
                "age",
                "gender",
                "respiratory_condition",
                "fever_muscle_pain",
                "status",
            ],
        ]
        tmp_df.columns = [c.upper() for c in tmp_df.columns]
        tmp_dict = tmp_df.to_dict(orient="records")
        # assert len(tmp_dict)==1, "ERROR! Multiple records for UUID {} : {}".format(uuid,len(tmp_dict))
        tmp_dict = OrderedDict(tmp_dict[0])
        tmp_dict.update(mfcc_feat_dict)
        tmp_dict.update(psd_feat_dict)
        # tmp_df = pd.DataFrame(tmp_dict, columns=tmp_dict.keys())
        all_data = pd.concat([all_data, pd.DataFrame(tmp_dict, index=[idx])], ignore_index=True)

        spectrograms_data[uuid]['features'] = tmp_dict
    # end for loop over raw audio files
    #print("\n{} files processed in {:.1f} seconds\n".format(idx + 1, timer() - t_start))
    return all_data, audio_metadata, spectrograms_data


def sample_df_balanced(df, group_col, n, random=42):
    assert isinstance(
        group_col, str
    ), "Input group_col must be a plain string with the column name: {}".format(
        type(group_col)
    )
    # df_count = df[[group_col]].groupby([group_col]).cumcount()+1
    df["N"] = np.zeros(len(df[group_col]))
    df_count = (
        df[[group_col, "N"]].groupby([group_col]).count().reset_index()
    )  # cumcount()+1

    out_df = pd.DataFrame()
    for igroup in df[group_col].unique():

        n_orig = df_count.loc[df_count[group_col] == igroup, "N"].values[0]
        if n_orig < n:  # need to upsample
            delta = max(n - n_orig, 0)
            tmp_df = df.loc[df[group_col] == igroup,]
            delta_df = tmp_df.sample(n=delta, random_state=random, replace=False)
            out_df = pd.concat([out_df, tmp_df, delta_df])
        else:  # downsample
            tmp_df = df.loc[df[group_col] == igroup,].sample(
                n=n, random_state=random, replace=False
            )
            out_df = pd.concat([out_df, tmp_df])
    # end for loop over groups
    return out_df.drop("N", axis=1, inplace=False)
